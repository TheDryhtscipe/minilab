"""The model adapter: the one way minilab calls a model.

`call` enforces the rules every model call must follow (design doc-gmj,
"Constraints" and "Generator"):

- Gemini generates, and makes proof attempts in the triviality screen
  (`Role.screen`) and the prover loop (`Role.prove`). Lean checks every
  proof, so a bad one proves nothing.
  Any other role for `Family.gemini` is refused (`GEMINI_ROLES`), the
  screen's corollary check (`Role.corollary`) included.
- The pair's token budget is checked before the call. Inside one
  `Ledger.pair_lock` the adapter checks the pair's spend, which counts the
  reservations of calls still in flight, and reserves tokens for this call:
  the prompt's estimate plus `Budget.reserve_output_tokens`. Then it releases
  the lock and makes the call, so other calls on the pair can run at the same
  time. Afterwards the call's charge settles the reservation. The lock does
  not nest: never call `call` while holding the pair's lock.
  A call is refused once the pair's spend is at or over its budget. So the
  pair can overrun its budget by the last admitted call's tokens, plus, for
  each call that was in flight with it, what that call spent beyond its
  reservation.
- Every call runs isolated: a fresh empty temporary working directory, removed
  afterwards, and an environment reduced to an allowlist, so no
  `ANTHROPIC_*` or other credential or tool-routing variable reaches the tool.
- Every call that started is charged to its pair, including a failed one.
  A call that failed after the tool started (any exception from the backend)
  is charged as an estimate: input is the larger of the tokens the tool
  reported and the prompt's estimate; output is what the tool reported or,
  if it reported none, the call's output allowance
  (`Budget.reserve_output_tokens`), so that a failed call never frees the
  output it reserved. Then the exception is raised again. A call that failed before the
  tool started (`ModelNotStarted`) used nothing: its reservation is released
  and it is not charged. If the charge or the release itself fails, the
  reservation stays open and keeps counting as spend.
  The adapter never writes to a record, so a failure leaves the record as it
  was and the run can resume.

`pick_checker` chooses the family that checks another family's work.
"""

import os
import shutil
import stat
import tempfile
import time
import warnings
from collections.abc import Callable, Iterable, Mapping
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from minilab.ledger import MAX_TOKENS, Ledger
from minilab.models.backends import (
    DEFAULT_TIMEOUT_S,
    Backend,
    ClaudeBackend,
    CodexBackend,
    GeminiBackend,
    LocalBackend,
    ModelNotStarted,
    ModelReply,
    ModelUnavailable,
    estimate_tokens,
)
from minilab.records import Family, Stage, check_record_id, run_of
from minilab.run import RunManifest
from minilab.scheduler import Pair, canonical_pair

__all__ = [
    "ENV_ALLOWLIST",
    "GEMINI_ROLES",
    "ROLE_STAGE",
    "Budget",
    "BudgetExceeded",
    "ModelNotStarted",
    "ModelReply",
    "ModelUnavailable",
    "NoChecker",
    "Role",
    "RoleNotAllowed",
    "call",
    "backends_for",
    "default_backends",
    "isolated_run",
    "pick_checker",
    "scrubbed_env",
]


class Role(Enum):
    """A model call's job.

    A plain `Enum`, not a `StrEnum`: pyright in strict mode treats a member
    named `encode` as an incompatible override of `str.encode`. (At run time a
    `StrEnum` would work.)
    """

    generate = "generate"
    encode = "encode"
    reformulate = "reformulate"
    mechanism = "mechanism"
    formalise = "formalise"
    roundtrip = "roundtrip"
    audit = "audit"
    literature = "literature"
    prove = "prove"
    screen = "screen"
    """A proof attempt in the triviality screen."""
    corollary = "corollary"
    """The triviality screen's corollary check: a claim and its proof."""
    curate = "curate"
    grade = "grade"


ROLE_STAGE: dict[Role, Stage] = {
    Role.generate: Stage.generator,
    Role.encode: Stage.encoding,
    Role.reformulate: Stage.falsifier,
    Role.mechanism: Stage.falsifier,
    Role.formalise: Stage.formaliser,
    Role.roundtrip: Stage.formaliser,
    Role.audit: Stage.formaliser,
    Role.literature: Stage.literature,
    Role.prove: Stage.prover,
    Role.screen: Stage.triviality,
    Role.corollary: Stage.triviality,
    Role.curate: Stage.queue,
    Role.grade: Stage.queue,
}
"""The stage each role's cost is charged to. Curating and grading the benchmark
have no pipeline stage of their own; they are charged to the queue. They are
meant to run under their own run id and manifest, not a benchmark run's ledger."""

GEMINI_ROLES: frozenset[Role] = frozenset({Role.generate, Role.screen, Role.prove})
"""The roles Gemini may take. `Role.screen` is a proof attempt in the
triviality screen: Lean checks the proof in proved mode, and the proved file
must match the original, so a wrong proof cannot mark a record trivial.
Gemini checks nothing: no encoding, audit, round trip, formalisation,
literature check, or corollary claim (user decision 2026-11-26). `Role.prove`
is a proof attempt in the prover loop, checked the same way; a proof that adds
hypotheses goes to the audit, whose checker is never Gemini (user decision on
bead minilab-880beb, 2026-11-26)."""

ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "TZ",
        "TMPDIR",
        "XDG_RUNTIME_DIR",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "XDG_STATE_HOME",
        "DBUS_SESSION_BUS_ADDRESS",
    }
)
"""The only variables a tool sees. They locate its binary, the user's home
directory and XDG directories (where each tool keeps its default login), the
OS keyring over D-Bus, and the locale, time zone, and temporary directory.
Every other variable is dropped: every `ANTHROPIC_*` variable, any API key,
and variables that pick another account or another TLS trust store
(`CLAUDE_CONFIG_DIR`, `CODEX_HOME`, `SSL_CERT_FILE`, `SSL_CERT_DIR`). A tool
can still read its default login and settings under `HOME`."""


class BudgetExceeded(Exception):
    """The pair's token spend is at or over its budget; no call was made."""


class RoleNotAllowed(Exception):
    """The family may not take this role (Gemini only generates and makes proof attempts)."""


class NoChecker(Exception):
    """No family other than the author, and other than Gemini, is available to check."""


class Budget(BaseModel):
    """The token budget for one domain pair: every model call for the pair counts.

    It may be smaller than the run manifest's budget, never larger.
    `reserve_output_tokens` is the output each call reserves on top of its
    prompt's estimate while it runs. A call that writes more than this can
    push the pair further over its budget when other calls run beside it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tokens_per_pair: int = Field(gt=0)
    reserve_output_tokens: int = Field(default=32_000, ge=0, le=MAX_TOKENS)

    @classmethod
    def from_manifest(cls, manifest: RunManifest) -> "Budget":
        return cls(tokens_per_pair=manifest.budget_tokens_per_pair)


def scrubbed_env() -> dict[str, str]:
    """Return the current environment reduced to `ENV_ALLOWLIST`."""
    return {name: value for name, value in os.environ.items() if name in ENV_ALLOWLIST}


def backends_for(models: Mapping[Family, str]) -> dict[Family, Backend]:
    """Build a backend for each family in `models`, asking for its model id."""
    makers: dict[Family, Callable[[str], Backend]] = {
        Family.claude: lambda model: ClaudeBackend(model=model),
        Family.codex: lambda model: CodexBackend(model=model),
        Family.gemini: lambda model: GeminiBackend(model=model),
        Family.local: lambda model: LocalBackend(model=model),
    }
    return {family: makers[family](model) for family, model in models.items()}


def default_backends(manifest: RunManifest) -> dict[Family, Backend]:
    """Build a backend for each of the manifest's families, asking for its model id."""
    return backends_for({f: manifest.model_of(f) for f in manifest.families})


def _remove_dir(path: Path) -> None:
    """Remove `path`, making read-only parts writable as needed. Warn if it stays."""

    def make_writable_and_retry(
        func: Callable[[str], object], failed: str, exc: BaseException
    ) -> None:
        for target in (os.path.dirname(failed), failed):
            try:
                if not stat.S_ISLNK(os.lstat(target).st_mode):
                    os.chmod(target, 0o700)
            except OSError:
                pass
        try:
            func(failed)
        except OSError:
            pass

    shutil.rmtree(path, onexc=make_writable_and_retry)
    if path.exists():
        # A second pass reaches directories the first pass made readable.
        shutil.rmtree(path, onexc=make_writable_and_retry)
    if path.exists():
        warnings.warn(f"could not remove the call's working directory {path}", stacklevel=3)


def isolated_run(backend: Backend, prompt: str, *, timeout_s: float) -> ModelReply:
    """Run one prompt through `backend` in a fresh empty directory with a scrubbed environment.

    The directory is removed afterwards, whether or not the call succeeds; if
    it cannot be removed, a warning names it and the call's result stands.
    """
    try:
        cwd = Path(tempfile.mkdtemp(prefix="minilab-call-"))
    except OSError as exc:
        raise ModelNotStarted(f"cannot make the call's working directory: {exc}") from exc
    try:
        return backend.run(prompt, cwd=cwd, env=scrubbed_env(), timeout_s=timeout_s)
    finally:
        _remove_dir(cwd)


def call(
    family: Family | str,
    role: Role,
    prompt: str,
    *,
    record_id: str | None,
    pair: Pair,
    ledger: Ledger,
    budget: Budget,
    backends: Mapping[Family, Backend] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ModelReply:
    """Call `family`'s model in `role`, charge the call to `pair`, and return the reply.

    `record_id` is None for a call made before any record exists, such as
    generation. Raise `RoleNotAllowed` for Gemini in a role not in
    `GEMINI_ROLES`;
    `ValueError` for a family or pair the run's manifest does not list, or a
    record of another run or pair, a backend of another family or of a model
    other than the one the manifest names, or a budget
    larger than the manifest's; `KeyError` for a record id that names no
    stored record; `BudgetExceeded` if the pair's spend is at or over the
    budget; and `ModelUnavailable` if the tool fails (`ModelNotStarted` if it
    failed before it started). Each of these is raised before the call except
    `ModelUnavailable`. Any exception the backend raises after the tool
    started is charged as a failed, estimated call and raised again: input is
    the larger of the reported count and the prompt estimate; output is the
    reported count, else an estimate of what the tool printed if it ended the
    call itself, else the whole output reservation. No record is written.
    """
    family = Family(family)
    role = Role(role)
    if family is Family.gemini and role not in GEMINI_ROLES:
        raise RoleNotAllowed(
            f"gemini only generates and makes proof attempts; refused role {role.value}"
        )
    pair = canonical_pair(pair)
    if family not in ledger.manifest.families:
        raise ValueError(f"family {family} is not in run {ledger.run_id}'s manifest")
    if pair not in ledger.manifest.pairs:
        raise ValueError(f"pair {pair} is not in run {ledger.run_id}'s manifest")
    if record_id is not None:
        record_id = check_record_id(record_id)
        if run_of(record_id) != ledger.run_id:
            raise ValueError(f"record {record_id} is not in run {ledger.run_id}")
        record = ledger.store.load(record_id)
        if canonical_pair(record.pair) != pair:
            raise ValueError(f"record {record_id} is for pair {record.pair}, not {pair}")
    if budget.tokens_per_pair > ledger.manifest.budget_tokens_per_pair:
        raise ValueError(
            f"budget {budget.tokens_per_pair} is larger than run {ledger.run_id}'s "
            f"{ledger.manifest.budget_tokens_per_pair} tokens per pair"
        )
    backend = (backends if backends is not None else default_backends(ledger.manifest)).get(family)
    if backend is None:
        raise ModelNotStarted(f"no backend for family {family}")
    if backend.family != family:
        raise ValueError(f"the backend for {family} is a {backend.family} backend")
    if backend.model != ledger.manifest.model_of(family):
        raise ValueError(
            f"the {family} backend runs {backend.model}, but run {ledger.run_id}'s "
            f"manifest names {ledger.manifest.model_of(family)}"
        )
    stage = ROLE_STAGE[role]
    reserve = min(estimate_tokens(prompt) + budget.reserve_output_tokens, MAX_TOKENS)
    with ledger.pair_lock(pair):
        spend = ledger.pair_spend(pair)
        if spend >= budget.tokens_per_pair:
            raise BudgetExceeded(
                f"pair {pair} has spent {spend} of {budget.tokens_per_pair} tokens"
            )
        reservation = ledger.reserve(
            pair=pair, stage=stage, family=family, tokens=reserve, record_id=record_id
        )
    started = time.monotonic()
    try:
        reply = isolated_run(backend, prompt, timeout_s=timeout_s)
    except ModelNotStarted as exc:
        try:
            ledger.release(reservation)
        except Exception as release_exc:
            exc.add_note(f"releasing this call's reservation also failed: {release_exc!r}")
        raise
    except BaseException as exc:
        reported_in = exc.tokens_in if isinstance(exc, ModelUnavailable) else None
        reported_out = exc.tokens_out if isinstance(exc, ModelUnavailable) else None
        printed = exc.output if isinstance(exc, ModelUnavailable) else None
        if reported_out is not None:
            tokens_out = reported_out
        elif printed is not None:
            tokens_out = min(estimate_tokens(printed), MAX_TOKENS)
        else:
            tokens_out = budget.reserve_output_tokens
        try:
            ledger.charge(
                pair=pair,
                stage=stage,
                family=family,
                tokens_in=max(reported_in or 0, estimate_tokens(prompt)),
                tokens_out=tokens_out,
                seconds=time.monotonic() - started,
                estimated=True,
                failed=True,
                record_id=record_id,
                reservation=reservation,
            )
        except Exception as charge_exc:
            exc.add_note(f"charging this failed call also failed: {charge_exc!r}")
        raise
    ledger.charge(
        pair=pair,
        stage=stage,
        family=family,
        tokens_in=reply.tokens_in,
        tokens_out=reply.tokens_out,
        seconds=time.monotonic() - started,
        estimated=reply.estimated,
        record_id=record_id,
        reservation=reservation,
    )
    return reply


def pick_checker(
    author: Family | str,
    available: Iterable[Family | str],
    *,
    short: bool = False,
    codex_budget_left: bool = False,
) -> Family:
    """Pick the family that checks work written by `author`.

    Never the author and never Gemini. The local weak model is not a checker
    either. Prefer Claude over Codex, because Codex has the tightest limits;
    but for a short checking call while Codex still has budget, prefer Codex
    (design doc-gmj: Codex budget goes to checking calls first). Raise
    `NoChecker` if neither Claude nor Codex is available apart from the author.
    """
    author = Family(author)
    available = [Family(f) for f in available]
    candidates = [f for f in (Family.claude, Family.codex) if f in available and f is not author]
    if not candidates:
        raise NoChecker(f"no checker for work by {author} among {available}")
    if short and codex_budget_left and Family.codex in candidates:
        return Family.codex
    return candidates[0]
