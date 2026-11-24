"""The model adapter: the one way minilab calls a model.

`call` enforces the rules every model call must follow (design doc-gmj,
"Constraints" and "Generator"):

- Gemini generates only. Any other role for `Family.gemini` is refused.
- The pair's token budget is checked before the call. The check, the call,
  and the ledger charge run inside one `Ledger.pair_lock`, so two processes
  on the same pair cannot both pass the check and then overrun the budget.
  The lock does not nest: never call `call` while holding the pair's lock.
  A call is refused once the pair's spend is at or over its budget, so the
  last call that is let through can overrun the budget by its own tokens.
- Every call runs isolated: a fresh empty temporary working directory, removed
  afterwards, and an environment reduced to an allowlist, so no
  `ANTHROPIC_*` or other credential or tool-routing variable reaches the tool.
- Every call is charged to its pair, including a failed one. A failed call is
  charged its estimated prompt tokens, marked as an estimate, because the tool
  may have spent tokens before it failed. Then `ModelUnavailable` is raised.
  The adapter never writes to a record, so a failure leaves the record as it
  was and the run can resume.

`pick_checker` chooses the family that checks another family's work.
"""

import os
import shutil
import tempfile
import time
from collections.abc import Mapping
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from minilab.ledger import Ledger
from minilab.models.backends import (
    DEFAULT_TIMEOUT_S,
    Backend,
    ClaudeBackend,
    CodexBackend,
    GeminiBackend,
    LocalBackend,
    ModelReply,
    ModelUnavailable,
    estimate_tokens,
)
from minilab.records import Family, Stage, check_record_id, run_of
from minilab.run import RunManifest
from minilab.scheduler import Pair, canonical_pair

__all__ = [
    "ENV_ALLOWLIST",
    "ROLE_STAGE",
    "Budget",
    "BudgetExceeded",
    "ModelReply",
    "ModelUnavailable",
    "NoChecker",
    "Role",
    "RoleNotAllowed",
    "call",
    "default_backends",
    "isolated_run",
    "pick_checker",
    "scrubbed_env",
]


class Role(Enum):
    """A model call's job. A plain `Enum`, not a `StrEnum`: the member `encode`
    would shadow `str.encode` on every member."""

    generate = "generate"
    encode = "encode"
    reformulate = "reformulate"
    mechanism = "mechanism"
    formalise = "formalise"
    roundtrip = "roundtrip"
    audit = "audit"
    literature = "literature"
    prove = "prove"
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
    Role.curate: Stage.queue,
    Role.grade: Stage.queue,
}
"""The stage each role's cost is charged to. Curating and grading the benchmark
have no pipeline stage of their own; they are charged to the queue."""

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
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "CODEX_HOME",
        "CLAUDE_CONFIG_DIR",
    }
)
"""Variables a tool may see: what it needs to find its binary, its login, and
the OS keyring. Everything else, including every `ANTHROPIC_*` variable and
any API key, is dropped."""


class BudgetExceeded(Exception):
    """The pair's token spend is at or over its budget; no call was made."""


class RoleNotAllowed(Exception):
    """The family may not take this role (Gemini generates only)."""


class NoChecker(Exception):
    """No family other than the author, and other than Gemini, is available to check."""


class Budget(BaseModel):
    """The token budget for one domain pair: every model call for the pair counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tokens_per_pair: int = Field(gt=0)

    @classmethod
    def from_manifest(cls, manifest: RunManifest) -> "Budget":
        return cls(tokens_per_pair=manifest.budget_tokens_per_pair)


def scrubbed_env() -> dict[str, str]:
    """Return the current environment reduced to `ENV_ALLOWLIST`."""
    return {name: value for name, value in os.environ.items() if name in ENV_ALLOWLIST}


def default_backends() -> dict[Family, Backend]:
    return {
        Family.claude: ClaudeBackend(),
        Family.codex: CodexBackend(),
        Family.gemini: GeminiBackend(),
        Family.local: LocalBackend(),
    }


def isolated_run(backend: Backend, prompt: str, *, timeout_s: float) -> ModelReply:
    """Run one prompt through `backend` in a fresh empty directory with a scrubbed environment.

    The directory is removed afterwards, whether or not the call succeeds.
    """
    cwd = Path(tempfile.mkdtemp(prefix="minilab-call-"))
    try:
        return backend.run(prompt, cwd=cwd, env=scrubbed_env(), timeout_s=timeout_s)
    finally:
        shutil.rmtree(cwd, ignore_errors=True)


def call(
    family: Family,
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
    generation. Raise `RoleNotAllowed` for Gemini in any role but generate;
    `ValueError` for a family or pair the run's manifest does not list, or a
    record of another run or pair; `BudgetExceeded` if the pair's spend is at
    or over the budget; and `ModelUnavailable` if the tool fails. Each of these
    is raised before the call except `ModelUnavailable`. No record is written.
    """
    if family is Family.gemini and role is not Role.generate:
        raise RoleNotAllowed(f"gemini generates only; refused role {role.value}")
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
    backend = (backends if backends is not None else default_backends()).get(family)
    if backend is None:
        raise ModelUnavailable(f"no backend for family {family}")
    stage = ROLE_STAGE[role]
    with ledger.pair_lock(pair):
        spend = ledger.pair_spend(pair)
        if spend >= budget.tokens_per_pair:
            raise BudgetExceeded(
                f"pair {pair} has spent {spend} of {budget.tokens_per_pair} tokens"
            )
        started = time.monotonic()
        try:
            reply = isolated_run(backend, prompt, timeout_s=timeout_s)
        except ModelUnavailable:
            ledger.charge(
                pair=pair,
                stage=stage,
                family=family,
                tokens_in=estimate_tokens(prompt),
                tokens_out=0,
                seconds=time.monotonic() - started,
                estimated=True,
                record_id=record_id,
            )
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
        )
    return reply


def pick_checker(
    author: Family,
    available: list[Family],
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
    candidates = [f for f in (Family.claude, Family.codex) if f in available and f is not author]
    if not candidates:
        raise NoChecker(f"no checker for work by {author} among {available}")
    if short and codex_budget_left and Family.codex in candidates:
        return Family.codex
    return candidates[0]
