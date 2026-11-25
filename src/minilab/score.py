"""The scorer: triviality signals and a compression measure, for ranking the queue.

The scorer never removes anything from the queue (design doc-gmj, "Scorer").
It reads what earlier stages wrote on the record:

- the triviality screen's `near_misses`: names of the screens that nearly
  succeeded (`Stage.triviality`, `data["near_misses"]`, a list of strings);
- the prover's proof text (`Stage.prover`, `data["proof"]`, a string);
- the formal statement (`Stage.formaliser`, `data["lean_statement"]`, a
  string), or the informal statement if there is none.

Compression is statement tokens divided by proof tokens. Both sides are
counted in Lean where they can be, so the ratio compares like with like: the
Lean statement against the Lean proof. Tokens come from `token_count`, a fixed
lexical count (a dotted name such as `Real.add_pos` is one token), not a model
tokenizer, so a score never changes with the model in use. A record with no
proof has no compression.
"""

import re

from pydantic import BaseModel, ConfigDict, JsonValue

from minilab.records import Record, Stage

_TOKEN = re.compile(r"[^\W\d][\w.'!?]*|\d+(?:\.\d+)?|\S")
"""A name (letters, digits, `_`, `.`, `'`, `!`, `?`), a number, or one other symbol."""

_TACTICS: frozenset[str] = frozenset(
    {
        "apply_fun",
        "by_cases",
        "by_contra",
        "exact_mod_cast",
        "field_simp",
        "fin_cases",
        "interval_cases",
        "linear_combination",
        "mod_cast",
        "norm_cast",
        "norm_num",
        "push_cast",
        "push_neg",
        "ring_nf",
        "rw_mod_cast",
        "set_option",
        "simp_all",
        "simp_rw",
        "split_ifs",
    }
)
"""Tactic names that look like lemma names (they hold `_`) but are not."""


class Score(BaseModel):
    """Ranking signals for one record. None of them is a verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    statement_tokens: int
    """Tokens in the Lean statement, or in the informal statement if there is none."""
    proof_tokens: int | None
    """Tokens in the proof; None if the record has no proof."""
    compression: float | None
    """`statement_tokens / proof_tokens`; None without a proof."""
    near_misses: list[str]
    """Screens of the triviality stage that nearly succeeded."""
    library_lemmas: list[str]
    """Distinct library names the proof cites, sorted."""

    def rank_key(self) -> tuple[int, int, float]:
        """Sort key: fewer near misses first, then records with a proof, then
        higher compression (the statement says more for each token of proof)."""
        if self.compression is None:
            return (len(self.near_misses), 1, 0.0)
        return (len(self.near_misses), 0, -self.compression)


def token_count(text: str) -> int:
    """Count the lexical tokens in `text`: names, numbers, and single symbols."""
    return len(_TOKEN.findall(text))


def _stage_str(record: Record, stage: Stage, key: str) -> str | None:
    output = record.stages.get(stage)
    if output is None:
        return None
    value: JsonValue = output.data.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _near_misses(record: Record) -> list[str]:
    output = record.stages.get(Stage.triviality)
    if output is None:
        return []
    value: JsonValue = output.data.get("near_misses")
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def library_lemmas(proof: str) -> list[str]:
    """Return the distinct names in `proof` that look like library lemmas.

    A lemma name holds `.` or `_` (as in `Real.add_pos` or `sq_nonneg`); a
    short list of tactics with `_` in their names is left out.
    """
    names = {
        token.rstrip(".")
        for token in _TOKEN.findall(proof)
        if token[0].isalpha() and ("." in token.rstrip(".") or "_" in token)
    }
    return sorted(names - _TACTICS)


def score(record: Record) -> Score:
    """Score one record for ranking. It reads the record only."""
    statement = _stage_str(record, Stage.formaliser, "lean_statement") or record.statement
    proof = _stage_str(record, Stage.prover, "proof")
    statement_tokens = token_count(statement)
    proof_tokens = None if proof is None else token_count(proof)
    return Score(
        statement_tokens=statement_tokens,
        proof_tokens=proof_tokens,
        compression=None if not proof_tokens else statement_tokens / proof_tokens,
        near_misses=_near_misses(record),
        library_lemmas=[] if proof is None else library_lemmas(proof),
    )
