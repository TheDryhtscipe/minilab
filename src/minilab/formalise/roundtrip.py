"""The round-trip check: another family reads the Lean back into English.

Build plan doc-5wh, Task 11; design doc-gmj, stage 5.

A family other than the formaliser's, never Gemini (`pick_checker`),
translates the Lean file into English. Its prompt holds only the Lean source
with its comments removed: never the record's statement, its assumptions, or
any comment the formaliser wrote. Names and string literals in the source
stay, so the formaliser can still carry some of the original wording in
them; the round trip narrows that channel, it does not close it.
The record keeps both texts, so the reviewer can compare them.
"""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from minilab.formalise.formaliser import THEOREM, FormalOutput, strip_comments
from minilab.ledger import Ledger
from minilab.models.adapter import Budget, Role, call, pick_checker
from minilab.models.backends import Backend
from minilab.records import Family, Record


class RoundTrip(BaseModel):
    """The translator's English beside the record's original statement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Family
    model: str
    lean_statement: str
    english: str
    original: str


def roundtrip_prompt(formal: FormalOutput) -> str:
    """The prompt: the comment-free Lean source and nothing from the record."""
    return f"""Translate the Lean theorem `{THEOREM}` below into plain mathematical English.

State every hypothesis and the conclusion exactly as the Lean says them,
including the meaning of each definition the theorem uses. Do not guess what
the author meant, and do not add or drop conditions. Reply with the English
statement only.

```lean
{strip_comments(formal.source).strip()}
```"""


def roundtrip(
    formal: FormalOutput,
    author: Family | str,
    *,
    record: Record,
    ledger: Ledger,
    budget: Budget,
    backends: Mapping[Family, Backend] | None = None,
) -> RoundTrip:
    """Translate `formal` back to English with a family other than `author`.

    `record` is used only to charge the call and to keep the original text;
    nothing from it reaches the prompt. Raise NoChecker if no family can
    translate, and what `call` raises.
    """
    family = pick_checker(author, ledger.manifest.families)
    reply = call(
        family,
        Role.roundtrip,
        roundtrip_prompt(formal),
        record_id=record.id,
        pair=record.pair,
        ledger=ledger,
        budget=budget,
        backends=backends,
    )
    return RoundTrip(
        family=family,
        model=reply.model,
        lean_statement=formal.lean_statement,
        english=reply.text.strip(),
        original=record.statement,
    )
