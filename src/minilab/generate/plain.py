"""Plain generation: one prompt asks for conjectures that link two domains.

Design doc-gmj, "Generator"; build plan doc-5wh, Task 15. The run's manifest
must be a plain run. The call is charged to the pair and to no record, since
no record exists yet.
"""

from collections.abc import Mapping

from minilab.generate.core import Generation, domain_name, json_instructions, run_generation
from minilab.ledger import Ledger
from minilab.models.adapter import Budget
from minilab.models.backends import Backend
from minilab.records import Family, Record
from minilab.scheduler import Pair, canonical_pair


def plain_prompt(pair: Pair, n: int) -> str:
    """The one prompt of plain mode."""
    a, b = (domain_name(slug) for slug in canonical_pair(pair))
    return f"""Propose new mathematical conjectures that link these two domains, each
of which bears on machine learning:

- {a}
- {b}

Each conjecture must use ideas from both domains. Prefer precise, checkable
claims, such as inequalities, identities, or bounds, that are not already
standard results. Give every assumption explicitly, and state the domain where
the claim is meant to hold, because counterexamples are checked against them.

{json_instructions(n)}"""


def generate_plain(
    pair: Pair,
    family: Family | str,
    n: int,
    *,
    ledger: Ledger,
    budget: Budget,
    backends: Mapping[Family, Backend] | None = None,
) -> list[Record]:
    """Ask `family` for up to `n` conjectures on `pair`, and store them as records.

    Return the records made, in the reply's order. A conjecture with no
    instance family is stored as `untestable`. A reply that cannot be read is
    asked for once more, then dropped and logged, and the result is empty.
    Raise ValueError for a family that does not generate (only Claude, Codex,
    and Gemini do) or a run that is not a plain run. Model failures
    (`ModelUnavailable`, `BudgetExceeded`, ...) are raised as `call` raises
    them; no record is written for that batch.
    """
    generation = Generation(
        "plain", pair, family, n, ledger=ledger, budget=budget, backends=backends
    )
    return run_generation(generation, lambda g: ("generate", plain_prompt(g.pair, g.n)))
