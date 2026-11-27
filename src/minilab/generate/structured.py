"""Structured generation: elicit, abstract, cross.

Design doc-gmj, "Generator"; build plan doc-5wh, Task 15. Three calls to one
family, each building on the reply before it:

1. *Elicit* the central objects, operations, and inequalities of each domain.
2. *Abstract* them to their structure.
3. *Cross* that structure into the other domain and state conjectures, as JSON.

Each step's prompt and reply are logged on every record the run makes. The
run's manifest must be a structured run.
"""

from collections.abc import Mapping, Sequence

from minilab.generate.core import (
    Generation,
    domain_name,
    earlier_section,
    json_instructions,
    run_generation,
)
from minilab.ledger import Ledger
from minilab.models.adapter import Budget
from minilab.models.backends import Backend
from minilab.records import Family, Record
from minilab.scheduler import Pair, canonical_pair


def _names(pair: Pair) -> tuple[str, str]:
    a, b = canonical_pair(pair)
    return domain_name(a), domain_name(b)


def elicit_prompt(pair: Pair) -> str:
    a, b = _names(pair)
    return f"""For each of these two domains, list its central objects, the operations
on them, and its most important inequalities and identities. Be concrete and
give each inequality in symbols.

- {a}
- {b}

Write a plain list under one heading per domain."""


def abstract_prompt(pair: Pair, elicited: str) -> str:
    a, b = _names(pair)
    return f"""Here are the central objects, operations, and inequalities of two domains,
{a} and {b}:

{elicited}

Abstract each to its structure: what makes it work, stated without the domain's
own vocabulary (for example convexity, a variational form, a monotone quantity
along a flow, a duality, a symmetry). For each structure, say which items above
have it. Note where a structure appears in one domain but not yet in the other."""


def cross_prompt(pair: Pair, abstracted: str, n: int, earlier: Sequence[str] = ()) -> str:
    a, b = _names(pair)
    return f"""Here are structures abstracted from two domains, {a} and {b}:

{abstracted}

Cross them: take a structure from one domain into the other, and state
conjectures that would follow if it carried over. Each conjecture must use ideas
from both domains. Prefer precise, checkable claims, such as inequalities,
identities, or bounds, that are not already standard results. Give every
assumption explicitly, and state the domain where the claim is meant to hold,
because counterexamples are checked against them.

{earlier_section(earlier)}{json_instructions(n)}"""


def _steps(generation: Generation) -> tuple[str, str]:
    elicited = generation.ask("elicit", elicit_prompt(generation.pair))
    abstracted = generation.ask("abstract", abstract_prompt(generation.pair, elicited.reply))
    return "cross", cross_prompt(
        generation.pair, abstracted.reply, generation.n, generation.earlier
    )


def generate_structured(
    pair: Pair,
    family: Family | str,
    n: int,
    *,
    ledger: Ledger,
    budget: Budget,
    backends: Mapping[Family, Backend] | None = None,
    earlier: Sequence[str] = (),
) -> list[Record]:
    """Ask `family` in three steps for up to `n` conjectures on `pair`, and store them.

    Return the records made, in the reply's order. Each record's generator
    stage logs every step (`data["steps"]`). Parsing, retries, and errors are as
    for `generate_plain`, except that the run must be a structured run; only
    the cross step is asked again if its reply cannot be read. `earlier` goes
    into the cross step only.
    """
    generation = Generation(
        "structured",
        pair,
        family,
        n,
        ledger=ledger,
        budget=budget,
        backends=backends,
        earlier=earlier,
    )
    return run_generation(generation, _steps)
