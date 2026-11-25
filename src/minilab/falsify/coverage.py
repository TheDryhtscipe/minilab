"""The coverage report: what a search that found no kill actually searched.

"Survived" means "no counterexample found under this coverage", never "true"
(design doc-gmj, "Falsifier"). The report states the space and dimensions
searched, the number of starts, the worst margin found and where, and what the
search did not cover.
"""

from pydantic import BaseModel, ConfigDict, Field

from minilab.encoding.api import Instance


class CoverageReport(BaseModel):
    """What one search covered. Margins are decimal strings, so any size fits in JSON."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    space: str
    """The parameters searched and the range each reached."""
    dimensions: int = Field(ge=0)
    """The most coordinates the local optimiser moved at once."""
    starts: int = Field(ge=0)
    """Cold starts: fresh samples, each the start of one local optimisation."""
    worst_margin: str | None
    """The lowest margin at a point in the domain that met every assumption, or
    None if no such point was evaluated."""
    worst_point: Instance | None
    uncovered: list[str]
    """Regions and cases the search did not cover."""
    evaluations: int = Field(default=0, ge=0)
    """Points evaluated in the domain with every assumption met."""
    failed_evaluations: int = Field(default=0, ge=0)
    """Points where the encoding raised or the sandbox failed."""
    boundary_points: int = Field(default=0, ge=0)
    """Points found next to the edge of the domain."""
    local_evaluations: int = Field(default=0, ge=0)
    """Points the local optimiser asked for."""
