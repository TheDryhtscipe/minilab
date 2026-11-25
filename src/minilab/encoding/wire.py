"""Values sent from the encoding sandbox back to the checker, written exactly.

The child process turns what an encoding method returns into one of these plain
records. The parent rebuilds fresh values of exact library types (bool,
Fraction, mpmath.mpf, flint.arb, Instance) and makes every comparison itself
(decision doc-j6t). Integers travel as hexadecimal text: Python refuses to read
a decimal integer of more than 4300 digits, and a binary mantissa at thousands
of digits is longer than that.

A value of any other type, or a subclass of an allowed type, comes back as
`Opaque`, which no check accepts.
"""

from fractions import Fraction
from typing import Annotated, Literal, cast

import flint
import mpmath
from pydantic import BaseModel, ConfigDict, Field

from minilab.encoding.api import Instance

# Hexadecimal digits allowed in one integer: about 160,000 bits, far beyond the
# mantissa of a value at the checker's highest precision.
MAX_HEX = 40_000

# The largest binary exponent accepted.
MAX_EXPONENT = 10**7

Hex = Annotated[str, Field(max_length=MAX_HEX + 3, pattern=r"^-?0x[0-9a-f]+$")]
Exponent = Annotated[int, Field(ge=-MAX_EXPONENT, le=MAX_EXPONENT)]


class _Wire(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class WireBool(_Wire):
    kind: Literal["bool"] = "bool"
    value: bool


class WireBools(_Wire):
    kind: Literal["bools"] = "bools"
    value: list[bool]


class WireRational(_Wire):
    kind: Literal["rational"] = "rational"
    numerator: Hex
    denominator: Hex


class WireMpf(_Wire):
    """A finite mpf, exactly: man * 2**exp."""

    kind: Literal["mpf"] = "mpf"
    man: Hex
    exp: Exponent


class WireArb(_Wire):
    """An arb ball: midpoint man * 2**exp, radius rad_man * 2**rad_exp."""

    kind: Literal["arb"] = "arb"
    finite: bool
    man: Hex = "0x0"
    exp: Exponent = 0
    rad_man: Hex = "0x0"
    rad_exp: Exponent = 0


class WirePair(_Wire):
    kind: Literal["pair"] = "pair"
    first: WireArb
    second: WireArb


class WireInstance(_Wire):
    kind: Literal["instance"] = "instance"
    value: Instance


class WireOpaque(_Wire):
    """A value of a type the checker does not accept; only its type name travels."""

    kind: Literal["opaque"] = "opaque"
    type_name: Annotated[str, Field(max_length=200)]


WireValue = Annotated[
    WireBool | WireBools | WireRational | WireMpf | WireArb | WirePair | WireInstance | WireOpaque,
    Field(discriminator="kind"),
]


class Opaque:
    """A value the sandbox could not send as an allowed type. No check accepts it."""

    def __init__(self, type_name: str) -> None:
        self.type_name = type_name

    def __repr__(self) -> str:
        return f"Opaque({self.type_name})"


def _hex(n: int) -> str:
    text = hex(n)
    if len(text) > MAX_HEX + 3:
        raise ValueError("an integer is too large to send")
    return text


def _arb_to_wire(ball: flint.arb) -> WireArb:
    if not ball.is_finite():
        return WireArb(finite=False)
    man, exp = cast(tuple[int, int], ball.mid().man_exp())
    rad_man, rad_exp = cast(tuple[int, int], ball.rad().man_exp())
    return WireArb(finite=True, man=_hex(man), exp=exp, rad_man=_hex(rad_man), rad_exp=rad_exp)


def to_wire(value: object) -> WireValue:
    """Write what an encoding method returned. Runs in the child process.

    Exact type checks: a subclass could override comparisons or iteration.
    """
    if type(value) is bool:
        return WireBool(value=value)
    if type(value) is list:
        items = cast(list[object], value)
        if all(type(item) is bool for item in items):
            return WireBools(value=cast(list[bool], items))
        return WireOpaque(type_name="list with non-bool items")
    if type(value) is Fraction:
        return WireRational(numerator=_hex(value.numerator), denominator=_hex(value.denominator))
    if type(value) is mpmath.mpf:
        if not mpmath.isfinite(value):
            return WireOpaque(type_name="non-finite mpf")
        # mpmath keeps the sign apart: `man` is never negative.
        sign, man, exp, _ = cast(tuple[int, int, int, int], value._mpf_)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        return WireMpf(man=_hex(-man if sign else man), exp=exp)
    if type(value) is flint.arb:
        return _arb_to_wire(value)
    if type(value) is tuple:
        items = cast(tuple[object, ...], value)
        if len(items) == 2 and all(type(item) is flint.arb for item in items):
            first, second = cast(tuple[flint.arb, flint.arb], items)
            return WirePair(first=_arb_to_wire(first), second=_arb_to_wire(second))
        return WireOpaque(type_name="tuple that is not two arb balls")
    if type(value) is Instance:
        return WireInstance(value=value)
    return WireOpaque(type_name=type(value).__name__[:200])


def _arb_from_wire(wire: WireArb) -> flint.arb:
    if not wire.finite:
        return flint.arb("nan")
    mid = flint.arf((int(wire.man, 16), wire.exp))
    rad = flint.arf((int(wire.rad_man, 16), wire.rad_exp))
    # The midpoint is kept exactly and the radius rounded up, so the rebuilt
    # ball contains the one the child sent.
    return flint.arb(mid, rad)


def from_wire(wire: WireValue) -> object:
    """Rebuild a fresh value of an exact library type. Runs in the parent.

    An mpf is rebuilt at the current mpmath precision; the parent runs at the
    precision the child used, so a mantissa the child computed fits exactly.
    """
    match wire:
        case WireBool():
            return wire.value
        case WireBools():
            return list(wire.value)
        case WireRational():
            return Fraction(int(wire.numerator, 16), int(wire.denominator, 16))
        case WireMpf():
            return mpmath.mpf((int(wire.man, 16), wire.exp))
        case WireArb():
            return _arb_from_wire(wire)
        case WirePair():
            return (_arb_from_wire(wire.first), _arb_from_wire(wire.second))
        case WireInstance():
            return Instance.model_validate_json(wire.value.model_dump_json())
        case WireOpaque():
            return Opaque(wire.type_name)
