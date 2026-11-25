"""The child side of the encoding sandbox. Library code; models never write it.

Run as `python -I -m minilab.encoding._child` inside the sandbox. It reads one
request (the encoding's source and a list of calls) from stdin, runs the module,
builds the encoding once, makes each call at the precision it names, and writes
one JSON response to the stdout it started with. Anything the encoding prints
goes to stderr.

Nothing here is trusted by the parent: an encoding can patch this module too.
That only lets it lie about its own values, which the checker already has to
assume (the two-family guard in `check_kill`).
"""

import os
import random
import sys
import types

import flint
import mpmath

from minilab.certify.result import working_digits
from minilab.encoding.sandbox import (
    CAPABILITIES,
    Call,
    CallResult,
    EncodingOutput,
    Request,
    describe,
)
from minilab.encoding.wire import to_wire


def _one(enc: object, call: Call) -> CallResult:
    try:
        with working_digits(call.digits):
            expected = (mpmath.mp.prec, flint.ctx.prec)
            if call.method == "sample":
                value = getattr(enc, "sample")(random.Random(call.seed))  # noqa: B009
            else:
                value = getattr(enc, call.method)(call.point)
            if (mpmath.mp.prec, flint.ctx.prec) != expected:
                return CallResult(
                    error=f"the encoding changed the working precision in {call.method}"
                )
            return CallResult(value=to_wire(value))
    except BaseException as exc:  # noqa: BLE001 - report everything, SystemExit included
        return CallResult(error=f"the encoding raised {describe(exc)} in {call.method}")


def _capabilities(enc: object) -> list[str]:
    found: list[str] = []
    for name in CAPABILITIES:
        try:
            if callable(getattr(enc, name, None)):
                found.append(name)
        except BaseException:  # noqa: BLE001, S112 - a property may raise; treat as absent
            continue
    return found


def main() -> None:
    out = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    request = Request.model_validate_json(sys.stdin.read())
    module = types.ModuleType("encoding")
    try:
        exec(compile(request.source, "<encoding>", "exec"), module.__dict__)  # noqa: S102
        enc: object = module.__dict__["ENCODING"]()
    except BaseException as exc:  # noqa: BLE001 - report everything, SystemExit included
        response = EncodingOutput(failure=f"loading the encoding raised {describe(exc)}")
    else:
        response = EncodingOutput(
            capabilities=_capabilities(enc),
            results=[_one(enc, call) for call in request.calls],
        )
    out.write(response.model_dump_json())
    out.flush()


if __name__ == "__main__":
    main()
