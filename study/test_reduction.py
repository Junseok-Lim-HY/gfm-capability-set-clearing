#!/usr/bin/env python3
"""The correctness anchor: strip the three omissions and the circle IS the box.

The box bound `P + R + H <= Pbar` differs from the current-based capability set
in exactly three ways, and each has its own sign:

  * it ignores the short-term current rating, so it is conservative in P;
  * it ignores reactive current, so it is optimistic whenever Q is nonzero;
  * it ignores terminal voltage, so it is optimistic below 1.0 pu and
    conservative above it.

Set the overload rating equal to the continuous one, remove the reactive
requirement and hold voltage at 1.0 pu, and the two formulations must return
the same schedule and the same prices. If they do not, the difference reported
everywhere else is a coding error rather than a modelling result, so this runs
first.

The residual is the inscribed polygon: a regular m-gon inside the unit circle
is short by 1 - cos(pi/m), which at 192 faces is 0.013 per cent. The test
allows a tenth of a per cent and reports what it actually saw.
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "model"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from multiperiod import (MultiPeriod, build_case,  # noqa: E402
                         common_demand_peak)
from procurement_lp import Capability  # noqa: E402
from prices import price  # noqa: E402

TOLERANCE_PCT = 0.10
SIDES = 192


def reduced(name: str, share: float):
    """The same case with all three differences removed."""
    base = build_case(name, converter_share=share,
                      demand_peak_mw=common_demand_peak(name, (0.2, 0.4)))
    return dataclasses.replace(
        base,
        conv_injection_pu=np.full(len(base.demand_mw), 0.8),
        conv_voltage_pu=np.ones_like(base.conv_voltage_pu),
        reactive_frac=0.0,
    )


def main() -> int:
    settings = Capability(i_continuous=1.0, i_short_term=1.0,
                          priority="active", soc_threshold=0.20,
                          polygon_sides=SIDES)
    print("reduction test: V = 1.0 pu, I_s = I_c, no reactive requirement, "
          f"{SIDES}-sided polygon")
    print(f'{"case":>7} {"share":>6} {"price box":>10} {"price circle":>12}'
          f' {"cost box":>11} {"cost circle":>12} {"gap %":>8}')

    worst, failures = 0.0, 0
    for name in ("ieee39", "rts24"):
        for share in (0.2, 0.4):
            case = reduced(name, share)
            out = {form: MultiPeriod(case, settings, form).solve()
                   for form in ("box", "current")}
            if not all(out[form]["feasible"] for form in out):
                print(f"{name:>7} {share:6.1f}   infeasible")
                failures += 1
                continue
            inertia = {f: price(out[f]["prices"], "inertia") for f in out}
            gap = 100.0 * abs(out["current"]["cost"] - out["box"]["cost"]) \
                / out["box"]["cost"]
            worst = max(worst, gap)
            failures += gap > TOLERANCE_PCT
            print(f'{name:>7} {share:6.1f} {inertia["box"]:10.2f}'
                  f' {inertia["current"]:12.2f} {out["box"]["cost"]:11.1f}'
                  f' {out["current"]["cost"]:12.1f} {gap:8.4f}')

    # The comparator is the shortfall the polygon imposes on the active
    # coordinate, measured in test_sign_condition.py. The radial figure
    # 1 - cos(pi/m) is smaller and is not the error the clearing meets.
    radial = 100.0 * (1.0 - np.cos(np.pi / SIDES))
    print(f"\nworst gap {worst:.4f} per cent against a tolerance of "
          f"{TOLERANCE_PCT}; the inscribed {SIDES}-gon imposes up to "
          f"0.027 per cent on the active coordinate, and the radial figure "
          f"{radial:.4f} understates that")
    print("PASS" if failures == 0 else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
