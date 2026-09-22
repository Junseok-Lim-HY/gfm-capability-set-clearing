#!/usr/bin/env python3
"""Does a formulation that is called safe keep the bridge inside its current?

Two claims in the manuscript are safety claims and neither was tested.

The proposed set is said to keep every converter inside its continuous rating.
It did, at the activated point (P + R, Q), which is the larger current when the
converter is discharging. A charging converter moves *towards* the origin when
its reserve is called, so its largest current is the one it carries before the
call, and that point was not constrained.

The uniform derating is said to be safe everywhere, which is what makes it the
careful planner's comparator. Its factor was computed from the zonal
requirement fraction as though that were a bound on each unit's Q/S. It is not,
and re-clearing RTS-24 produced continuous currents above 1.3 pu under it.

So both are measured, on the cleared schedules, at the voltage the clearing
used, before and after activation:

    i_before = |(P, Q)|     / (V S)
    i_after  = |(P + R, Q)| / (V S)

and the test fails if either exceeds the continuous rating for the forms that
claim safety. The apparent-power comparator ignores voltage and is reported,
not tested: exceeding the rating below 1 pu is what that model does.

Writes results/current_safety.csv.
"""
from __future__ import annotations

import dataclasses
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "model"))

from multiperiod import (MultiPeriod, build_case,  # noqa: E402
                         common_demand_peak, derating_reactive_cap,
                         uniform_derating)
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
CASES = ("ieee39", "rts24")
SHARES = (0.2, 0.4)
DISPATCH = (0.0, 0.8)
FORMS = ("current", "derated", "apparent", "box")
CLAIMS_SAFETY = ("current", "derated")
TOL = 1e-6


def currents(case, outcome) -> tuple:
    injection = np.zeros_like(outcome["pd"])
    if case.conv_injection_pu is not None:
        injection = np.outer(np.asarray(case.conv_injection_pu, dtype=float),
                             case.conv_rating_mva)
    net = outcome["pd"] - outcome["pc"] - outcome["curt"] + injection
    base = case.conv_voltage_pu * case.conv_rating_mva
    before = np.hypot(net, outcome["q"]) / base
    after = np.hypot(net + outcome["r"], outcome["q"]) / base
    return float(before.max()), float(after.max()), float(net.min())


def main() -> int:
    settings = Capability(i_short_term=1.5, priority="reactive",
                          soc_threshold=0.20)
    rows, failed = [], []
    for name, share, dispatch in itertools.product(CASES, SHARES, DISPATCH):
        base = build_case(name, converter_share=share,
                          demand_peak_mw=common_demand_peak(name, SHARES))
        case = dataclasses.replace(
            base, reactive_frac=0.35,
            conv_injection_pu=np.full(len(base.demand_mw), dispatch))
        for form in FORMS:
            outcome = MultiPeriod(case, settings, form).solve()
            if not outcome["feasible"]:
                rows.append(dict(case=name, share=share, dispatch=dispatch,
                                 form=form, feasible=False))
                continue
            before, after, deepest = currents(case, outcome)
            rows.append(dict(
                case=name, share=share, dispatch=dispatch, form=form,
                feasible=True, i_before=before, i_after=after,
                most_negative_p_mw=deepest,
                gamma=(uniform_derating(case, settings)
                       if form == "derated" else np.nan),
                kappa=(derating_reactive_cap(case)
                       if form == "derated" else np.nan)))
            if form in CLAIMS_SAFETY and max(before, after) \
                    > settings.i_continuous + TOL:
                failed.append((name, share, dispatch, form, before, after))

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "current_safety.csv", index=False)
    print(frame.to_string(index=False,
                          float_format=lambda x: f"{x:9.4f}"))
    if failed:
        print("\nFAIL: a formulation that claims safety exceeded the "
              "continuous rating")
        for item in failed:
            print("  ", item)
        return 1
    print("\nPASS: the proposed set and the uniform derating stay inside the "
          "continuous rating before and after activation, on every schedule "
          "cleared here")
    return 0


if __name__ == "__main__":
    sys.exit(main())
