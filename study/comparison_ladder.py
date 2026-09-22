#!/usr/bin/env python3
"""Four models, not two.

Comparing the proposed capability set against a single active-power bound
invites the reply that the comparator was chosen to lose. The reply is worth
taking seriously, because the active-power bound really is the crudest thing
anyone writes down, and a result measured against it says less than a result
measured against what a careful planner would do instead.

So the comparison becomes a ladder, and each rung adds one thing:

    box       P + R + H <= Pbar             no reactive, no voltage, no band
    apparent  |(P + R + H, Q)| <= S         reactive, still no voltage or band
    derated   P + R + H <= k S              one active limit, set to be safe
    current   the proposed set              reactive, voltage, band, priority

The derating factor k is computed from the case rather than chosen. It is what
is left of the continuous current for active power once the converter is
carrying its reactive obligation, at the worst terminal voltage the case
produces -- so by construction the derated bound is nowhere optimistic. That
is what makes it the strongest of the three comparators and the one the result
has to survive.

All three comparators are polygonised at the same number of sides as the
proposed set wherever they carry a disc, so that a step between rungs is a
step in modelling and not a step in approximation error.

Writes results/comparison_ladder.csv.
"""
from __future__ import annotations

import dataclasses
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "model"))

from multiperiod import (HOURS, MultiPeriod, build_case,  # noqa: E402
                         common_demand_peak, uniform_derating)
from prices import price  # noqa: E402
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"

CASES = ("ieee39", "rts24")
SHARES = (0.2, 0.4)
DISPATCH = (0.4, 0.6, 0.8, 1.0)
LADDER = ("box", "apparent", "derated", "current")
BASE = dict(i_short=1.5, reactive=0.35, threshold=0.20, priority="reactive")


def main() -> None:
    settings = Capability(i_short_term=BASE["i_short"],
                          priority=BASE["priority"],
                          soc_threshold=BASE["threshold"])
    rows = []
    for name, share in itertools.product(CASES, SHARES):
        base = build_case(name, converter_share=share,
                          demand_peak_mw=common_demand_peak(name, SHARES))
        base = dataclasses.replace(base, reactive_frac=BASE["reactive"])
        for dispatch in DISPATCH:
            loaded = dataclasses.replace(
                base,
                conv_injection_pu=np.full(len(base.demand_mw), dispatch))
            factor = uniform_derating(loaded, settings)
            for form in LADDER:
                outcome = MultiPeriod(loaded, settings, form).solve()
                row = dict(case=name, share=share, dispatch=dispatch,
                           form=form, derating=factor,
                           feasible=outcome["feasible"])
                if outcome["feasible"]:
                    row.update(
                        cost=outcome["cost"],
                        price_inertia=price(outcome["prices"], "inertia"),
                        price_reserve=price(outcome["prices"], "reserve"),
                        curtailed_mwh=float(outcome["curt"].sum()) * HOURS)
                rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "comparison_ladder.csv", index=False)

    live = frame[frame.feasible]
    print(f"디레이팅 계수: {live.derating.min():.4f} ~ {live.derating.max():.4f}\n")
    for column, title in (("cost", "비용"),
                          ("price_inertia", "관성 가격"),
                          ("curtailed_mwh", "출력제한 (MWh)")):
        wide = live.pivot_table(index=["case", "share", "dispatch"],
                                columns="form", values=column)[list(LADDER)]
        print(f"=== {title}")
        print(wide.to_string(float_format=lambda x: f"{x:12.2f}"))
        print()

    wide = live.pivot_table(index=["case", "share", "dispatch"],
                            columns="form", values="cost")
    print("=== 제안 모델 대비 비용차 (%), 음수면 제안 모델이 더 싸다")
    gap = pd.DataFrame({f: 100.0 * (wide["current"] - wide[f]) / wide[f]
                        for f in ("box", "apparent", "derated")})
    print(gap.to_string(float_format=lambda x: f"{x:9.4f}"))
    print("\n각 비교 모델에 대해 부호가 한결같은가:")
    for f in ("box", "apparent", "derated"):
        positive, negative = int((gap[f] > 0).sum()), int((gap[f] < 0).sum())
        print(f"  {f:9s}: 양수 {positive:2d}, 음수 {negative:2d}, "
              f"0 {len(gap) - positive - negative:2d}  "
              f"(범위 {gap[f].min():+.4f} .. {gap[f].max():+.4f} %)")
    print("\nwrote results/comparison_ladder.csv")


if __name__ == "__main__":
    sys.exit(main())
