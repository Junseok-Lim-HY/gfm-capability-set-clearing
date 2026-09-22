#!/usr/bin/env python3
"""Rating convention against comparator bound, side by side.

The review's fourth requirement has two halves. The converter rating should be
a converter's, not the corner of a synchronous machine's P-Q box; and the
active-power comparator should be reported at both of its natural bounds,
P-bar = S and P-bar = P_sch, in parallel, rather than one being chosen as the
baseline and the other left to a sensitivity row.

So the table is a cross. Rows are rating conventions, each a ratio of the
active limit to the apparent rating with a physical reading:

    P/S = 1.00   an inverter sized to its active power, with no reactive
                 capability at full output;
    P/S = 0.95   the 0.95 power-factor capability that transmission grid
                 codes ask of inverter-based plant at rated active power;
    P/S = 0.90   the 0.90 requirement of the stricter codes, and close to a
                 storage inverter sized for full four-quadrant operation;
    case         S = sqrt(Pmax^2 + Qmax^2) from the machine being replaced,
                 which is what the manuscript has used (0.86--0.95 here).

Columns are the two comparator bounds. Each cell is the cost of the proposed
capability set relative to that comparator, in per cent; the sign, and whether
it is stable across the cross, is what the sign condition predicts, and the
magnitude is what depends on the convention.

Writes results/rating_cross.csv.
"""
from __future__ import annotations

import dataclasses
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "model"))

import prices  # noqa: E402
from multiperiod import (MultiPeriod, build_case,  # noqa: E402
                         common_demand_peak)
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
CASES = ("ieee39", "rts24")
SHARES = (0.2, 0.4)
DISPATCH = 0.8
RATIOS = (1.00, 0.95, 0.90, None)          # None: the case's own
BOUNDS = ("rating", "pmax")


def main() -> None:
    peaks = {name: common_demand_peak(name, SHARES) for name in CASES}
    rows = []
    for name, share in itertools.product(CASES, SHARES):
        base = build_case(name, converter_share=share,
                          demand_peak_mw=peaks[name])
        for ratio in RATIOS:
            rating = (base.conv_rating_mva if ratio is None
                      else base.conv_pmax_mw / ratio)
            case = dataclasses.replace(
                base, conv_rating_mva=rating, reactive_frac=0.35,
                conv_injection_pu=np.full(len(base.demand_mw), DISPATCH))
            proposed = MultiPeriod(
                case, Capability(i_short_term=1.5, priority="reactive",
                                 soc_threshold=0.20), "current").solve()
            for bound in BOUNDS:
                settings = Capability(i_short_term=1.5, priority="reactive",
                                      soc_threshold=0.20, box_bound=bound)
                box = MultiPeriod(case, settings, "box").solve()
                row = dict(case=name, share=share,
                           p_over_s="case" if ratio is None else ratio,
                           bound="S" if bound == "rating" else "P_sch",
                           feasible=bool(proposed["feasible"]
                                         and box["feasible"]))
                if row["feasible"]:
                    row.update(
                        cost_box=box["cost"], cost_set=proposed["cost"],
                        gap_pct=100.0 * (proposed["cost"] - box["cost"])
                        / box["cost"],
                        price_inertia_box=prices.price(box["prices"],
                                                       "inertia"),
                        price_inertia_set=prices.price(proposed["prices"],
                                                       "inertia"))
                rows.append(row)
                print(row, flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "rating_cross.csv", index=False)
    wide = frame.pivot_table(index=["p_over_s", "bound"],
                             columns=["case", "share"], values="gap_pct")
    print("\n제안 집합의 비용, 비교모형 대비 (%)\n")
    print(wide.to_string(float_format=lambda v: f"{v:8.2f}"))
    print("\nwrote results/rating_cross.csv")


if __name__ == "__main__":
    sys.exit(main())
