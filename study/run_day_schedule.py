#!/usr/bin/env python3
"""One representative day, period by period, under both capability models.

The aggregate figures say the two formulations differ; this says where in the
day they differ and in what. Fleet totals are written per period for both
models on one configuration, chosen because it is the one in which the
converters are loaded enough for the shapes to disagree.
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "model"))

from multiperiod import (HOURS, MultiPeriod, build_case,  # noqa: E402
                         common_demand_peak, optimal_face_profile)
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"

CASE, SHARE, DISPATCH = "rts24", 0.4, 0.8


def main() -> None:
    base = build_case(CASE, converter_share=SHARE,
                      demand_peak_mw=common_demand_peak(CASE, (0.2, 0.4)))
    case = dataclasses.replace(
        base, conv_injection_pu=np.full(len(base.demand_mw), DISPATCH))
    settings = Capability(i_short_term=1.5, priority="reactive",
                          soc_threshold=0.20)

    rows = []
    for form in ("box", "current"):
        out = MultiPeriod(case, settings, form).solve()
        if not out["feasible"]:
            raise SystemExit(f"{form} infeasible")

        # The reserve a period reports is one point on the cost-optimal face,
        # and the face is wide here: reserve moves between converters, and
        # between hours through the store, at identical cost. So the interval
        # is measured alongside the point, and the text quotes whichever the
        # data supports.
        span = optimal_face_profile(case, settings, form, "r")
        for t in range(len(case.demand_mw)):
            rows.append(dict(
                form=form, period=t, hour=t * HOURS,
                reserve_lo_mw=float(span[t][0]),
                reserve_hi_mw=float(span[t][1]),
                demand_mw=float(case.demand_mw[t]),
                inertia_mw=float(out["h"][t].sum()),
                reserve_mw=float(out["r"][t].sum()),
                reactive_mvar=float(out["q"][t].sum()),
                curtailed_mw=float(out["curt"][t].sum()),
                discharge_mw=float(out["pd"][t].sum()),
                charge_mw=float(out["pc"][t].sum()),
                synchronous_mw=float(out["pg"][t].sum()),
                mean_soc=float(out["soc"][t].mean()),
                mean_availability=float(out["availability"][t].mean()),
            ))
    frame = pd.DataFrame(rows)
    frame.insert(0, "case", CASE)
    frame.insert(1, "share", SHARE)
    frame.to_csv(RESULTS / "day_schedule.csv", index=False)

    print(f"{CASE}, converter share {SHARE}, dispatch {DISPATCH} pu\n")
    for column in ("inertia_mw", "curtailed_mw", "reserve_mw", "mean_soc"):
        print(f"{column}")
        print(frame.pivot_table(index="period", columns="form", values=column)
              .to_string(float_format=lambda v: f"{v:9.2f}"))
        print()

    # What survives the choice of optimum: the least the capability set holds
    # anywhere on its face, less the most the bound holds anywhere on its.
    box = frame[frame.form == "box"].set_index("period")
    setf = frame[frame.form == "current"].set_index("period")
    excess = setf.reserve_lo_mw - box.reserve_hi_mw
    print("guaranteed reserve excess (set_lo - box_hi), MW")
    print(excess.to_string(float_format=lambda v: f"{v:9.2f}"))
    print(f"\nwrote {RESULTS / 'day_schedule.csv'} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
