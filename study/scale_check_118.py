#!/usr/bin/env python3
"""One scale check on the IEEE 118-bus system, under the dc approximation.

The two benchmark systems of the case study are small. This runs the same
comparison -- active-power bound against capability set, everything else
identical -- on the 118-bus case as shipped with pandapower, with converters
rated at P/S = 0.95, at two converter shares, two RoCoF limits and two
dispatch levels. It is a check of sign and of when the inertia requirement
binds on a larger system, not a third case study: no ac validation is run on
it.

Writes results/scale_check_118.csv.
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


def main() -> None:
    settings = Capability(i_short_term=1.5, priority="reactive",
                          soc_threshold=0.20)
    peak = common_demand_peak("ieee118")
    rows = []
    for share, rocof, dispatch in itertools.product((0.2, 0.4), (1.0, 0.5),
                                                    (0.8, 1.0)):
        base = build_case("ieee118", converter_share=share,
                          demand_peak_mw=peak)
        case = dataclasses.replace(
            base, conv_rating_mva=base.conv_pmax_mw / 0.95,
            rocof_hz_s=rocof,
            conv_injection_pu=np.full(len(base.demand_mw), dispatch))
        box = MultiPeriod(case, settings, "box").solve()
        cur = MultiPeriod(case, settings, "current").solve()
        row = dict(share=share, rocof=rocof, dispatch=dispatch,
                   converters=len(case.conv_rating_mva),
                   synchronous=len(case.sync_rating_mva),
                   feasible=bool(box["feasible"] and cur["feasible"]))
        if row["feasible"]:
            row.update(
                delta_pct=100.0 * (box["cost"] - cur["cost"]) / box["cost"],
                price_inertia_box=prices.price(box["prices"], "inertia"),
                price_inertia_set=prices.price(cur["prices"], "inertia"))
        rows.append(row)
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "scale_check_118.csv", index=False)
    print("wrote results/scale_check_118.csv")


if __name__ == "__main__":
    sys.exit(main())
