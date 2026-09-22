#!/usr/bin/env python3
"""How far the cleared response runs past the scheduled active bound.

Eq. (14) bounds the scheduled net injection at P^sch. It does not bound the
response: the sustained pair P + R is held against the continuous current
circle, and the short-term triple against the overload circle, so both may
exceed P^sch wherever the disc reaches further than the bound does. The
manuscript says so, and this measures by how much, because a claim about a
model's definitions is worth no more than the numbers that follow from them.

The measurement is also the answer to an obvious objection --- that the
capability set is being allowed something the comparison bound is not. It is
not: the comparison bound is drawn at S, and S exceeds P^sch too, so both
formulations put P + R above it. The counts below are of converter-periods
where they do.
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "model"))

from multiperiod import (MultiPeriod, build_case,  # noqa: E402
                         common_demand_peak)
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
CASES = ("ieee39", "rts24")
SHARES = (0.2, 0.4)
DISPATCH = 0.8


def main() -> None:
    settings = Capability(i_short_term=1.5, priority="reactive",
                          soc_threshold=0.20)
    rows = []
    for name in CASES:
        peak = common_demand_peak(name, SHARES)
        for share in SHARES:
            base = build_case(name, converter_share=share,
                              demand_peak_mw=peak)
            case = dataclasses.replace(
                base,
                conv_injection_pu=np.full(len(base.demand_mw), DISPATCH))
            for form in ("box", "current"):
                model = MultiPeriod(case, settings, form)
                out = model.solve()
                if not out["feasible"]:
                    raise SystemExit(f"{name} {share} {form} infeasible")
                bound = case.conv_pmax_mw
                p = (model.injection + out["pd"] - out["pc"] - out["curt"])
                sustained = (p + out["r"]) / bound
                short = (p + case.reserve_overlap * out["r"]
                         + out["h"]) / bound
                rows.append(dict(
                    case=name, share=share, form=form,
                    max_p=float((p / bound).max()),
                    max_p_plus_r=float(sustained.max()),
                    max_short_term=float(short.max()),
                    periods_over=int((sustained > 1.0 + 1e-9).sum()),
                    converter_periods=int(sustained.size),
                ))

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "scheduled_bound.csv", index=False)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print(f"\nwrote {RESULTS / 'scheduled_bound.csv'} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
