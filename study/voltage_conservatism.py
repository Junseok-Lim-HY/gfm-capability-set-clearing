#!/usr/bin/env python3
"""Which way does the pre-contingency voltage assumption push the result?

study/dynamic_check.py shows the terminal voltage during an event sitting
below the pre-contingency value the capability set is written at, by a tenth
of a per cent on a stiff bus and by nine and a half on a weak one. That is a
limitation. Whether it is one that threatens the paper's conclusion or one
that reinforces it is a separate question with a computable answer.

The tempting argument is that it reinforces: the capability set carries the
voltage and the active-power box does not, so lowering the voltage shrinks one
model and leaves the other alone, and the gap between them widens. That
argument is wrong, and the run below is what shows it. The box is the dearer
of the two. Shrinking the capability set raises its cost toward the box's, so
the gap closes rather than opens. The reported difference is an upper bound,
not a lower one, and the paper has to say so.

Two ways of applying the depression are reported because they answer different
questions.

Lowering the voltage in both faces is the cruder reading, and it overstates
the effect: the continuous constraint is a pre-disturbance constraint, so the
pre-disturbance voltage is the right voltage for it and there is nothing wrong
with using it.

Lowering it only in the short-term face is the faithful one. That face has
radius V I^s S cos(pi/m), so depressing V there by a factor is arithmetically
the same as depressing I^s by it, which needs no change to the model.

Writes results/voltage_conservatism.csv.
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
                         common_demand_peak)
from prices import price  # noqa: E402
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"

CASES = ("ieee39", "rts24")
SHARES = (0.2, 0.4)
# The depressions dynamic_check.py reports with the grid itself undisturbed:
# a stiff bus, an ordinary one at its mean and at its worst, and a weak one at
# both. Every one of these is produced by the converter's own current.
DROPS = (0.0, 0.002, 0.013, 0.023, 0.055, 0.095)
WHERE = ("both faces", "short-term face only")
# The short-term face only binds when the converter is loaded. At a dispatch
# of 0.8 it is slack, the inertia price is set by the offers rather than by
# the constraint, and depressing the voltage moves nothing -- which would
# read as reassuring and would mean nothing. The paper's headline price
# ratio is quoted at full dispatch, so the sweep has to go there.
DISPATCHES = (0.8, 1.0)
BASE = dict(i_short=1.5, reactive=0.35, threshold=0.20,
            priority="reactive")


def cleared(case, form: str, drop: float, where: str) -> dict:
    """Clear one model with the depression applied where `where` says."""
    i_short = BASE["i_short"]
    if where == "short-term face only":
        i_short *= 1.0 - drop
    else:
        case = dataclasses.replace(
            case, conv_voltage_pu=case.conv_voltage_pu * (1.0 - drop))
    settings = Capability(i_short_term=i_short, priority=BASE["priority"],
                          soc_threshold=BASE["threshold"])
    return MultiPeriod(case, settings, form).solve()


def main() -> None:
    rows = []
    for name, share in itertools.product(CASES, SHARES):
        base = build_case(name, converter_share=share,
                          demand_peak_mw=common_demand_peak(name, SHARES))
        base = dataclasses.replace(
            base,
            reactive_frac=BASE["reactive"])
        for dispatch, where, drop in itertools.product(
                DISPATCHES, WHERE, DROPS):
            loaded = dataclasses.replace(
                base, conv_injection_pu=np.full(len(base.demand_mw),
                                               dispatch))
            got = {form: cleared(loaded, form, drop, where)
                   for form in ("box", "current")}
            row = dict(case=name, share=share, dispatch=dispatch,
                       applied_to=where, drop=drop,
                       feasible=all(o["feasible"] for o in got.values()))
            if row["feasible"]:
                box, current = got["box"], got["current"]
                p_box = price(box["prices"], "inertia")
                p_cur = price(current["prices"], "inertia")
                row.update(cost_box=box["cost"], cost_current=current["cost"],
                           delta_pct=100.0 * (current["cost"] - box["cost"])
                           / box["cost"],
                           price_box=p_box, price_current=p_cur,
                           price_ratio=p_box / p_cur if p_cur > 0 else np.nan)
            rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "voltage_conservatism.csv", index=False)

    live = frame[frame.feasible]
    for column, title in (("delta_pct", "비용차 (%)"),
                          ("price_ratio", "관성 가격비 (박스/능력집합)")):
        for where in WHERE:
            part = live[live.applied_to == where]
            if part[column].isna().all():
                continue
            print(f"\n=== {title} — 전압 강하를 {where}에 적용")
            print(part.pivot_table(index=["case", "share", "dispatch"], columns="drop",
                                   values=column)
                  .to_string(float_format=lambda x: f"{x:8.4f}"))

    faithful = live[live.applied_to == "short-term face only"]
    table = faithful.pivot_table(index=["case", "share", "dispatch"], columns="drop",
                                 values="delta_pct")
    lo, hi = table.columns.min(), table.columns.max()
    print("\n단기 제약에만 걸었을 때, 보고된 비용차가 어떻게 되나:")
    for index, row in table.iterrows():
        keeps = abs(row[0.013] / row[lo]) if row[lo] else np.nan
        print(f"  {index}: 무강하 {row[lo]:8.4f}%   "
              f"보통계통(1.3%) {row[0.013]:8.4f}% ({keeps:5.2f}배)   "
              f"약계통(9.5%) {row[hi]:8.4f}%")
    if len(live) < len(frame):
        print(f"\n해가 없어진 설정 {len(frame) - len(live)}개")
        print(frame[~frame.feasible][["case", "share", "dispatch", "applied_to", "drop"]]
              .to_string(index=False))
    print("\nwrote results/voltage_conservatism.csv")


if __name__ == "__main__":
    sys.exit(main())
