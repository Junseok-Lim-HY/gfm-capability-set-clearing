#!/usr/bin/env python3
"""Brackets on the inertia price along the dispatch sweep.

The factor the abstract quotes is read off the dispatch sweep at e = 1.0, and
the existing dual test brackets only the day schedule, so the number has never
been checked for uniqueness. Section 4.3 already says every inertia row is
degenerate, which is exactly the condition under which a single dual is not
the price.

The reported price is the per-period *sum* of that period's inertia duals, so
the quantity to bracket is that sum, not each row on its own. Raising every
inertia row of one period together by delta and re-clearing gives the right
difference of the sum; lowering them gives the left. That is two solves per
period rather than two per row, and it brackets the number the paper actually
prints.

Two step sizes are used. A bracket that moves with the step is an artefact of
the step; one that does not is the subdifferential.

Writes results/price_brackets.csv. Nothing in the manuscript is touched.
"""
from __future__ import annotations

import collections
import itertools
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linprog

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "model"))

from multiperiod import (MultiPeriod, build_case,  # noqa: E402
                         common_demand_peak)
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
PERIOD = re.compile(r"_t(\d+)")
HOURS = 3.0
STEPS = (1.0, 0.25, 0.01)
CASES = ("ieee39", "rts24")
SHARES = (0.2, 0.4)
DISPATCHES = (0.8, 1.0)

# Held at the case study's values, or the bracket belongs to a different price.
SETTINGS = dict(i_short_term=1.5, priority="reactive", soc_threshold=0.20)
REACTIVE = 0.35


def period_of(name: str) -> int:
    found = PERIOD.search(name)
    return int(found.group(1)) if found else -1


def variant(base, dispatch):
    import dataclasses
    return dataclasses.replace(
        base, conv_injection_pu=np.full(len(base.demand_mw), dispatch),
        reactive_frac=REACTIVE, zonal_reactive=True)


def cost_of(parts, shift=None) -> float:
    """Clear at the given right-hand side and return the cost.

    ``shift`` is {row index: delta on the requirement level}. Requirements are
    written -sum(...) <= -level, so raising a level lowers its b_ub entry.
    The tie-break of MultiPeriod.solve is not needed: it selects among equal
    cost schedules and cannot move the cost this reads.
    """
    a_eq, b_eq, a_ub, b_ub, _, _ = parts["matrices"]
    b_ub = np.array(b_ub, dtype=float)
    for index, delta in (shift or {}).items():
        b_ub[index] -= delta
    result = linprog(parts["cost"], A_ub=a_ub, b_ub=b_ub, A_eq=a_eq,
                     b_eq=b_eq, bounds=parts["bounds"], method="highs-ds")
    return float(result.fun) if result.success else float("nan")


def bracket(case, form, step):
    """Quoted price and its left and right brackets, per period then averaged."""
    market = MultiPeriod(case, Capability(**SETTINGS), form)
    outcome = market.solve()
    if not outcome["feasible"]:
        return {}
    a_eq, b_eq, a_ub, b_ub, req_names, n_req = market.build()
    parts = {"matrices": (a_eq, b_eq, a_ub, b_ub, req_names, n_req),
             "cost": market.objective(), "bounds": market.bounds()}
    base_cost = cost_of(parts)

    rows_by_period: dict[int, list[int]] = collections.defaultdict(list)
    for index, name in enumerate(req_names[:n_req]):
        if name.startswith("inertia"):
            rows_by_period[period_of(name)].append(index)

    duals_by_period: dict[int, float] = collections.defaultdict(float)
    for name, value in outcome["prices"].items():
        if name.startswith("inertia"):
            duals_by_period[period_of(name)] += float(value)

    left, right, quoted = [], [], []
    for period, rows in sorted(rows_by_period.items()):
        up = cost_of(parts, {i: step for i in rows})
        down = cost_of(parts, {i: -step for i in rows})
        left.append((base_cost - down) / step)
        right.append((up - base_cost) / step)
        quoted.append(duals_by_period[period])
    return {"quoted": float(np.mean(quoted)) / HOURS,
            "left": float(np.mean(left)) / HOURS,
            "right": float(np.mean(right)) / HOURS}


def main() -> None:
    peaks = {name: common_demand_peak(name, SHARES) for name in CASES}
    rows = []
    for name, share, dispatch, step in itertools.product(
            CASES, SHARES, DISPATCHES, STEPS):
        base = build_case(name, converter_share=share,
                          demand_peak_mw=peaks[name])
        case = variant(base, dispatch)
        record = dict(case=name, share=share, dispatch=dispatch, step=step)
        for form in ("box", "current"):
            got = bracket(case, form, step)
            for key, value in got.items():
                record[f"{form}_{key}"] = value
        rows.append(record)
        print(f"  {name} {share} e={dispatch} step={step}: "
              f"box {record.get('box_left', np.nan):7.3f}"
              f" ..{record.get('box_right', np.nan):7.3f}"
              f"   set {record.get('current_left', np.nan):7.3f}"
              f" ..{record.get('current_right', np.nan):7.3f}", flush=True)

    frame = pd.DataFrame(rows)
    frame["overlap"] = ~((frame.box_left > frame.current_right)
                         | (frame.current_left > frame.box_right))
    frame["ratio_quoted"] = frame.box_quoted / frame.current_quoted
    frame["ratio_low"] = frame.box_left / frame.current_right
    frame["ratio_high"] = frame.box_right / frame.current_left
    frame.to_csv(RESULTS / "price_brackets.csv", index=False)
    print()
    print(frame.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
