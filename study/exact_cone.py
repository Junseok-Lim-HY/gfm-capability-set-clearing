#!/usr/bin/env python3
"""What the polygon costs, measured against the cone rather than argued.

Section 2.5 bounds the polyhedral approximation two ways: analytically, by
noting that an inscribed polygon can only withhold capability, so the reported
cost difference understates the exact one; and empirically, by refining the
number of sides until the answer stops moving. Neither actually solves the
exact problem. Refining m is a proxy for the cone and a good one, but a
referee is entitled to ask for the cone itself.

So here it is. The clearing is built once with the capability faces withheld
and everything else -- requirements, storage, network, the state-of-charge
gate on the overload band, the reactive priority equality -- left exactly as
the linear programme writes it. Second-order cones then go on the same
variables:

    |(P + R, Q)|        <= V I^c S
    |(P + chi R + H, Q~)| <= V I^c S + a V (I^s - I^c) S

These are Eqs. (3) and (4) without the cos(pi/m) the polygon carries. Building
the conic problem from the linear programme's own matrices rather than writing
it out again matters: if the two were transcribed separately, a difference
between them could be a difference in transcription, and there would be no way
to tell which.

Writes results/exact_cone.csv.
"""
from __future__ import annotations

import dataclasses
import itertools
import pathlib
import sys

import cvxpy as cp
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "model"))

import capability as cap  # noqa: E402
from multiperiod import (MultiPeriod, build_case,  # noqa: E402
                         common_demand_peak)
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"

CASES = ("ieee39", "rts24")
SHARES = (0.2, 0.4)
DISPATCH = (0.8, 1.0)
SIDES = (24, 96, 384)
BASE = dict(i_short=1.5, reactive=0.35, threshold=0.20, priority="reactive")


def solve_cone(case, settings) -> dict:
    """The exact conic clearing, on the linear programme's own matrices."""
    model = MultiPeriod(case, settings, "cone")
    a_eq, b_eq, a_ub, b_ub, _, _ = model.build()
    x = cp.Variable(model.size)
    lower = np.array([b[0] for b in model.bounds()], dtype=float)
    upper = np.array([b[1] for b in model.bounds()], dtype=float)

    constraints = [a_ub @ x <= b_ub, a_eq @ x == b_eq, x >= lower, x <= upper]
    index, injection = model.index, model.injection
    for t, i in itertools.product(range(model.T), range(model.n)):
        v = float(case.conv_voltage_pu[t, i])
        s = float(case.conv_rating_mva[i])
        radius = v * settings.i_continuous * s
        # The short-term cone is taken at the voltage of the second after the
        # outage at both of its ends, as the linear programme now does.
        v_short = (float(case.conv_voltage_short_pu[t, i])
                   if case.conv_voltage_short_pu is not None else v)
        radius_short = v_short * settings.i_continuous * s
        band = v_short * (settings.i_short_term - settings.i_continuous) * s
        net = (x[index["pd"][t, i]] - x[index["pc"][t, i]]
               - x[index["curt"][t, i]] + injection[t, i])
        constraints += [
            # before the reserve is called, which is the larger current for a
            # charging unit, and after it
            cp.norm(cp.hstack([net, x[index["q"][t, i]]]), 2) <= radius,
            cp.norm(cp.hstack([net + x[index["r"][t, i]],
                               x[index["q"][t, i]]]), 2) <= radius,
            cp.norm(cp.hstack([net + case.reserve_overlap * x[index["r"][t, i]]
                               + x[index["h"][t, i]],
                               x[index["qe"][t, i]]]), 2)
            <= radius_short + band * x[index["a"][t, i]],
        ]

    problem = cp.Problem(cp.Minimize(model.objective() @ x), constraints)
    problem.solve(solver=cp.CLARABEL)
    return {"feasible": problem.status in ("optimal", "optimal_inaccurate"),
            "cost": float(problem.value) if problem.value is not None
            else float("nan"), "status": problem.status}


def main() -> None:
    settings = Capability(i_short_term=BASE["i_short"],
                          priority=BASE["priority"],
                          soc_threshold=BASE["threshold"])
    rows = []
    for name, share, dispatch in itertools.product(CASES, SHARES, DISPATCH):
        base = build_case(name, converter_share=share,
                          demand_peak_mw=common_demand_peak(name, SHARES))
        loaded = dataclasses.replace(
            base, reactive_frac=BASE["reactive"],
            conv_injection_pu=np.full(len(base.demand_mw), dispatch))

        box = MultiPeriod(loaded, settings, "box").solve()
        cone = solve_cone(loaded, settings)
        row = dict(case=name, share=share, dispatch=dispatch,
                   cost_box=box["cost"] if box["feasible"] else np.nan,
                   cost_cone=cone["cost"], status=cone["status"])
        for sides in SIDES:
            tighter = dataclasses.replace(settings, polygon_sides=sides)
            got = MultiPeriod(loaded, tighter, "current").solve()
            row[f"cost_m{sides}"] = got["cost"] if got["feasible"] else np.nan
        rows.append(row)

    frame = pd.DataFrame(rows)
    for sides in SIDES:
        frame[f"gap_m{sides}"] = 100.0 * (frame[f"cost_m{sides}"]
                                          - frame.cost_cone) / frame.cost_cone
        frame[f"delta_m{sides}"] = 100.0 * (frame[f"cost_m{sides}"]
                                            - frame.cost_box) / frame.cost_box
    frame["delta_cone"] = 100.0 * (frame.cost_cone
                                   - frame.cost_box) / frame.cost_box
    frame.to_csv(RESULTS / "exact_cone.csv", index=False)

    key = ["case", "share", "dispatch"]
    print("다각형이 원뿔보다 얼마나 비싼가 (%), 안쪽 근사이므로 0 이상이어야 한다")
    print(frame.set_index(key)[[f"gap_m{m}" for m in SIDES]]
          .to_string(float_format=lambda x: f"{x:9.5f}"))
    print("\n보고되는 비용차 delta (%), 다각형과 원뿔")
    print(frame.set_index(key)[[f"delta_m{m}" for m in SIDES] + ["delta_cone"]]
          .to_string(float_format=lambda x: f"{x:9.4f}"))
    worst = frame[[f"gap_m{m}" for m in SIDES]].max()
    print(f"\n다각형 초과 비용 최대: m=24 {worst['gap_m24']:.5f} %, "
          f"m=96 {worst['gap_m96']:.5f} %, m=384 {worst['gap_m384']:.5f} %")
    understates = (frame.delta_m24 >= frame.delta_cone - 1e-9).all()
    print(f"m=24 의 delta 가 원뿔의 delta 를 과소평가하는가 (2.5절의 주장): "
          f"{'예' if understates else '아니오'}")
    if not understates:
        print(frame.set_index(key)[["delta_m24", "delta_cone"]]
              .to_string(float_format=lambda x: f"{x:9.4f}"))
    print("\nwrote results/exact_cone.csv")


if __name__ == "__main__":
    sys.exit(main())
