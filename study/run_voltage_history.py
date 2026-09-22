#!/usr/bin/env python3
"""Per-round trace of the voltage fixed point, kept for the figure.

`run_case_study.py` reports only the endpoints of the iteration. The path is
what answers the referee's question, so it is written out separately here.

Three residuals are tracked, and the third is the one that matters. Voltage
and cost can both settle while the schedule underneath them is still moving,
because cost is flat across alternative optima by construction. The primal
residual measures the decision vector itself.

The clearing is made single-valued first, by a lexicographic tie-break, so
that there is a map to converge to at all rather than a correspondence. With
that in place RTS-24 reaches a fixed point in every quantity. IEEE 39-bus does
not, at any damping from 0.5 down to 0.05, and its cost is identical at all of
them: the cost is a fixed point, the schedule is not. That is reported as a
failure to converge rather than dressed up, because a cost that does not move
while the dispatch producing it does is not evidence that the voltage
objection has been answered.
"""
from __future__ import annotations

import dataclasses
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "model"))

from multiperiod import (build_case, common_demand_peak,  # noqa: E402
                         solve_to_voltage_fixed_point)
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
TAIL_FROM = 5


def main() -> None:
    settings = Capability(i_short_term=1.5, priority="reactive",
                          soc_threshold=0.20)
    rows = []
    for name, share, form in itertools.product(
            ("ieee39", "rts24"), (0.2, 0.4), ("box", "current")):
        base = build_case(name, converter_share=share,
                          demand_peak_mw=common_demand_peak(name, (0.2, 0.4)))
        case = dataclasses.replace(
            base, conv_injection_pu=np.full(len(base.demand_mw), 0.8))
        outcome = solve_to_voltage_fixed_point(case, settings, form)
        if not outcome["feasible"]:
            print(f"{name} {share} {form}: infeasible")
            continue
        start = outcome["history"][0]["cost"]
        for record in outcome["history"]:
            rows.append(dict(case=name, share=share, form=form,
                             converged=outcome["converged"], **record,
                             cost_rel=100.0 * (record["cost"] - start) / start))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "voltage_history.csv", index=False)

    # The tail is what the reported numbers have to be stable over. Five rounds
    # in, any transient from the seed voltages is gone, so anything still
    # moving after that is the iteration itself.
    tail = frame[frame["round"] >= TAIL_FROM]
    summary = tail.groupby(["case", "share", "form"]).agg(
        rounds=("round", "max"),
        converged=("converged", "last"),
        v_resid=("voltage_move", "last"),
        cost_resid=("cost_drift", "last"),
        x_resid=("primal_move", "last"),
        cost_spread=("cost", lambda s: s.max() - s.min()),
        price_spread=("price_inertia", lambda s: s.max() - s.min()),
    )
    summary["rounds"] += 1
    print(summary.to_string(float_format=lambda v: f"{v:10.6f}"))

    stalled = summary[~summary["converged"].astype(bool)]
    if len(stalled):
        print(f"\n{len(stalled)} of {len(summary)} did not reach a fixed "
              f"point. Worst voltage residual {stalled.v_resid.max():.4f} pu, "
              f"worst primal residual {stalled.x_resid.max():.3f}. Over those "
              f"runs the cleared cost moves by at most "
              f"{stalled.cost_spread.max():.2e}, so the cost is a fixed point "
              f"where the schedule is not. Those configurations are reported "
              f"as unresolved rather than as answering the voltage objection.")
    print(f"\nwrote {RESULTS / 'voltage_history.csv'} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
