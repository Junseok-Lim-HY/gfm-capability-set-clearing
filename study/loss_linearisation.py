#!/usr/bin/env python3
"""What happens when the clearing is charged for the losses it causes.

The dc balance carries no losses, so the slack machine invents them after the
fact. Writing them in is easy enough to do and keeps the programme linear: the
losses are a smooth function of the injections, the power flow gives its
gradient, and the balance becomes generation equals demand plus a linear
expression in the decisions. The energy price then carries a marginal-loss
component, which is where that component belongs.

Easy to do is not the same as sound, and this measures the difference. A
gradient is a statement about a neighbourhood, and a linear programme handed a
gradient treats it as a statement about everything. If reducing losses is
worth money, the clearing will move in the direction the gradient points until
some other constraint stops it, and nothing in the programme knows that the
gradient stopped being true a long way back.

So the loss expression is evaluated at the schedule the clearing returns and
compared with what the power flow says at the same schedule. Writes
results/loss_linearisation.csv.
"""
from __future__ import annotations

import dataclasses
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import ac_clearing  # noqa: E402
from multiperiod import (MultiPeriod, ac_voltage_pass,  # noqa: E402
                         build_case, common_demand_peak)
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
CASES = ("ieee39", "rts24")
SHARES = (0.2, 0.4)
DISPATCH = 0.8


def main() -> None:
    settings = Capability(i_short_term=1.5, priority="reactive",
                          soc_threshold=0.20)
    rows = []
    for name, share in itertools.product(CASES, SHARES):
        base = build_case(name, converter_share=share,
                          demand_peak_mw=common_demand_peak(name))
        case = dataclasses.replace(
            base, conv_injection_pu=np.full(len(base.demand_mw), DISPATCH))

        model = MultiPeriod(case, settings, "current")
        plain = model.solve()
        nets = []
        ac_voltage_pass(case, plain, solved=nets)
        losses = ac_clearing.loss_rows(model, case, plain, nets)

        # The fit, checked where it was taken. If this does not reproduce the
        # power flow the rest of the experiment is measuring the wrong thing.
        fitted = np.array([float(r @ plain["x"] + c) for r, c in losses])
        actual = np.array([float(n.res_line.pl_mw.sum()
                                 + n.res_trafo.pl_mw.sum()) for n in nets])

        charged = MultiPeriod(case, settings, "current", losses=losses).solve()
        claimed = np.array([float(r @ charged["x"] + c) for r, c in losses])

        rows.append(dict(
            case=name, share=share,
            actual_min=float(actual.min()), actual_max=float(actual.max()),
            fit_error_mw=float(np.abs(fitted - actual).max()),
            claimed_min=float(claimed.min()), claimed_max=float(claimed.max()),
            periods_negative=int((claimed < 0.0).sum()),
            periods=int(len(claimed)),
            cost_before=plain["cost"], cost_after=charged["cost"],
            cost_change_pct=100.0 * (charged["cost"] - plain["cost"])
            / plain["cost"]))
        row = rows[-1]
        print(f"{name} {share:.0%}: 조류의 손실 "
              f"{row['actual_min']:.1f}~{row['actual_max']:.1f} MW  "
              f"(식의 오차 {row['fit_error_mw']:.2e})  "
              f"물린 뒤 식이 말하는 손실 "
              f"{row['claimed_min']:.1f}~{row['claimed_max']:.1f} MW  "
              f"음수 {row['periods_negative']}/{row['periods']}  "
              f"비용 {row['cost_change_pct']:+.3f}%")

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "loss_linearisation.csv", index=False)
    print("\n손실을 물리면 청산은 손실을 줄이는 쪽으로 갑니다. 기울기가 "
          "그 방향으로만 참이라는 것은 프로그램이 모릅니다.")
    print(f"\nwrote {RESULTS / 'loss_linearisation.csv'}")


if __name__ == "__main__":
    sys.exit(main())
