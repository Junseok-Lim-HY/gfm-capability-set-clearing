#!/usr/bin/env python3
"""Scan the reactive requirement instead of bisecting it.

study/reactive_headroom.py bisected for "the largest requirement the network
carries", and bisection is only meaningful if carrying is monotone in the
requirement: if a value works, every smaller value must work. That assumption
was never checked and it is false.

study/reactive_rule.py found the counterexample without looking for it. On
IEEE 39-bus at 40 per cent converter share the requirement of 0.35 used
throughout the paper closes the voltage band exactly, and 0.2701 -- a smaller
requirement -- does not. On RTS-24 at 20 per cent, 0.2351 closes while the
bisection had reported 0.2297 as the largest that does.

The mechanism is not mysterious once stated. The requirement changes the
schedule, not just the reactive dispatch: asking converters for less reactive
power moves the duty to synchronous machines somewhere else on the network,
and that moves flows and voltages. There is no reason for the result to be
monotone and it is not.

So the quantity "the largest requirement that works" does not exist, and any
number reported for it -- including the two this study has already printed --
is an artefact of assuming it did. This scans a grid instead and reports the
set of values that close, which is what there is.

Writes results/reactive_scan.csv.
"""
from __future__ import annotations

import copy
import dataclasses
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import voltage_control  # noqa: E402
from multiperiod import (MultiPeriod, ac_voltage_pass,  # noqa: E402
                         build_case, common_demand_peak)
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
CASES = ("ieee39", "rts24")
SHARES = (0.2, 0.4)
DISPATCH = 0.8
GRID = np.round(np.arange(0.05, 0.501, 0.025), 4)
CLOSED = 1e-9


def residual(name: str, share: float, fraction: float, settings) -> float:
    base = build_case(name, converter_share=share,
                      demand_peak_mw=common_demand_peak(name))
    case = dataclasses.replace(
        base, conv_injection_pu=np.full(len(base.demand_mw), DISPATCH),
        reactive_frac=float(fraction))
    outcome = MultiPeriod(case, settings, "current").solve()
    if not outcome["feasible"]:
        return float("inf")
    networks: list = []
    try:
        ac_voltage_pass(case, outcome, solved=networks)
    except Exception:                                    # noqa: BLE001
        # ac_voltage_pass raises rather than returning stale voltages, which
        # is right for the fixed point: a diverged power flow reported as a
        # zero move is the one thing that must never look like convergence.
        # For a scan it is a data point, not a stop.
        return float("nan")
    worst = 0.0
    for net in networks:
        _, left = voltage_control.correct(copy.deepcopy(net))
        worst = max(worst, left)
    return worst


def main() -> None:
    settings = Capability(i_short_term=1.5, priority="reactive",
                          soc_threshold=0.20)
    rows = []
    for name, share in itertools.product(CASES, SHARES):
        for fraction in GRID:
            left = residual(name, share, fraction, settings)
            import math  # noqa: PLC0415
            rows.append(dict(case=name, share=share,
                             fraction=float(fraction), residual=left,
                             diverged=bool(math.isnan(left)),
                             closes=bool(left <= CLOSED)))
        part = [r for r in rows if r["case"] == name
                and np.isclose(r["share"], share)]
        closing = [r["fraction"] for r in part if r["closes"]]
        print(f"{name} {share}: 닫히는 값 {len(closing)}/{len(part)}개"
              + (f"  {min(closing):.3f} ~ {max(closing):.3f}" if closing
                 else "  없음"))
        if closing:
            gaps = [f for f in part
                    if not f["closes"] and min(closing) < f["fraction"]
                    < max(closing)]
            if gaps:
                print("    구간 안에 닫히지 않는 값: "
                      + ", ".join(f"{g['fraction']:.3f}" for g in gaps)
                      + "  -- 단조가 아닙니다")

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "reactive_scan.csv", index=False)
    monotone = True
    for name, share in itertools.product(CASES, SHARES):
        part = frame[(frame.case == name) & np.isclose(frame.share, share)]
        part = part.sort_values("fraction")
        closes = part.closes.to_numpy()
        if closes.any() and (~closes[:np.max(np.flatnonzero(closes)) + 1]).any():
            monotone = False
    print(f"\n닫힘이 요구 크기에 단조인가: {'예' if monotone else '아니오'}")
    print(f"\nwrote {RESULTS / 'reactive_scan.csv'}")


if __name__ == "__main__":
    sys.exit(main())
