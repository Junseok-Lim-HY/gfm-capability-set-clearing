#!/usr/bin/env python3
"""How much reactive duty the network will carry at its declared voltages.

The iterative correction of study/run_ac_clearing.py closes the voltage band
in every period on IEEE 39-bus and in five of eight on RTS-24. The three that
do not have a single bus over the ceiling, always the same one, and it is a
converter bus. Setting the zonal reactive requirement to zero closes all
eight. So the residual is not the network refusing the schedule; it is the
requirement and the voltage limits asking for incompatible things, and no
setting of generator setpoints or transformer taps undoes it because the
injection causing it is at that bus.

That makes the useful question quantitative rather than yes-or-no: how large
can the reactive requirement be before the voltage band stops closing? This
bisects for it. The answer bounds a modelling choice the paper made -- the
requirement is a fraction of zonal reactive load, and that fraction was
picked, not derived -- and a fraction the network cannot serve is not a
conservative assumption but an infeasible one.

Writes results/reactive_headroom.csv.
"""
from __future__ import annotations

import copy
import dataclasses
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
DISPATCH = 0.8
STEPS = 5
BASE_FRACTION = 0.35


def residual(name: str, share: float, fraction: float, settings) -> float:
    """Worst voltage violation left after correcting the controls."""
    base = build_case(name, converter_share=share,
                      demand_peak_mw=common_demand_peak(name))
    case = dataclasses.replace(
        base, conv_injection_pu=np.full(len(base.demand_mw), DISPATCH),
        reactive_frac=fraction)
    outcome = MultiPeriod(case, settings, "current").solve()
    if not outcome["feasible"]:
        return float("inf")
    networks: list = []
    ac_voltage_pass(case, outcome, solved=networks)
    worst = 0.0
    for net in networks:
        _, left = voltage_control.correct(copy.deepcopy(net))
        worst = max(worst, left)
    return worst


def main() -> None:
    settings = Capability(i_short_term=1.5, priority="reactive",
                          soc_threshold=0.20)
    rows = []
    for name in ("rts24", "ieee39"):
        for share in (0.2, 0.4):
            at_base = residual(name, share, BASE_FRACTION, settings)
            if at_base <= 1e-9:
                rows.append(dict(case=name, share=share,
                                 closes_at_base=True,
                                 largest_that_closes=BASE_FRACTION,
                                 smallest_that_fails=np.nan,
                                 residual_at_base=at_base))
                print(f"{name} {share}: 기준 {BASE_FRACTION} 에서 닫힙니다")
                continue
            low, high = 0.0, BASE_FRACTION
            for _ in range(STEPS):
                mid = 0.5 * (low + high)
                if residual(name, share, mid, settings) <= 1e-9:
                    low = mid
                else:
                    high = mid
            rows.append(dict(case=name, share=share, closes_at_base=False,
                             largest_that_closes=low,
                             smallest_that_fails=high,
                             residual_at_base=at_base))
            print(f"{name} {share}: 닫히는 최대 {low:.4f}, "
                  f"닫히지 않는 최소 {high:.4f} "
                  f"(기준 {BASE_FRACTION} 에서 잔여 {at_base:.2e})")

    pd.DataFrame(rows).to_csv(RESULTS / "reactive_headroom.csv", index=False)
    print(f"\nwrote {RESULTS / 'reactive_headroom.csv'}")


if __name__ == "__main__":
    sys.exit(main())
