#!/usr/bin/env python3
"""What changes if a converter is allowed to absorb reactive power.

The Proposition of Section 3 is written on |Qtilde| and holds for either sign.
The clearing has held Q >= 0. That is a narrower domain than the analysis it
illustrates, and it is the kind of gap a reader finds by putting the two
side by side. It also biases the ac screen: the cases show overvoltage, a real
grid-forming converter would absorb to hold a bus down, and the schedule was
not allowed to.

So the restriction is lifted and the difference measured. Absorption is a
separate non-negative column priced like injection, so the clearing pays for
reactive power in either direction and cannot earn by absorbing.

Three things are asked. Does delta move -- the paper's central quantity. Does
the clearing actually use absorption when offered it. And does the ac screen
improve, which is the reason for expecting the restriction to have mattered.

Writes results/reactive_absorption.csv.
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

from multiperiod import (MultiPeriod, ac_feasibility,  # noqa: E402
                         build_case, common_demand_peak)
from prices import price  # noqa: E402
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
CASES = ("ieee39", "rts24")
SHARES = (0.2, 0.4)
DISPATCH = 0.8


def clear(name, share, both, settings):
    base = build_case(name, converter_share=share,
                      demand_peak_mw=common_demand_peak(name))
    case = dataclasses.replace(
        base, conv_injection_pu=np.full(len(base.demand_mw), DISPATCH),
        bidirectional_reactive=both)
    out = {form: MultiPeriod(case, settings, form).solve()
           for form in ("box", "current")}
    return case, out


def main() -> None:
    settings = Capability(i_short_term=1.5, priority="reactive",
                          soc_threshold=0.20)
    rows = []
    for name, share in itertools.product(CASES, SHARES):
        entry = dict(case=name, share=share)
        for both in (False, True):
            case, out = clear(name, share, both, settings)
            tag = "both" if both else "export"
            if not all(o["feasible"] for o in out.values()):
                entry[f"{tag}_feasible"] = False
                continue
            delta = (100.0 * (out["box"]["cost"] - out["current"]["cost"])
                     / out["box"]["cost"])
            screen = ac_feasibility(case, out["current"])
            entry.update({
                f"{tag}_feasible": True,
                f"{tag}_cost": out["current"]["cost"],
                f"{tag}_delta": delta,
                f"{tag}_price": price(out["current"]["prices"], "inertia"),
                f"{tag}_price_box": price(out["box"]["prices"], "inertia"),
                f"{tag}_absorb_mvar": float(out["current"]["q_absorb"].sum()),
                f"{tag}_v_min": screen["v_min"],
                f"{tag}_v_max": screen["v_max"],
                f"{tag}_loading": screen["loading_max"],
            })
        rows.append(entry)
        if entry.get("both_feasible") and entry.get("export_feasible"):
            print(f"{name} {share:.0%}: "
                  f"delta {entry['export_delta']:+.4f} -> "
                  f"{entry['both_delta']:+.4f}%  "
                  f"흡수 {entry['both_absorb_mvar']:.1f} MVArh  "
                  f"V {entry['export_v_min']:.4f}~{entry['export_v_max']:.4f}"
                  f" -> {entry['both_v_min']:.4f}~{entry['both_v_max']:.4f}  "
                  f"선로 {entry['export_loading']:.1f} -> "
                  f"{entry['both_loading']:.1f}%")
        else:
            print(f"{name} {share:.0%}: 한쪽이 풀리지 않았습니다")

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "reactive_absorption.csv", index=False)
    print(f"\nwrote {RESULTS / 'reactive_absorption.csv'}")


if __name__ == "__main__":
    sys.exit(main())
