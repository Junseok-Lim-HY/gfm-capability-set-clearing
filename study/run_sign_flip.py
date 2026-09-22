#!/usr/bin/env python3
"""The three omissions, isolated one at a time.

The box bound differs from the current-based capability set in three ways, and
the point of the paper is that they do not share a sign. Sweeping them jointly
mixes the signs and shows only a net figure that depends on the case. Sweeping
them one at a time, with the other two switched off, shows what each one does.

  overload   I_s above I_c, voltage at 1.0, no reactive requirement
  voltage    voltage away from 1.0, I_s = I_c, no reactive requirement
  reactive   reactive duty raised, voltage at 1.0, I_s = I_c

Reported as the difference in procurement cost,

    delta = 100 (C_box - C_set) / C_box,

positive where the active-power bound is the more expensive of the two, which
is where it is conservative about capability the converters do have.
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "model"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from multiperiod import (MultiPeriod, build_case,  # noqa: E402
                         common_demand_peak)
from procurement_lp import Capability  # noqa: E402
from prices import price  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

SIDES = 192          # the polygon gap is then 0.013 per cent and not the story
DISPATCH = 0.8
CASES = ("ieee39", "rts24")
SHARES = (0.2, 0.4)


def settings(i_short: float) -> Capability:
    """The other omissions really switched off, which they were not before.

    The gate was left at 0.20, so the state-of-charge term was active in every
    panel of a figure whose caption says the other two are off. And the
    priority was active, under which the short-term reactive term is free to
    fall to zero, so raising the reactive requirement did not raise the
    reactive current the capability set has to carry: the panel meant to
    isolate reactive current was not isolating it. Reactive priority ties
    Qtilde to Q, which is the case the panel is about.
    """
    return Capability(i_continuous=1.0, i_short_term=i_short,
                      priority="reactive", soc_threshold=0.0,
                      polygon_sides=SIDES)


def evaluate(base, voltage: float, i_short: float, reactive: float) -> dict:
    case = dataclasses.replace(
        base,
        conv_injection_pu=np.full(len(base.demand_mw), DISPATCH),
        conv_voltage_pu=np.full_like(base.conv_voltage_pu, voltage),
        reactive_frac=reactive,
    )
    out = {form: MultiPeriod(case, settings(i_short), form).solve()
           for form in ("box", "current")}
    if not all(out[form]["feasible"] for form in out):
        return {}
    inertia = {f: price(out[f]["prices"], "inertia") for f in out}
    return dict(cost_box=out["box"]["cost"], cost_circle=out["current"]["cost"],
                price_box=inertia["box"], price_circle=inertia["current"],
                error_pct=100.0 * (out["box"]["cost"] - out["current"]["cost"])
                / out["box"]["cost"])


def sweep(name: str, share: float) -> pd.DataFrame:
    base = build_case(name, converter_share=share,
                      demand_peak_mw=common_demand_peak(name, SHARES))
    rows = []
    for value in (1.0, 1.1, 1.2, 1.35, 1.5):
        got = evaluate(base, 1.0, value, 0.0)
        rows.append(dict(omission="overload", value=value, **got))
    for value in (0.95, 0.98, 1.00, 1.02, 1.05):
        got = evaluate(base, value, 1.0, 0.0)
        rows.append(dict(omission="voltage", value=value, **got))
    for value in (0.0, 0.35, 0.80, 1.20):
        got = evaluate(base, 1.0, 1.0, value)
        rows.append(dict(omission="reactive", value=value, **got))
    frame = pd.DataFrame(rows)
    frame.insert(0, "case", name)
    frame.insert(1, "share", share)
    return frame


def main() -> None:
    frames = [sweep(name, share) for name in CASES for share in SHARES]
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(RESULTS / "sign_flip.csv", index=False)

    for omission, label in (
        ("overload", "1. overload alone: the bound cannot see I_s, so it is"
                     " conservative and only ever conservative"),
        ("voltage", "2. voltage alone: the sign flips at 1.0 pu, cleanly"),
        ("reactive", "3. reactive alone: the bound ignores Q, so it is"
                     " optimistic and only ever optimistic"),
    ):
        print(f"\n{label}\n")
        view = frame[frame.omission == omission].dropna(subset=["error_pct"])
        print(view.pivot_table(index="value", columns=["case", "share"],
                               values="error_pct")
              .to_string(float_format=lambda v: f"{v:8.2f}"))

    print("\ndifference in procurement cost, per cent of the box-based cost;"
          "\npositive means the active-power bound is the more expensive of"
          "\nthe two, which is where it is conservative.")
    print(f"\nwrote {RESULTS / 'sign_flip.csv'} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
