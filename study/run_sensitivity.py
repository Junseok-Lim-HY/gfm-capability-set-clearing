#!/usr/bin/env python3
"""How far the headline result moves when the fixed inputs are varied.

Every number in the case study rests on inputs that were held rather than
swept: the RoCoF limit, the contingency, the synchronous inertia constant, the
storage duration and the polygon resolution. This asks what each of them is
worth, so the paper can say which conclusions are robust and which are
conditional.
"""
from __future__ import annotations

import dataclasses
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "model"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from multiperiod import (MultiPeriod, build_case,  # noqa: E402
                         common_demand_peak, store_energy)
from procurement_lp import Capability  # noqa: E402
from prices import price  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
BASE = dict(dispatch=0.8, i_short=1.5, threshold=0.20, priority="reactive")


def clear(case, i_short=1.5, priority="reactive", threshold=0.20,
          sides=24, box_bound="rating", form="current"):
    settings = Capability(i_short_term=i_short, priority=priority,
                          soc_threshold=threshold, polygon_sides=sides,
                          box_bound=box_bound)
    return MultiPeriod(case, settings, form).solve()


def gap(case, **kwargs) -> dict:
    """Cost and inertia price under both representations, and the gap."""
    out = {f: clear(case, form=f, **kwargs) for f in ("box", "current")}
    if not all(out[f]["feasible"] for f in out):
        return {}
    inertia = {f: price(out[f]["prices"], "inertia") for f in out}
    return dict(cost_box=out["box"]["cost"], cost_set=out["current"]["cost"],
                price_box=inertia["box"], price_set=inertia["current"],
                gap_pct=100.0 * (out["box"]["cost"] - out["current"]["cost"])
                / out["box"]["cost"])


def main() -> None:
    rows = []
    shares = (0.2, 0.4)
    peaks = {n: common_demand_peak(n, shares) for n in ("ieee39", "rts24")}
    for name, share in itertools.product(("ieee39", "rts24"), shares):
        base = build_case(name, converter_share=share,
                          demand_peak_mw=peaks[name])
        loaded = dataclasses.replace(
            base, conv_injection_pu=np.full(len(base.demand_mw),
                                            BASE["dispatch"]))
        reference = gap(loaded)
        if not reference:
            continue

        def record(axis, value, case, **kwargs):
            got = gap(case, **kwargs)
            if got:
                rows.append(dict(case=name, share=share, axis=axis,
                                 value=value, **got,
                                 shift_pct=got["gap_pct"]
                                 - reference["gap_pct"]))

        record("reference", 0.0, loaded)
        for rocof in (0.5, 1.0, 1.5, 2.0):
            record("rocof", rocof,
                   dataclasses.replace(loaded, rocof_hz_s=rocof))
        for inertia in (3.0, 4.0, 6.0):
            record("sync_inertia", inertia, dataclasses.replace(
                loaded, sync_h_s=np.full(len(loaded.sync_h_s), inertia)))
        # Through store_energy, so the four-hour row of this sweep is the
        # same machine as the four-hour base case. Scaling the apparent
        # rating here instead made them differ by the power factor, and the
        # bold reference row of the table then disagreed with every other
        # bold reference row in it.
        for hours in (1.0, 2.0, 4.0, 8.0):
            record("storage_hours", hours, dataclasses.replace(
                loaded, conv_energy_mwh=store_energy(loaded.conv_pmax_mw,
                                                     hours)))
        # The apparent rating is derived, not given: the cases carry a
        # rectangular P-Q box and S is taken as its corner. A reader is
        # entitled to ask what the answer would be under an assumed power
        # factor instead, which is the other common convention.
        for pf in (0.85, 0.90, 0.95, 1.00):
            record("rating_pf", pf, dataclasses.replace(
                loaded, conv_rating_mva=loaded.conv_pmax_mw / pf))
        # And what the comparison looks like if the active-power bound is
        # drawn at the active limit rather than at the apparent rating.
        record("box_at_pmax", 1.0, loaded, box_bound="pmax")
        # Charge and discharge are separate non-negative variables and no
        # unit ever does both at once, but across the fleet some charge while
        # others discharge in the same period, and nothing but the round-trip
        # loss discourages it. A throughput cost is the usual remedy, so the
        # question is what it would do to the answer.
        for cycling in (0.0, 1.0, 2.0, 5.0):
            record("cycling_cost", cycling,
                   dataclasses.replace(loaded, conv_cost_cycling=cycling))
        # How much of the containment reserve has activated by the time the
        # inertial response peaks. The base case charges all of it against the
        # two-second envelope, which is the worst case and not a fact.
        for overlap in (0.0, 0.25, 0.5, 1.0):
            record("reserve_overlap", overlap,
                   dataclasses.replace(loaded, reserve_overlap=overlap))
        # Out to 384, because the pointwise shortfall of the 24-gon is the
        # same order as delta itself and a reader is entitled to ask whether
        # the result is the approximation. It is not, and this is where that
        # is shown rather than argued.
        for sides in (8, 12, 24, 48, 96, 192, 384):
            record("polygon", sides, loaded, sides=sides)
        for dv in (-0.05, -0.02, 0.0, 0.02, 0.05):
            record("voltage_shift", dv, dataclasses.replace(
                loaded, conv_voltage_pu=loaded.conv_voltage_pu * (1.0 + dv)))
        # What cancels between the two formulations is a scaling common to
        # both objectives, not the offers themselves: the two substitute
        # between energy and the two capacity products in different
        # proportions, so the ratio of the capacity offers to the energy
        # offers moves the reported difference. This axis is what entitles the
        # paper to say that, instead of asserting that the constants cancel.
        for ratio in (0.25, 0.5, 1.0, 2.0, 4.0):
            record("offer_ratio", ratio, dataclasses.replace(
                loaded,
                conv_cost_reserve=loaded.conv_cost_reserve * ratio,
                conv_cost_inertia=loaded.conv_cost_inertia * ratio,
                sync_cost_reserve=loaded.sync_cost_reserve * ratio))

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "sensitivity.csv", index=False)

    for axis, label in (("rocof", "RoCoF limit (Hz/s)"),
                        ("sync_inertia", "synchronous inertia constant (s)"),
                        ("storage_hours", "storage duration (h)"),
                        ("polygon", "polygon faces"),
                        ("voltage_shift", "voltage shift (pu)"),
                        ("offer_ratio", "capacity/energy offer ratio"),
                        ("rating_pf", "assumed power factor for S"),
                        ("box_at_pmax", "active bound drawn at Pmax"),
                        ("cycling_cost", "storage throughput cost ($/MWh)"),
                        ("reserve_overlap", "reserve activated at the inertia peak")):
        view = frame[frame.axis == axis]
        if view.empty:
            continue
        print(f"\n{label}: cost gap between the two representations, %\n")
        print(view.pivot_table(index="value", columns=["case", "share"],
                               values="gap_pct")
              .to_string(float_format=lambda v: f"{v:8.2f}"))

    print(f"\nwrote {RESULTS / 'sensitivity.csv'} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
