#!/usr/bin/env python3
"""Can the reserve and inertia the clearing sold actually reach the network?

Eq. (20) constrains the pre-contingency dispatch and nothing else. The products
this paper is about are reserve and inertial power, the requirements are
written one row per credible outage, and the schedule is never asked whether
the response it holds can be delivered through the network once the outage has
happened. That is the first question a reader of a reserve-and-inertia paper
asks, and the paper has had no answer.

This is a screen and not a constraint. The cleared schedules are taken as they
are and put through the outages the requirements were written for, and the
resulting flows are compared with the same branch ratings the clearing used.
It answers "does the delivered response overload anything" without changing
what the clearing solves, which is the smallest honest step: putting it inside
the programme is a different formulation and belongs with the ac work the
paper already says it does not do.

What the response is taken to be. The tripped unit's *scheduled output* stops
flowing -- not its rating, which is the convention the requirement rows use to
size how much inertia to buy, but not what physically disappears from a branch.
The fleet then covers that imbalance in proportion to what each member is
holding: reserve and inertial power for converters, reserve for synchronous
units, each capped at its own holding. A responder cannot supply more than it
sold, and the fleet does not supply more than was lost.

Both representations are screened, since a difference that appeared only in the
capability set's schedules would be a different finding from one that appears
in both.

Writes results/post_contingency.csv.
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

import capability as cap  # noqa: E402
from multiperiod import (MultiPeriod, build_case,  # noqa: E402
                         common_demand_peak)
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
CASES = ("ieee39", "rts24")
SHARES = (0.2, 0.4)
FORMS = ("box", "current")
DISPATCH = 0.8
TOLERANCE = 1e-6          # MW, below which an excess is arithmetic


def nodal(case, outcome, model, period):
    """Net injection per bus column, pre-contingency, in MW."""
    vector = np.zeros(case.network.ptdf.shape[1])
    net_conv = (outcome["pd"][period] - outcome["pc"][period]
                - outcome["curt"][period] + model.injection[period])
    np.add.at(vector, case.conv_bus, net_conv)
    np.add.at(vector, case.sync_bus, outcome["pg"][period])
    return vector - case.nodal_p_mw[period]


def response(held, lost):
    """Share the lost infeed among holders, pro rata, capped at each holding.

    Pro rata rather than everything at once. The requirement rows size the
    fleet's holding against the outage taken at rating, so the holding is
    generally larger than the infeed that actually disappears; deploying all of
    it would put more power on the network after the outage than was lost
    before it, which is not a screen of anything.
    """
    total = float(held.sum())
    if total <= 0.0 or lost <= 0.0:
        return np.zeros_like(held)
    return held * min(1.0, lost / total)


def screen(name, share, form, settings):
    base = build_case(name, converter_share=share,
                      demand_peak_mw=common_demand_peak(name))
    case = dataclasses.replace(
        base, conv_injection_pu=np.full(len(base.demand_mw), DISPATCH))
    model = MultiPeriod(case, settings, form)
    outcome = model.solve()
    if not outcome["feasible"]:
        return dict(case=name, share=share, form=form, feasible=False)

    ptdf, limit = case.network.ptdf, case.network.limit_mw
    periods = len(case.demand_mw)
    worst, worst_where, over, checked = 0.0, "", 0, 0
    worst_base = 0.0
    # Infeed the fleet's holding could not replace. A shift-factor flow is
    # computed about a reference bus, so an imbalance left in the injection
    # vector is silently made up there: the flows then include a response
    # nobody sold. It is recorded, and the outage is counted as unscreened
    # rather than as passed.
    unmet_worst, unmet_count = 0.0, 0

    for period in range(periods):
        pre = nodal(case, outcome, model, period)
        base_flow = ptdf @ pre
        worst_base = max(worst_base,
                         float(np.max(np.abs(base_flow) / limit)) * 100.0)

        # Converters hold reserve and inertial power; synchronous units hold
        # reserve. Both are what the schedule says they hold.
        conv_held = outcome["r"][period] + outcome["h"][period]
        # Synchronous reserve is not surfaced in the outcome dictionary,
        # but the full primal is, and the model knows its columns.
        sync_held = outcome["x"][model.index["rg"]][period]

        outages = [("sync", w, float(outcome["pg"][period, w]))
                   for w in range(len(case.sync_rating_mva))]
        net_conv = (outcome["pd"][period] - outcome["pc"][period]
                    - outcome["curt"][period] + model.injection[period])
        outages += [("conv", v, float(net_conv[v]))
                    for v in range(len(case.conv_rating_mva))]

        for kind, index, lost in outages:
            if lost <= 0.0:
                continue          # nothing leaves, nothing to screen
            after = pre.copy()
            conv_avail = conv_held.copy()
            sync_avail = sync_held.copy()
            if kind == "sync":
                after[case.sync_bus[index]] -= lost
                sync_avail[index] = 0.0        # it tripped; it responds to nothing
            else:
                after[case.conv_bus[index]] -= lost
                conv_avail[index] = 0.0
            held = np.concatenate([conv_avail, sync_avail])
            share_of = response(held, lost)
            unmet = lost - float(share_of.sum())
            if unmet > TOLERANCE:
                unmet_count += 1
                unmet_worst = max(unmet_worst, unmet)
            np.add.at(after, case.conv_bus, share_of[:len(conv_avail)])
            np.add.at(after, case.sync_bus, share_of[len(conv_avail):])

            loading = np.abs(ptdf @ after) / limit * 100.0
            checked += 1
            top = float(loading.max())
            if top > worst:
                worst, worst_where = top, (
                    f"t{period} {kind}{index} "
                    f"{case.network.branch_name[int(np.argmax(loading))]}")
            if top > 100.0 + TOLERANCE:
                over += 1

    return dict(case=name, share=share, form=form, feasible=True,
                outages_checked=checked,
                loading_pre_max=worst_base,
                loading_post_max=worst,
                worst_at=worst_where,
                over_rating=over,
                over_fraction=over / checked if checked else np.nan,
                unmet_outages=unmet_count,
                unmet_worst_mw=unmet_worst)


def main() -> None:
    settings = Capability(i_short_term=1.5, priority="reactive",
                          soc_threshold=0.20)
    rows = []
    for name, share, form in itertools.product(CASES, SHARES, FORMS):
        row = screen(name, share, form, settings)
        rows.append(row)
        if row.get("feasible"):
            print(f"{name} {share:.0%} {form:8s}: "
                  f"사고 {row['outages_checked']:4d}개  "
                  f"사고 전 최악 {row['loading_pre_max']:6.1f}%  "
                  f"사고 후 최악 {row['loading_post_max']:6.1f}%  "
                  f"정격 초과 {row['over_rating']:4d}"
                  f" ({row['over_fraction']:.1%})  "
                  f"미충당 {row['unmet_outages']:3d}건"
                  f"(최대 {row['unmet_worst_mw']:.1f} MW)  "
                  f"[{row['worst_at']}]")
        else:
            print(f"{name} {share:.0%} {form}: 해 없음")

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "post_contingency.csv", index=False)
    print(f"\nwrote {RESULTS / 'post_contingency.csv'}")


if __name__ == "__main__":
    sys.exit(main())
