#!/usr/bin/env python3
"""Put the largest outages inside the clearing, and see what that costs.

study/post_contingency.py screens cleared schedules after the fact. The review
asks for more: that the response bought for the most consequential generator
outages be *required* to reach the network, as constraints, and that the paper
say whether its results survive that.

For each secured outage and each period the clearing now carries a deployment
of the reserve and inertial power it bought -- nobody deploying more than they
sold, the tripped unit deploying nothing, the total replacing the unit's
scheduled output exactly -- and the shift-factor flows under that deployment
are held inside the same branch ratings as the pre-contingency dispatch. The
deployment is a decision, so what is required is that a feasible one exists.
Everything is linear, the rows sit after the requirement rows, and the prices
are read exactly as before.

Secured set: the synchronous units with the largest active limit, which are
the ones the inertia requirement is sized by and whose loss moves the most
power across the network. Eight per system, or the whole synchronous fleet if
it is smaller. The rest are covered by the screen, which is rerun on both
schedules so the effect of securing can be read directly.

Writes results/secured_outages.csv.
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

import prices  # noqa: E402
from multiperiod import (MultiPeriod, build_case,  # noqa: E402
                         common_demand_peak)
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
CASES = ("ieee39", "rts24")
SHARES = (0.2, 0.4)
FORMS = ("box", "current")
DISPATCH = 0.8
SECURED = 8
TOLERANCE = 1e-6


def fractions(case, model, outcome, order) -> np.ndarray:
    """min(1, lost / held) for each period and secured outage."""
    out = np.ones((len(case.demand_mw), len(order)))
    sync_held = outcome["x"][model.index["rg"]]
    for t in range(out.shape[0]):
        conv_total = float((outcome["r"][t] + outcome["h"][t]).sum())
        for k, w in enumerate(order):
            held = conv_total + float(sync_held[t].sum() - sync_held[t, w])
            lost = float(outcome["pg"][t, w])
            out[t, k] = min(1.0, lost / held) if held > 0 else 1.0
    return out


def screen(case, model, outcome, only=None) -> dict:
    """The response model of the secured rows, applied after the fact.

    Holdings deployed pro rata and no further than the loss; what that leaves
    uncovered picked up by the surviving synchronous machines in proportion to
    their stored energy. The same model as the constraints, so a schedule that
    satisfies them passes this by construction and the two can be compared.
    """
    ptdf, limit = case.network.ptdf, case.network.limit_mw
    stored = np.asarray(case.sync_h_s) * np.asarray(case.sync_rating_mva)
    over = checked = short = 0
    worst = 0.0
    for t in range(len(case.demand_mw)):
        pre = np.zeros(ptdf.shape[1])
        net_conv = (outcome["pd"][t] - outcome["pc"][t] - outcome["curt"][t]
                    + model.injection[t])
        np.add.at(pre, case.conv_bus, net_conv)
        np.add.at(pre, case.sync_bus, outcome["pg"][t])
        pre -= case.nodal_p_mw[t]
        conv_held = outcome["r"][t] + outcome["h"][t]
        sync_held = outcome["x"][model.index["rg"]][t]
        for w in range(len(case.sync_rating_mva)):
            if only is not None and w not in only:
                continue
            lost = float(outcome["pg"][t, w])
            if lost <= 0.0:
                continue
            avail = np.concatenate([conv_held, sync_held]).copy()
            avail[len(conv_held) + w] = 0.0
            total = float(avail.sum())
            deployed = avail * min(1.0, lost / total) if total > 0                 else 0.0 * avail
            residual = lost - float(deployed.sum())
            short += int(residual > TOLERANCE)
            share = stored.copy()
            share[w] = 0.0
            share = share / share.sum()
            after = pre.copy()
            after[case.sync_bus[w]] -= lost
            np.add.at(after, case.conv_bus, deployed[:len(conv_held)])
            np.add.at(after, case.sync_bus,
                      deployed[len(conv_held):] + residual * share)
            loading = float(np.max(np.abs(ptdf @ after)
                                   / (case.secure_rating_factor * limit))) * 100.0
            checked += 1
            worst = max(worst, loading)
            over += int(loading > 100.0 + 1e-6)
    return dict(checked=checked, over=over, short=short, worst=worst)


def solve_secured(case, settings, form, order, rounds: int = 40, start=None):
    """Iterate the deployed fraction until it stops moving."""
    fraction = (np.ones((len(case.demand_mw), len(order)))
                if start is None else start)
    model = outcome = None
    for index in range(rounds):
        secured = dataclasses.replace(case, secure_outages=list(order),
                                      secure_fraction=fraction)
        model = MultiPeriod(secured, settings, form)
        outcome = model.solve()
        if not outcome["feasible"]:
            return model, outcome, index + 1, float("nan")
        fresh = fractions(case, model, outcome, order)
        move = float(np.abs(fresh - fraction).max())
        # Damped. The holdings sit on a wide optimal face, so the fraction
        # they imply can jump between equally cheap schedules from one round
        # to the next; averaging is what lets it settle.
        fraction = 0.5 * (fraction + fresh)
        if move < 1e-3:
            break
    return model, outcome, index + 1, move


def main() -> None:
    settings = Capability(i_short_term=1.5, priority="reactive",
                          soc_threshold=0.20)
    rows = []
    for name, share, form in itertools.product(CASES, SHARES, FORMS):
        base = build_case(name, converter_share=share,
                          demand_peak_mw=common_demand_peak(name))
        case = dataclasses.replace(
            base, conv_injection_pu=np.full(len(base.demand_mw), DISPATCH))
        order = np.argsort(-case.sync_pmax_mw)[:SECURED]
        order = [int(w) for w in order]
        plain_model = MultiPeriod(case, settings, form)
        plain = plain_model.solve()
        start = (fractions(case, plain_model, plain, order)
                 if plain["feasible"] else None)
        tight_model, tight, rounds, move = solve_secured(case, settings, form,
                                                         order, start=start)
        row = dict(case=name, share=share, form=form,
                   secured_units=len(order),
                   feasible_plain=plain["feasible"],
                   feasible_secured=tight["feasible"],
                   fraction_rounds=rounds, fraction_move=move)
        if plain["feasible"]:
            before = screen(case, plain_model, plain, only=set(order))
            row.update(cost_plain=plain["cost"],
                       price_inertia_plain=prices.price(plain["prices"],
                                                        "inertia"),
                       price_reserve_plain=prices.price(plain["prices"],
                                                        "reserve"),
                       screen_over_plain=before["over"],
                       screen_checked=before["checked"],
                       screen_worst_plain=before["worst"],
                       screen_short_plain=before["short"])
        if tight["feasible"]:
            after = screen(case, tight_model, tight, only=set(order))
            row.update(screen_over_secured=after["over"],
                       screen_worst_secured=after["worst"],
                       cost_secured=tight["cost"],
                       price_inertia_secured=prices.price(tight["prices"],
                                                          "inertia"),
                       price_reserve_secured=prices.price(tight["prices"],
                                                          "reserve"))
        if plain["feasible"] and tight["feasible"]:
            row["cost_increase_pct"] = 100.0 * (tight["cost"] - plain["cost"]) \
                / plain["cost"]
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in row.items()}, flush=True)

    frame = pd.DataFrame(rows)
    # the comparison the paper reports, with and without securing
    for label in ("plain", "secured"):
        column = f"cost_{label}"
        if column in frame:
            wide = frame.pivot_table(index=["case", "share"], columns="form",
                                     values=column)
            if {"box", "current"} <= set(wide.columns):
                delta = 100.0 * (wide["current"] - wide["box"]) / wide["box"]
                for (name, share), value in delta.items():
                    frame.loc[(frame.case == name) & (frame.share == share),
                              f"delta_{label}_pct"] = value
    frame.to_csv(RESULTS / "secured_outages.csv", index=False)
    print(f"\nwrote {RESULTS / 'secured_outages.csv'}")


if __name__ == "__main__":
    sys.exit(main())
