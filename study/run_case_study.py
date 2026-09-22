#!/usr/bin/env python3
"""Case study: two systems, two capability models, network and voltage checks.

Six sections. The first two are the result; the third is its other sign; the
fourth is a mechanism too weak to be a claim; the last two are the answers to
the two questions a referee will ask about the network and about voltage.
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

from multiperiod import (Case, HOURS, MultiPeriod, build_case,  # noqa: E402
                         common_demand_peak, optimal_face_range,
                         solve_to_voltage_fixed_point)
from procurement_lp import Capability  # noqa: E402
from prices import price  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

CASES = ("ieee39", "rts24")
FORMS = ("box", "current")
SHARES = (0.2, 0.4)
DISPATCH = (0.4, 0.6, 0.8, 1.0)

BASE = dict(dispatch=0.8, i_short=1.5, reactive=0.35,
            threshold=0.20, priority="reactive", share=0.4)


def mean_price(prices: dict, kind: str) -> float:
    """Per-period sum of the requirement's duals, averaged over the day.

    The inertia requirement is one row per credible outage, so averaging over
    rows would divide the price by the number of machines. See prices.py.
    """
    return price(prices, kind)


def variant(base: Case, dispatch: float, reactive: float,
            reserve: float | None, network: bool, zonal: bool) -> Case:
    fields = dict(conv_injection_pu=np.full(len(base.demand_mw), dispatch),
                  reactive_frac=reactive, zonal_reactive=zonal)
    if reserve is not None:
        fields["reserve_frac"] = reserve
    if not network:
        fields["network"] = None
    return dataclasses.replace(base, **fields)


def run(base: Case, form: str, dispatch: float, i_short: float,
        reactive: float, threshold: float, priority: str, share: float,
        reserve: float | None = None, network: bool = True,
        zonal: bool = True) -> dict:
    case = variant(base, dispatch, reactive, reserve, network, zonal)
    settings = Capability(i_short_term=i_short, priority=priority,
                          soc_threshold=threshold)
    outcome = MultiPeriod(case, settings, form).solve()
    row = dict(case=base.name, share=share, dispatch=dispatch,
               i_short_term=i_short, reactive_frac=reactive,
               reserve_frac=case.reserve_frac, soc_threshold=threshold,
               priority=priority, network=network, zonal=zonal, form=form,
               feasible=outcome["feasible"])
    if outcome["feasible"]:
        row.update(
            cost=outcome["cost"],
            price_inertia=mean_price(outcome["prices"], "inertia"),
            price_reserve=mean_price(outcome["prices"], "reserve"),
            price_energy=mean_price(outcome["prices"], "energy"),
            curtailed_mwh=float(outcome["curt"].sum()) * HOURS,
            mean_availability=float(outcome["availability"].mean()),
            min_soc=float(outcome["soc"].min()),
        )
    return row


def compare(frame: pd.DataFrame, index, value: str, fmt: str,
            kind: str = "ratio") -> None:
    table = frame[frame.feasible].pivot_table(index=index, columns="form",
                                              values=value)
    if {"box", "current"}.issubset(table.columns):
        if kind == "ratio":
            table["ratio"] = table["box"] / table["current"].replace(0.0, np.nan)
        elif kind == "excess":
            table["box excess"] = table["box"] - table["current"]
        else:
            table["error %"] = 100.0 * (table["current"] - table["box"]) \
                / table["current"].replace(0.0, np.nan)
    print(table.to_string(float_format=lambda v: format(v, fmt)))
    for _, r in frame[~frame.feasible].iterrows():
        print(f"    infeasible: {r['case']}, share {r['share']}, "
              f"dispatch {r['dispatch']}, {r['form']}")


def main() -> None:
    # One peak per system, shared by every converter share, so that a
    # comparison across shares is a comparison of shares.
    peaks = {name: common_demand_peak(name, SHARES) for name in CASES}
    bases = {(name, share): build_case(name, converter_share=share,
                                       demand_peak_mw=peaks[name])
             for name in CASES for share in SHARES}
    for (name, share), case in bases.items():
        if share == SHARES[0]:
            print(f"{name}: peak demand {case.demand_mw.max():.0f} MW, "
                  f"{case.network.ptdf.shape[0]} constrained branches, "
                  f"{len(case.network.zones)} zones, "
                  f"tightest branch {case.network.limit_mw.min():.0f} MW")
        print(f"   share {share:.1f}: {len(case.conv_rating_mva)} converters "
              f"({case.conv_rating_mva.sum():.0f} MVA), "
              f"{len(case.sync_rating_mva)} synchronous "
              f"({case.sync_rating_mva.sum():.0f} MVA)")
    print(f"\nheld at {BASE}\n")
    records = []

    # ---------------------------------------------------------------- 1 & 2 --
    grid_rows = []
    for (name, share), dispatch, form in itertools.product(
            bases, DISPATCH, FORMS):
        kwargs = dict(BASE)
        kwargs.update(dispatch=dispatch, share=share)
        grid_rows.append(run(bases[(name, share)], form=form, **kwargs))
    dispatch_frame = pd.DataFrame(grid_rows)
    dispatch_frame["axis"] = "dispatch"
    records.append(dispatch_frame)

    print("1. inertia price against pre-disturbance dispatch\n")
    compare(dispatch_frame, ["case", "share", "dispatch"],
            "price_inertia", "8.2f")

    print("\n2. curtailment the capability model forces, MWh over the day\n")
    compare(dispatch_frame, ["case", "share", "dispatch"],
            "curtailed_mwh", "10.0f", kind="excess")

    print("\n   and procurement cost\n")
    compare(dispatch_frame, ["case", "share", "dispatch"],
            "cost", "10.0f", kind="error")

    # A curtailment figure is only a result if the cost-optimal face is narrow
    # in it. Where the face is wide, the number a run prints is the tie-break's
    # choice of vertex rather than a property of the capability model. What can
    # be claimed is the worst case over every pair of optima: the box's minimum
    # against the capability set's maximum. If that is positive the finding
    # holds whatever vertex either solver picks; if it is not, there is no
    # finding.
    print("\n   and the same at full injection over the cost-optimal face,"
          "\n   which is what says whether those figures are results\n")
    face_rows = []
    for (name, share) in bases:
        loaded = variant(bases[(name, share)], 1.0, BASE["reactive"],
                         None, True, True)
        settings = Capability(i_short_term=BASE["i_short"],
                              priority=BASE["priority"],
                              soc_threshold=BASE["threshold"])
        span = {f: optimal_face_range(loaded, settings, f) for f in FORMS}
        worst = span["box"][0] - span["current"][1]
        face_rows.append(dict(
            case=name, share=share,
            box_lo=span["box"][0], box_hi=span["box"][1],
            set_lo=span["current"][0], set_hi=span["current"][1],
            guaranteed_excess=worst, identified=bool(worst > 0.0)))
    face = pd.DataFrame(face_rows)
    face.to_csv(RESULTS / "curtailment_range.csv", index=False)
    print(face.to_string(index=False, float_format=lambda v: f"{v:9.1f}"))

    # -------------------------------------------------------------------- 3 --
    rows = []
    for (name, share), i_short, form in itertools.product(
            bases, (1.0, 1.2, 1.5), FORMS):
        kwargs = dict(BASE)
        kwargs.update(i_short=i_short, share=share)
        rows.append(run(bases[(name, share)], form=form, **kwargs))
    frame = pd.DataFrame(rows)
    frame["axis"] = "i_short"
    records.append(frame)
    print("\n3. the overload term alone. The box cannot see I_s and so is flat"
          "\n   in it; the circle is not. The two do not coincide at I_s = 1.0,"
          "\n   because voltage and reactive current are still in play - for"
          "\n   the case where they must coincide see test_reduction.py\n")
    compare(frame, ["case", "share", "i_short_term"], "price_inertia", "8.2f")

    # -------------------------------------------------------------------- 4 --
    # Reserve is made scarce at 15 per cent of demand rather than 30. Under
    # the demand sizing of Section 5 a 30 per cent requirement exceeds the
    # reserve the fleets can hold and every cell was infeasible, which says
    # nothing about capability. The reactive duty is swept over the range in
    # which both formulations clear; what happens past it is reported below,
    # and is the sharper result.
    rows = []
    for (name, share), q, form in itertools.product(
            bases, (0.35, 0.50, 0.65), FORMS):
        kwargs = dict(BASE)
        kwargs.update(i_short=1.0, reactive=q, share=share)
        rows.append(run(bases[(name, share)], form=form, reserve=0.15, **kwargs))
    frame = pd.DataFrame(rows)
    frame["axis"] = "reactive_scarce"
    records.append(frame)
    print("\n4. the other sign: at I_s = 1.0 with reserve scarce, reactive duty"
          "\n   makes the box optimistic (positive means the box understates)\n")
    compare(frame, ["case", "share", "reactive_frac"], "cost", "10.0f",
            kind="error")

    # Past the swept range the difference stops being a cost and becomes
    # feasibility, which is the strongest form the discrepancy can take: the
    # active-power bound clears a schedule the capability set cannot deliver.
    edge = []
    for (name, share), q, form in itertools.product(
            bases, (0.80, 1.00, 1.20), FORMS):
        kwargs = dict(BASE)
        kwargs.update(i_short=1.0, reactive=q, share=share)
        got = run(bases[(name, share)], form=form, reserve=0.15, **kwargs)
        edge.append(dict(case=name, share=share, reactive=q, form=form,
                         feasible=got["feasible"]))
    edge = pd.DataFrame(edge)
    split = edge.pivot_table(index=["case", "share", "reactive"],
                             columns="form", values="feasible").astype(bool)
    only_box = split[split["box"] & ~split["current"]]
    print("\n   past that range the box clears where the capability set "
          "cannot:\n")
    print(only_box.to_string() if len(only_box)
          else "   (no such cell)")

    # -------------------------------------------------------------------- 5 --
    rows = []
    for (name, share), threshold in itertools.product(
            bases, (0.0, 0.10, 0.20, 0.30)):
        kwargs = dict(BASE)
        kwargs.update(threshold=threshold, share=share)
        rows.append(run(bases[(name, share)], form="current", **kwargs))
    frame = pd.DataFrame(rows)
    frame["axis"] = "threshold"
    records.append(frame)
    print("\n5. the state-of-charge gate, current-based model only\n")
    print(frame[frame.feasible]
          .pivot_table(index=["case", "share"], columns="soc_threshold",
                       values="mean_availability")
          .to_string(float_format=lambda v: f"{v:8.3f}"))

    # -------------------------------------------------------------------- 6 --
    # Three rungs, not two. Switching the network on used to switch the branch
    # constraints and the zonal reactive requirement on together, so a cost
    # movement could not be attributed to either. Each is now added in turn.
    rows = []
    rungs = (("single bus, global Q", False, False),
             ("dc branches, global Q", True, False),
             ("dc branches, zonal Q", True, True))
    for (name, share), (label, network, zonal), form in itertools.product(
            bases, rungs, FORMS):
        kwargs = dict(BASE)
        kwargs["share"] = share
        got = run(bases[(name, share)], form=form, network=network,
                  zonal=zonal, **kwargs)
        got["rung"] = label
        rows.append(got)
    frame = pd.DataFrame(rows)
    frame["axis"] = "network"
    records.append(frame)
    print("\n6. what the network adds, one constraint at a time\n")
    table = (frame[frame.feasible]
             .pivot_table(index=["case", "share", "form"], columns="rung",
                          values="cost", sort=False)
             .reindex(columns=[r[0] for r in rungs]))
    print(table.to_string(float_format=lambda v: f"{v:10.0f}"))

    # -------------------------------------------------------------------- 7 --
    print("\n7. voltage fixed point: clear, re-solve the ac network, re-clear\n")
    print(f'{"case":>7} {"share":>6} {"form":>8} {"conv":>6} {"rounds":>7}'
          f' {"cost first":>11} {"cost last":>10} {"move %":>7}'
          f' {"inertia":>8} {"V range":>13}')
    voltage_rows = []
    for (name, share), form in itertools.product(bases, FORMS):
        case = variant(bases[(name, share)], BASE["dispatch"],
                       BASE["reactive"], None, True, True)
        settings = Capability(i_short_term=BASE["i_short"],
                              priority=BASE["priority"],
                              soc_threshold=BASE["threshold"])
        outcome = solve_to_voltage_fixed_point(case, settings, form)
        if not outcome["feasible"]:
            print(f"{name:>7} {share:6.1f} {form:>8}  infeasible")
            continue
        first, last = outcome["history"][0], outcome["history"][-1]
        move = 100.0 * (last["cost"] - first["cost"]) / first["cost"]
        print(f'{name:>7} {share:6.1f} {form:>8}'
              f' {str(outcome["converged"]):>6} {outcome["rounds"]:7d}'
              f' {first["cost"]:11.0f} {last["cost"]:10.0f} {move:+7.2f}'
              f' {first["price_inertia"]:5.2f}->{last["price_inertia"]:5.2f}'
              f' {last["min_voltage"]:6.3f}-{last["max_voltage"]:.3f}')
        voltage_rows.append(dict(case=name, share=share, form=form,
                                 converged=outcome["converged"],
                                 rounds=outcome["rounds"],
                                 cost_first=first["cost"],
                                 cost_last=last["cost"], cost_move_pct=move,
                                 price_first=first["price_inertia"],
                                 price_last=last["price_inertia"]))
    pd.DataFrame(voltage_rows).to_csv(RESULTS / "voltage_fixed_point.csv",
                                      index=False)

    frame = pd.concat(records, ignore_index=True)
    frame.to_csv(RESULTS / "case_study.csv", index=False)
    print(f"\nwrote {RESULTS / 'case_study.csv'} ({len(frame)} rows, "
          f"{int((~frame.feasible).sum())} infeasible) and "
          f"{RESULTS / 'voltage_fixed_point.csv'}")


if __name__ == "__main__":
    main()
