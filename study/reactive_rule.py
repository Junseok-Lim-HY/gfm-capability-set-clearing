#!/usr/bin/env python3
"""Where the zonal reactive requirement should come from.

The requirement is written as a fraction of each zone's reactive load, and
that fraction is \\num{0.35} throughout the paper. It was chosen. The
iterative correction of Section 7.2 then finds that RTS-24 will not carry
0.35 at its declared voltages, which invites the obvious repair: measure the
largest fraction that does carry, and use that.

That repair is not available. Setting the requirement to the largest value
that produces a feasible answer is choosing the assumption after seeing which
assumption flatters the result, and it would read that way however it were
explained. The order has to be the other way round: state a rule, compute
what the rule gives, and then report whether the network carries it -- with
the third step allowed to come out badly.

So this computes what several rules give, and does not pick one. Each is a
sentence someone could have written before seeing any result:

  capacity share   converters supply the fraction of a zone's reactive load
                   that matches their share of its installed capacity
  power factor     each converter is asked for the reactive power a fixed
                   power factor of its rating implies, as grid codes write it
  capability cap   the requirement never exceeds what the zone's converters
                   can hold inside their own continuous current limit

The first two are market rules and are blind to the network. The third is a
physical ceiling: a requirement above it cannot be met by the fleet at all,
whatever the network then does with the result. None of them is derived from
whether the answer comes out feasible, which is the property that matters.

Writes results/reactive_rule.csv.
"""
from __future__ import annotations

import dataclasses
import itertools
import pathlib
import sys

import copy

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
POWER_FACTOR = 0.95        # what a grid code typically asks of a converter
I_CONTINUOUS = 1.0


def zone_tables(case, model):
    """Zonal reactive load and the converters that sit in each zone."""
    zones = np.unique(model.conv_zone)
    return zones, model.zone_q_mvar


def capacity_share(case, model) -> float:
    """Converters' share of installed capacity, worst zone.

    The rule a market would write without looking at a power flow: a resource
    is asked for the share of the duty that matches the share of the plant it
    represents. Worst zone, because the requirement is per zone and has to
    hold in all of them.
    """
    zones = np.unique(model.conv_zone)
    sync_zone = case.network.zone_of_bus[case.sync_bus]
    worst = np.inf
    for zone in zones:
        conv = float(case.conv_rating_mva[model.conv_zone == zone].sum())
        sync = float(case.sync_rating_mva[sync_zone == zone].sum())
        if conv + sync <= 0.0:
            continue
        worst = min(worst, conv / (conv + sync))
    return float(worst)


def power_factor_rule(case, model) -> float:
    """What a fixed power factor of rating implies, as a fraction of load."""
    room = float(np.sqrt(max(1.0 - POWER_FACTOR ** 2, 0.0)))
    zones = np.unique(model.conv_zone)
    worst = np.inf
    for zone in zones:
        supply = room * float(case.conv_rating_mva[model.conv_zone == zone]
                              .sum())
        demand = model.zone_q_mvar[:, int(zone)].max()
        if demand <= 0.0:
            continue
        worst = min(worst, supply / float(demand))
    return float(min(worst, 1.0))


def capability_cap(case, model) -> float:
    """The most the fleet can hold inside its continuous current limit.

    Reactive room is what the continuous constraint leaves once the converter
    is carrying its scheduled active power, so this is a ceiling rather than a
    target: a requirement above it is not conservative, it is unmeetable by
    the devices themselves before any network is considered.
    """
    zones = np.unique(model.conv_zone)
    worst = np.inf
    for zone in zones:
        members = np.flatnonzero(model.conv_zone == zone)
        if not len(members):
            continue
        demand = model.zone_q_mvar[:, int(zone)]
        for period in range(model.T):
            if demand[period] <= 0.0:
                continue
            room = 0.0
            for i in members:
                v = float(case.conv_voltage_pu[period, i])
                s = float(case.conv_rating_mva[i])
                p = float(model.injection[period, i])
                reach = (v * I_CONTINUOUS * s) ** 2 - p ** 2
                room += float(np.sqrt(reach)) if reach > 0.0 else 0.0
            worst = min(worst, room / float(demand[period]))
    return float(min(worst, 1.0))


RULES = {
    "capacity share": capacity_share,
    "power factor": power_factor_rule,
    "capability cap": capability_cap,
}


def carries(name: str, share: float, fraction: float, settings) -> float:
    """Worst voltage violation left after correcting the controls.

    This is step three and it is deliberately not part of any rule above. A
    rule that consulted this would be a rule chosen for its answer.
    """
    base = build_case(name, converter_share=share,
                      demand_peak_mw=common_demand_peak(name))
    case = dataclasses.replace(
        base, conv_injection_pu=np.full(len(base.demand_mw), DISPATCH),
        reactive_frac=float(fraction))
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
    for name, share in itertools.product(CASES, SHARES):
        base = build_case(name, converter_share=share,
                          demand_peak_mw=common_demand_peak(name))
        case = dataclasses.replace(
            base, conv_injection_pu=np.full(len(base.demand_mw), DISPATCH))
        model = MultiPeriod(case, settings, "current")
        row = dict(case=name, share=share, used_in_paper=0.35)
        for label, rule in RULES.items():
            row[label] = rule(case, model)
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "reactive_rule.csv", index=False)
    print("규칙마다 나오는 무효 요구 비율 (구역 무효부하 대비)\n")
    print(frame.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
    print("\n논문이 쓰는 0.35 와 견주어:")
    for _, row in frame.iterrows():
        for label in RULES:
            value = float(row[label])
            mark = "이하" if value <= 0.35 else "초과"
            print(f"  {row.case:7s} {row.share:.1f}  {label:15s} "
                  f"{value:.4f}  ({mark})")

    print("\n그 값에서 계통이 버티는가 (전압제어 보정 후 잔여):")
    for index, row in frame.iterrows():
        for label in list(RULES) + ["used_in_paper"]:
            value = float(row[label])
            residual = carries(row.case, row.share, value, settings)
            frame.loc[index, f"residual at {label}"] = residual
            verdict = ("닫힘" if residual <= 1e-9
                       else ("해 없음" if not np.isfinite(residual)
                             else f"잔여 {residual:.2e}"))
            print(f"  {row.case:7s} {row.share:.1f}  {label:15s} "
                  f"{value:.4f}  ->  {verdict}")

    frame.to_csv(RESULTS / "reactive_rule.csv", index=False)
    print(f"\nwrote {RESULTS / 'reactive_rule.csv'}")


if __name__ == "__main__":
    sys.exit(main())
