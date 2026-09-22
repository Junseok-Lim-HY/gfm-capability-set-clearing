#!/usr/bin/env python3
"""Put the event voltage into the short-term face, and check what arrives.

Section 7.3 shows that a converter reaches the capability boundary but at the
terminal voltage it actually arrives at, which is below the pre-contingency
value the constraint is written at, and Eq. (27) says by how much. That is a
correction, and until it is applied it is only a remark: the clearing still
schedules against the optimistic voltage, so it still sells inertial power the
converter will not deliver.

This applies it. For each converter and period the reactance behind the
terminal is measured from the power flow, the voltage the short-term face
should have been written at is computed, the day is cleared again with that
voltage in the short-term face alone -- the continuous face is a
pre-disturbance constraint and keeps the pre-disturbance voltage -- and the
resulting schedule is put back through the converter model of Section 7.3.

The question that matters is the last one: does the corrected schedule deliver
the inertial power it sold? Selling less and delivering it is the point;
selling the same and delivering less is what the correction was for.

It is not asked as a pass against a tolerance, because a shortfall here has
two causes and the correction can only reach one of them. Part of it is the
short-term face having been written at a voltage above the one that obtains,
which is what Eq. (27) measures and what re-clearing removes. The rest is the
time the outer loop needs to build the response, which belongs to the virtual
inertia and the damping rather than to the capability set -- Section 7.3 keeps
those two apart and so does this. Charging the correction for the second would
say it failed at something it was never a claim about.

Delivery is therefore read twice. The settled value is what the converter can
hold, and is the right reading while the current limiter is pinning the output
flat, which it does in every run of Section 7.3's sweep. It is not the right
reading here: the limiter does not always bind, and a machine nothing pins
rings at a damping ratio near 0.11 that has not decayed inside the containment
window, so the last fifth of the window samples a phase rather than a level.
The mean over the whole window is the energy frequency containment actually
receives, and it is well defined whether the limiter binds or not.

Either way it is aggregated by what each cell sold rather than reported as a
worst case over cells. The inertia requirement binds on the fleet sum, and a
ratio taken over a cell scheduled a megawatt is a statement about its
denominator.

Holding the reactive power at its schedule makes the correction close under
its own algebra. With Q fixed, Eq. (27) between the pre-event point and the
event gives

    V_corr^2 = V_pre^2 - x^2 (I_short^2 - I_pre^2) ,

so only the reactance and the current change matter, and neither needs the
clearing to be solved first.

Writes results/corrected_face.csv.
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

import ac_linear  # noqa: E402
import dynamic_check as dyn  # noqa: E402
from multiperiod import (MultiPeriod, ac_voltage_pass,  # noqa: E402
                         build_case, common_demand_peak)
from procurement_lp import Capability  # noqa: E402

sys.path.insert(0, str(ROOT / "study"))

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
CASES = ("ieee39", "rts24")
SHARES = (0.2, 0.4)
DISPATCH = 0.8
BASE_MVA = 100.0


def thevenin(case, outcome, settings) -> np.ndarray:
    """Reactance behind each converter terminal, per unit of its own rating.

    Measured rather than assumed: the voltage a bus loses per megavar injected
    into it is the reactance looking back from that bus, and ac_linear already
    perturbs and re-solves to get it. Converters that share a bus share the
    reactance and are given the bus value scaled by their own rating.
    """
    networks: list = []
    ac_voltage_pass(case, outcome, solved=networks)
    buses = sorted(set(int(b) for b in case.conv_bus_pp))
    reactance = np.zeros((len(networks), len(case.conv_bus_pp)))
    for period, net in enumerate(networks):
        sens = ac_linear.sensitivities(net, buses)
        for column, bus in enumerate(buses):
            at_bus = sens.dv_dq[column, bus] * sens.point.v[bus] * BASE_MVA
            for i in np.flatnonzero(np.asarray(case.conv_bus_pp) == bus):
                reactance[period, i] = (at_bus * case.conv_rating_mva[i]
                                        / BASE_MVA)
    return reactance


def event_voltage(case, outcome, reactance, settings) -> np.ndarray:
    """What the short-term face should have been written at."""
    net_p = (outcome["pd"] - outcome["pc"] - outcome["curt"]
             + np.outer(np.asarray(case.conv_injection_pu, dtype=float),
                        case.conv_rating_mva))
    corrected = np.array(case.conv_voltage_pu, dtype=float).copy()
    for t, i in itertools.product(range(corrected.shape[0]),
                                  range(corrected.shape[1])):
        v = float(case.conv_voltage_pu[t, i])
        s = float(case.conv_rating_mva[i])
        pre = float(np.hypot(net_p[t, i], outcome["q"][t, i])) / (v * s)
        short = settings.i_short_term
        drop = reactance[t, i] ** 2 * (short ** 2 - pre ** 2)
        corrected[t, i] = float(np.sqrt(max(v ** 2 - drop, 1e-6)))
    return corrected


def delivered(case, outcome, reactance, settings, period: int, i: int) -> dict:
    """Put one converter's cleared schedule through the model of Section 7.3.

    Every coordinate of the operating point comes from the clearing, not just
    the two that are easy to pass. The first version of this left the model's
    own active power and reserve in place and handed it only H and Q, which
    simulated a converter at 0.60 and 0.10 per unit while the clearing had put
    it anywhere from -0.15 to 0.80 with up to 0.45 of reserve. The delivery
    ratios that produced were for a machine nobody scheduled.
    """
    rating = float(case.conv_rating_mva[i])
    v = float(case.conv_voltage_pu[period, i])
    injection = (float(case.conv_injection_pu[period]) * rating
                 if case.conv_injection_pu is not None else 0.0)
    active = (float(outcome["pd"][period, i]) - float(outcome["pc"][period, i])
              - float(outcome["curt"][period, i]) + injection) / rating
    reserve = float(outcome["r"][period, i]) / rating
    q = float(outcome["q"][period, i]) / rating
    h = float(outcome["h"][period, i]) / rating
    if h <= 1e-9:
        return dict(scheduled=0.0, delivered=0.0, ratio=np.nan)
    x = max(float(reactance[period, i]), 1e-3)
    trace = dyn.run(v, q, settings.i_short_term, settings.priority, h, 0.0,
                    x, "hold schedule", p_sched=active, reserve=reserve)
    baseline = active + dyn.OVERLAP * reserve
    whole = trace[(trace.t >= dyn.EVENT_AT)
                  & (trace.t <= dyn.EVENT_AT + dyn.CONTAINMENT)]
    tail = trace[(trace.t >= dyn.EVENT_AT + 0.8 * dyn.CONTAINMENT)
                 & (trace.t <= dyn.EVENT_AT + dyn.CONTAINMENT)]
    # Two measures, because they answer different questions and one of them
    # stops meaning anything when the limiter never binds. The settled value
    # is what the converter can hold, and is the right reading while the
    # limiter is pinning the output flat. Where nothing pins it the machine
    # rings -- lightly damped, and not decayed inside one second -- and the
    # last fifth of the window samples a phase of that ring rather than a
    # level. The energy over the whole window is what frequency containment
    # actually receives, and it is well defined either way.
    energy = float(whole.p_out.mean() - baseline)
    settled = float(tail.p_out.mean() - baseline)
    return dict(scheduled=h, energy=energy, settled=settled,
                energy_ratio=energy / h, settled_ratio=settled / h,
                limited=bool(whole.clipped.any()))


def delivery(case, outcome, reactance, settings) -> dict:
    """How much of the day's scheduled inertial power arrives.

    Weighted by what each cell actually sold, because the inertia requirement
    is a constraint on the fleet sum and a converter holding 1 MW and one
    holding 381 MW do not bear on it equally. The first version of this took
    the least ratio over all cells, which reported the smallest denominator
    rather than the worst delivery: the cell it named was scheduled
    \SI{1.1}{MW}, swung either side of zero on a transient, and returned
    -1.88 while its settled value was exactly 1.0000.

    The spread is kept, since an aggregate can hide a converter that misses
    badly, but as the median and the tenth percentile of the distribution
    rather than as its minimum.
    """
    ratio_e, ratio_s, weight, limited = [], [], [], 0
    for t, i in itertools.product(range(len(case.demand_mw)),
                                  range(len(case.conv_rating_mva))):
        got = delivered(case, outcome, reactance, settings, t, i)
        if not np.isfinite(got.get("energy_ratio", np.nan)):
            continue
        ratio_e.append(got["energy_ratio"])
        ratio_s.append(got["settled_ratio"])
        weight.append(float(outcome["h"][t, i]))
        limited += int(got["limited"])
    if not weight:
        return dict(energy=np.nan, settled=np.nan, energy_median=np.nan,
                    energy_p10=np.nan, settled_median=np.nan,
                    limited_frac=np.nan, cells=0)
    e = np.asarray(ratio_e)
    q = np.asarray(ratio_s)
    w = np.asarray(weight)
    return dict(energy=float((e * w).sum() / w.sum()),
                settled=float((q * w).sum() / w.sum()),
                energy_median=float(np.median(e)),
                energy_p10=float(np.quantile(e, 0.10)),
                settled_median=float(np.median(q)),
                limited_frac=limited / len(w), cells=len(w))


def main() -> None:
    settings = Capability(i_short_term=1.5, priority="reactive",
                          soc_threshold=0.20)
    rows = []
    for name, share in itertools.product(CASES, SHARES):
        base = build_case(name, converter_share=share,
                          demand_peak_mw=common_demand_peak(name))
        case = dataclasses.replace(
            base, conv_injection_pu=np.full(len(base.demand_mw), DISPATCH))

        plain = MultiPeriod(case, settings, "current").solve()
        reactance = thevenin(case, plain, settings)
        before = delivery(case, plain, reactance, settings)

        corrected = event_voltage(case, plain, reactance, settings)
        fixed = dataclasses.replace(case, conv_voltage_short_pu=corrected)
        after_outcome = MultiPeriod(fixed, settings, "current").solve()
        after = delivery(fixed, after_outcome, reactance, settings)

        moved = int((np.abs(plain["h"] - after_outcome["h"]) > 1e-6).sum())
        rows.append(dict(
            case=name, share=share,
            x_min=float(reactance.min()), x_max=float(reactance.max()),
            v_drop_pct=float(100.0 * (case.conv_voltage_pu - corrected).max()
                             / case.conv_voltage_pu.max()),
            cost_before=plain["cost"], cost_after=after_outcome["cost"],
            cost_change_pct=100.0 * (after_outcome["cost"] - plain["cost"])
            / plain["cost"],
            h_before=float(plain["h"].sum()),
            h_after=float(after_outcome["h"].sum()),
            h_cells_moved=moved, h_cells=int(plain["h"].size),
            cells=before["cells"],
            energy_before=before["energy"], energy_after=after["energy"],
            settled_before=before["settled"], settled_after=after["settled"],
            energy_median_before=before["energy_median"],
            energy_median_after=after["energy_median"],
            energy_p10_before=before["energy_p10"],
            energy_p10_after=after["energy_p10"],
            settled_median_before=before["settled_median"],
            settled_median_after=after["settled_median"],
            limited_before=before["limited_frac"],
            limited_after=after["limited_frac"]))
        row = rows[-1]
        print(f"{name} {share}: x {row['x_min']:.3f}~{row['x_max']:.3f}  "
              f"단기전압 강하 {row['v_drop_pct']:.2f}%  "
              f"비용 {row['cost_change_pct']:+.4f}%  "
              f"H합 {row['h_before']:.0f}->{row['h_after']:.0f} "
              f"({moved}칸 이동)  "
              f"에너지 {before['energy']:.3f}->{after['energy']:.3f} "
              f"(중앙 {before['energy_median']:.3f}->"
              f"{after['energy_median']:.3f}, "
              f"10분위 {before['energy_p10']:.3f}->"
              f"{after['energy_p10']:.3f})  "
              f"정착 {before['settled']:.3f}->{after['settled']:.3f}  "
              f"제한기 {before['limited_frac']:.0%}->{after['limited_frac']:.0%}")

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "corrected_face.csv", index=False)
    print("\n보정이 고치는 것은 전압 몫뿐입니다. 상승시간 몫은 가상관성과 "
          "제동의 성질이라 남습니다.")
    print(f"\nwrote {RESULTS / 'corrected_face.csv'}")


if __name__ == "__main__":
    sys.exit(main())
