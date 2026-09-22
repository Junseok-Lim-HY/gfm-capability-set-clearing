#!/usr/bin/env python3
"""Correct the voltage controls between clearings, and see what is left.

Section 5.11 reports that every cleared schedule violates something. The
review asks for those limits to enter "the clearing or an iterative
correction". This is the second, and it is the one that fits: voltage is set
by generator setpoints, transformer taps and shunts, which are the voltage
control layer's to move and not products the market clears. Putting them in
the linear programme would make them priced quantities, which they are not,
and would put at risk the one property the whole paper depends on -- that the
programme is linear and its duals are the prices.

So the loop is: clear, run the ac power flow at that schedule, move the
voltage controls in each period until its profile is inside the band, feed
the resulting terminal voltages back into the capability set, and clear
again. The controls are corrected per period rather than once for the day,
because a profile lifted to clear the floor at the peak sits against the
ceiling at the trough.

What this can establish is bounded and worth stating in advance. If the band
closes, the voltage excursions of Section 5.11 were a matter of control
settings the study had not adjusted, and the schedules stand. If a residual
survives, no setting of these controls delivers the schedule and the
requirement is asking for something the network cannot carry -- which is a
result about the case, not a failure of the loop. Either way the reactive
limits and the branch loadings are checked afterwards rather than assumed,
because moving a setpoint to fix a voltage spends reactive power somewhere.

Writes results/ac_clearing.csv.
"""
from __future__ import annotations

import copy
import dataclasses
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import ac_clearing  # noqa: E402
import voltage_control  # noqa: E402
from multiperiod import (MultiPeriod, ac_feasibility,  # noqa: E402
                         ac_voltage_pass, build_case, common_demand_peak)
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
ROUNDS = 5
DISPATCH = 0.8
DAMPING = 0.5          # how far the assumed voltages move toward the result
# The loss term gets the same treatment, and for the same reason. The loss
# sensitivity is itself a function of the operating point, so replacing it
# outright each round lets the schedule overshoot, puts the next
# linearisation somewhere else, and overshoots again. Undamped it swings
# the worst branch loading between 83 and 102 per cent on RTS-24 and
# drives the power flow to divergence on IEEE 39-bus. That is a property
# of the iteration, not of charging for losses, and it should not be
# reported as the latter.
LOSS_DAMPING = 0.5


def corrected(case, outcome, controls):
    """Move each period's voltage controls until its profile is inside band.

    The settings accumulate. `moved` reports what changed relative to the
    network it was handed, and that network already carries the previous
    round's settings, so returning only the new moves would throw the old ones
    away every round -- which is what the first version did, and why the
    number of controls it moved oscillated between sixty and six.
    """
    networks: list = []
    ac_voltage_pass(case, outcome, solved=networks, controls=controls)
    carried, corrected_nets, residual = [], [], 0.0
    for period, net in enumerate(networks):
        before = copy.deepcopy(net)
        fixed, left = voltage_control.correct(net)
        residual = max(residual, left)
        settings = (copy.deepcopy(controls[period]) if controls
                    else {"gen": {}, "tap": {}})
        for change in voltage_control.moved(before, fixed):
            settings.setdefault(change["control"], {})[change["index"]] =                 change["now"]
        carried.append(settings)
        corrected_nets.append(fixed)
    # The corrected networks, not the ones handed in. Sensitivities taken
    # before the correction describe an operating point the next clearing will
    # never be near, and the clearing would then optimise against a
    # linearisation of somewhere else.
    return carried, residual, corrected_nets


def one(name: str, share: float, form: str, settings,
        include=("branch",), charge_losses: bool = False) -> list[dict]:
    # What the clearing was made to carry, written out in full. Recording
    # only `include` left the branch run and the branch-plus-loss run bearing
    # the same label, which is the sort of thing that reads as a duplicate row
    # rather than as two experiments.
    label = "+".join(include) or "none"
    if charge_losses:
        label += "+loss"
    base = build_case(name, converter_share=share,
                      demand_peak_mw=common_demand_peak(name))
    case = dataclasses.replace(
        base, conv_injection_pu=np.full(len(base.demand_mw), DISPATCH))

    rows, controls, extra, losses = [], None, None, None
    tolerances = ac_clearing.Tolerances()
    for index in range(ROUNDS):
        started = time.perf_counter()
        # One model object, used both to solve and to build the next round's
        # rows. Two would not be guaranteed to lay their columns out the same
        # way, and a row pointing at the wrong column still solves.
        model = MultiPeriod(case, settings, form, extra=extra, losses=losses)
        outcome = model.solve()
        if not outcome["feasible"]:
            rows.append(dict(case=name, share=share, form=form, round=index,
                             carried=label,
                             feasible=False))
            print(f"  round {index}: 해 없음")
            break

        try:
            controls, residual, networks = corrected(case, outcome, controls)
        except Exception as error:                       # noqa: BLE001
            # A power flow that does not converge is a fact about this round,
            # not about the run. Record it and stop this configuration; the
            # others still have something to say.
            rows.append(dict(case=name, share=share, form=form, round=index,
                             carried=label, feasible=True, converged=False,
                             cost=outcome["cost"],
                             note=type(error).__name__))
            print(f"  round {index}: 조류가 안 풀립니다 ({type(error).__name__})")
            break
        # The screen has to see the corrected controls, or it reports the
        # profile the loop just finished fixing.
        screen = ac_feasibility(case, outcome, controls=controls)
        moves = sum(len(c["gen"]) + len(c["tap"]) for c in controls)

        # The corrected voltages are what the capability set should have been
        # written at, so they go back into the case -- damped, because the map
        # from assumed voltage to resulting voltage is not a contraction and
        # Section 4.4 says so.
        resulting = ac_voltage_pass(case, outcome, controls=controls)
        blended = (case.conv_voltage_pu
                   + DAMPING * (resulting - case.conv_voltage_pu))
        move = float(np.abs(blended - case.conv_voltage_pu).max())
        case = dataclasses.replace(case, conv_voltage_pu=blended)

        # Branch flows are what the clearing does move, so they go on as
        # rows. Voltage does not: the correction above owns it. Losses are
        # neither -- they are a term the dc balance omits, so they are folded
        # into the balance rather than appended as a constraint.
        extra = ac_clearing.rows_for(model, case, outcome, networks,
                                     tolerances, include=include)
        if charge_losses:
            fresh = ac_clearing.loss_rows(model, case, outcome, networks)
            if losses is None:
                losses = fresh
            else:
                losses = [(old_row + LOSS_DAMPING * (new_row - old_row),
                           old_c + LOSS_DAMPING * (new_c - old_c))
                          for (old_row, old_c), (new_row, new_c)
                          in zip(losses, fresh)]

        rows.append(dict(case=name, share=share, form=form, round=index,
                         carried=label,
                         feasible=True, converged=True,
                         cost=outcome["cost"],
                         residual_pu2=residual, controls_moved=moves,
                         v_move_pu=move, v_min=screen["v_min"],
                         v_max=screen["v_max"],
                         loading_max=screen["loading_max"],
                         q_at_limit=screen["q_at_limit"],
                         branch_rows=len(extra[0]),
                         losses_charged=bool(charge_losses),
                         loss_mw=(float(sum(c for _, c in losses))
                                  if losses is not None else float("nan")),
                         seconds=time.perf_counter() - started))
        print(f"  round {index}: cost {outcome['cost']:,.0f}  "
              f"V {screen['v_min']:.4f}~{screen['v_max']:.4f}  "
              f"잔여 {residual:.2e}  제어 {moves}개  "
              f"부하 {screen['loading_max']:.1f}%  "
              f"V이동 {move:.4f}  ({time.perf_counter()-started:.0f}s)")
        if (residual <= 1e-9 and move < 1e-3
                and screen["loading_max"] <= 100.0):
            print("    전압·선로 모두 한계 안이고 더 움직이지 않습니다")
            break
    return rows


def main() -> None:
    settings = Capability(i_short_term=1.5, priority="reactive",
                          soc_threshold=0.20)
    # A target is case:share:form, optionally with :rows where rows names
    # what the clearing is made to carry. "branch" is what the loop has been
    # running; "branch+loss" adds the loss term to the balance; "none" turns
    # both off, which is the comparison the other two are read against.
    argv = sys.argv[1:]
    # The variant runs must not land in the same file as the baseline loop.
    # The checker reads that file grouped by case and share alone and takes
    # each group's last round, so three variants in one file give it three
    # last rounds and no way to tell them apart.
    out = "ac_clearing.csv"
    if argv and argv[0].startswith("out="):
        out = argv[0][4:]
        argv = argv[1:]
    targets = argv or ["rts24:0.2:current"]
    rows = []
    for target in targets:
        parts = target.split(":")
        name, share, form = parts[0], parts[1], parts[2]
        carried = parts[3] if len(parts) > 3 else "branch"
        include = tuple(k for k in ("branch", "voltage", "reactive")
                        if k in carried)
        charge = "loss" in carried
        print(f"=== {name} share {share} {form}  "
              f"rows={include or '()'} losses={'on' if charge else 'off'}")
        rows += one(name, float(share), form, settings,
                    include=include, charge_losses=charge)

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / out, index=False)
    print(f"\nwrote {RESULTS / out}")


if __name__ == "__main__":
    sys.exit(main())
