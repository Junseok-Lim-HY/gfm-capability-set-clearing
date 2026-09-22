#!/usr/bin/env python3
"""Clear against a linearised ac network, and iterate until it stops moving.

The clearing decides a schedule under a dc active-power flow, and the ac
network then reports what that schedule actually does to voltages, machine
reactive output, branch loading and the slack. Section 5.11 shows those
reports failing in every configuration. This closes the loop: the ac
relations are linearised about the current schedule, written as rows the
linear programme can carry, and the schedule is re-cleared against them. The
programme stays linear, so its duals stay prices, which is the constraint the
whole paper works under.

The loop is the point. A linearisation is only good near the point it was
taken at, and model/ac_linear.py measures how good: at a movement of a few
tens of megawatts the voltage error is a few times 1e-5 per unit, at a couple
of hundred it is a few times 1e-3. So the first pass is not expected to land
inside the limits; each pass moves less than the last, and the linearisation
it is written against is better each time.

Nothing here rebuilds the network. It asks multiperiod.ac_voltage_pass for
the per-period networks it already builds at a cleared schedule, because a
second copy of that logic is exactly how the study came to be running the
clearing on a repaired network and the ac checks on the shipped one.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandapower as pp

import ac_linear
import ac_rows


@dataclasses.dataclass
class Tolerances:
    """How much slack each linearised row is given.

    Not a taste. The linearisation carries a measured error, and a row
    tightened past it would have the clearing chase a limit the power flow
    does not agree is there, oscillating between two schedules that each look
    feasible to the model that produced them. These are set from what
    ac_linear.validate reports over the movements this loop actually makes.
    """
    voltage_pu: float = 2.0e-3
    reactive_mvar: float = 15.0
    branch_mw: float = 5.0


def controlled_buses(case) -> np.ndarray:
    """Buses whose injections the clearing sets, each listed once."""
    return np.unique(np.concatenate([np.asarray(case.conv_bus_pp, dtype=int),
                                     np.asarray(case.sync_bus_pp, dtype=int)]))


def injections_at(model, case, buses, outcome, period: int):
    """Map each bus to the LP columns that inject into it, and to its level.

    Several converters can sit on one bus -- RTS-24 puts six on bus 21 -- and
    a sensitivity is per bus, so the columns are summed rather than listed
    separately. The level is what the columns held when the sensitivities
    were measured, which is what the deltas are measured from -- the
    columns, not the physical injection. The two differ by the fixed
    converter injection, which is a constant of the case and has no column,
    so it cannot appear on the variable side of the difference. Carrying it
    on the level side alone made the deviation at the measured point equal
    to minus that injection rather than zero, which anchored every branch
    row to a system with the converters switched off: on IEEE 39-bus at 40
    per cent share the cleared schedule then violated its own branch rows
    by 543 MW while the power flow put the worst branch at 99.8 per cent of
    rating, and the round built from those rows had nowhere to go.
    """
    inj = ac_rows.Injections(model.size)
    net_p = outcome["pd"] - outcome["pc"] - outcome["curt"]
    for bus in buses:
        active, reactive = [], []
        p0 = q0 = 0.0
        for i in np.flatnonzero(np.asarray(case.conv_bus_pp) == bus):
            active += [(model.index["pd"][period, i], 1.0),
                       (model.index["pc"][period, i], -1.0),
                       (model.index["curt"][period, i], -1.0)]
            reactive += [(model.index["q"][period, i], 1.0)]
            p0 += float(net_p[period, i])
            q0 += float(outcome["q"][period, i])
        for g in np.flatnonzero(np.asarray(case.sync_bus_pp) == bus):
            active += [(model.index["pg"][period, g], 1.0)]
            p0 += float(outcome["pg"][period, g])
        inj.add(active, reactive, p0, q0)
    return inj


def limits(net):
    """Declared voltage band, machine reactive capability, branch ratings."""
    low = net.bus.min_vm_pu.fillna(0.94).to_numpy()
    high = net.bus.max_vm_pu.fillna(1.06).to_numpy()
    q_low = net.gen.min_q_mvar.to_numpy(dtype=float)
    q_high = net.gen.max_q_mvar.to_numpy(dtype=float)
    rating = np.concatenate([
        (net.line.max_i_ka.to_numpy(dtype=float)
         * net.bus.vn_kv.to_numpy(dtype=float)[net.line.from_bus.to_numpy()]
         * np.sqrt(3.0)),
        net.trafo.sn_mva.to_numpy(dtype=float)])
    return low, high, q_low, q_high, rating


def rows_for(model, case, outcome, networks, tolerances: Tolerances,
             include=("branch",)):
    """Linearised ac rows, for every period, of the kinds asked for.

    `include` is not a convenience. Writing a row for a quantity the clearing
    cannot move makes the programme infeasible without teaching it anything,
    which is what happened the first time these rows went in: voltage rows
    went on, and the clearing has no lever that reaches a bus voltage, because
    voltage is held by generator setpoints and transformer taps and those are
    not products a market clears. They are corrected outside the programme
    instead.

    What the clearing does move is where the active power comes from and how
    much reactive each converter is dispatched to, and those decide branch
    flows and losses. So those are the rows it gets.

    Machine reactive limits are also left out, and for the same reason rather
    than for convenience: a machine's reactive output is a consequence of
    voltage control, and a machine at its limit means the control layer has
    run out of room at that bus. Handing that to the market asks the market to
    fix something it does not hold. It is reported instead.
    """
    buses = controlled_buses(case)
    rows, rhs = [], []
    for period, net in enumerate(networks):
        sens = ac_linear.sensitivities(net, buses)
        inj = injections_at(model, case, buses, outcome, period)
        low, high, q_low, q_high, rating = limits(net)

        if "voltage" in include:
            more, more_rhs = ac_rows.voltage_rows(sens, inj, low, high,
                                                  tolerances.voltage_pu)
            rows += more
            rhs += more_rhs
        if "reactive" in include:
            more, more_rhs = ac_rows.reactive_rows(sens, inj, q_low, q_high,
                                                   tolerances.reactive_mvar)
            rows += more
            rhs += more_rhs
        if "branch" in include:
            more, more_rhs = ac_rows.branch_rows(sens, inj, rating,
                                                 tolerances.branch_mw)
            rows += more
            rhs += more_rhs
    return rows, rhs


def loss_rows(model, case, outcome, networks):
    """One linearised loss expression per period, for the energy balance.

    Kept apart from rows_for() because it is a different kind of object. Those
    are inequalities appended after the requirement rows; this modifies an
    equality that is already there. Returning it through the same channel
    would invite it to be appended as a row, which would constrain the losses
    rather than charge for them, and the balance would still not know they
    exist.
    """
    buses = controlled_buses(case)
    out = []
    for period, net in enumerate(networks):
        sens = ac_linear.sensitivities(net, buses)
        inj = injections_at(model, case, buses, outcome, period)
        out.append(ac_rows.loss_row(sens, inj))
    return out


def report(networks, case=None) -> dict:
    """Worst violation of each kind over the day, on the given networks."""
    worst = dict(v_min=np.inf, v_max=-np.inf, v_low=np.inf, v_high=-np.inf,
                 loading=0.0, q_slack=0.0)
    for net in networks:
        pp.runpp(net, enforce_q_lims=True, numba=False)
        low = net.bus.min_vm_pu.fillna(0.94).to_numpy()
        high = net.bus.max_vm_pu.fillna(1.06).to_numpy()
        v = net.res_bus.vm_pu.to_numpy()
        worst["v_min"] = min(worst["v_min"], float(v.min()))
        worst["v_max"] = max(worst["v_max"], float(v.max()))
        worst["v_low"] = min(worst["v_low"], float((v - low).min()))
        worst["v_high"] = max(worst["v_high"], float((v - high).max()))
        loading = [net.res_line.loading_percent.max()]
        if len(net.res_trafo):
            loading.append(net.res_trafo.loading_percent.max())
        worst["loading"] = max(worst["loading"], float(np.nanmax(loading)))
        if len(net.res_ext_grid):
            worst["q_slack"] = max(worst["q_slack"],
                                   abs(float(net.res_ext_grid.p_mw.sum())))
    return worst
