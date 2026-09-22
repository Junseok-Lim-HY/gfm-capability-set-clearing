#!/usr/bin/env python3
"""Move the voltage controls until the profile is inside its band.

Two callers want this and they want the same thing. study/base_case_repair.py
puts a shipped case inside its declared limits before anything is scheduled on
it; the iterative correction of Section 5.11 does the same to each period of a
cleared schedule. Keeping one copy is not tidiness. The study spent an evening
reporting a repaired network's clearing next to an unrepaired network's ac
check, and the reason was a second copy of the network-building code.

What this does not do is redispatch. Active power is the market's to set and
this is the voltage-control layer, which is why it can sit outside the linear
programme without disturbing the duals that are its prices. Generator voltage
setpoints and transformer taps are what an operator moves between dispatches,
and they are what moves here.

The search is a plain local one, and deliberately so. It is not looking for
the best setpoints; it is looking for setpoints that clear the band while
staying near the ones the case came with, and a large move would be a
different system rather than a better answer.
"""
from __future__ import annotations

import copy

import numpy as np
import pandapower as pp

SETPOINT_STEP = 0.005
SETPOINT_SPAN = 0.06        # how far a setpoint may move from where it started
ROUNDS = 12


def band(net):
    """Declared voltage limits, with the pandapower defaults where absent."""
    return (net.bus.min_vm_pu.fillna(0.94).to_numpy(),
            net.bus.max_vm_pu.fillna(1.06).to_numpy())


def violation(net, enforce_q_lims: bool = True) -> float:
    """Squared voltage violation over all buses, in per unit."""
    try:
        pp.runpp(net, enforce_q_lims=enforce_q_lims, numba=False)
    except Exception:                                    # noqa: BLE001
        return float("inf")
    low, high = band(net)
    v = net.res_bus.vm_pu.to_numpy()
    return float(np.sum(np.maximum(low - v, 0.0) ** 2
                        + np.maximum(v - high, 0.0) ** 2))


def correct(net, enforce_q_lims: bool = True, rounds: int = ROUNDS):
    """Least-change local search over generator setpoints and taps.

    Returns the corrected network and the residual violation. A residual above
    zero means the band cannot be cleared with these controls alone, which is
    a result about the case rather than a failure of the search, and it is
    returned rather than raised.
    """
    start = net.gen.vm_pu.to_numpy().copy() if len(net.gen) else np.array([])
    start_ext = (net.ext_grid.vm_pu.to_numpy().copy() if len(net.ext_grid)
                 else np.array([]))
    best = violation(net, enforce_q_lims)
    if best <= 1e-12:
        return net, best

    for _ in range(rounds):
        improved = False

        for index in net.gen.index:
            shipped = float(start[net.gen.index.get_loc(index)])
            for step in (SETPOINT_STEP, -SETPOINT_STEP):
                value = float(net.gen.at[index, "vm_pu"]) + step
                if abs(value - shipped) > SETPOINT_SPAN + 1e-9:
                    continue
                trial = copy.deepcopy(net)
                trial.gen.at[index, "vm_pu"] = value
                score = violation(trial, enforce_q_lims)
                if score < best - 1e-12:
                    net, best, improved = trial, score, True
                    break

        # The reference machine has an excitation system like any other, and
        # its setpoint is a voltage control like any other. Leaving it out
        # left the one machine whose reactive output nothing else could move.
        for index in net.ext_grid.index:
            shipped = float(start_ext[net.ext_grid.index.get_loc(index)])
            for step in (SETPOINT_STEP, -SETPOINT_STEP):
                value = float(net.ext_grid.at[index, "vm_pu"]) + step
                if abs(value - shipped) > SETPOINT_SPAN + 1e-9:
                    continue
                trial = copy.deepcopy(net)
                trial.ext_grid.at[index, "vm_pu"] = value
                score = violation(trial, enforce_q_lims)
                if score < best - 1e-12:
                    net, best, improved = trial, score, True
                    break

        for index in net.trafo.index:
            position = net.trafo.at[index, "tap_pos"]
            if position != position:                     # NaN
                continue
            low = net.trafo.at[index, "tap_min"]
            high = net.trafo.at[index, "tap_max"]
            for step in (1, -1):
                moved = position + step
                if low == low and moved < low:
                    continue
                if high == high and moved > high:
                    continue
                trial = copy.deepcopy(net)
                trial.trafo.at[index, "tap_pos"] = moved
                score = violation(trial, enforce_q_lims)
                if score < best - 1e-12:
                    net, best, improved = trial, score, True
                    break

        if not improved or best <= 1e-12:
            break
    return net, best


def moved(before, after) -> list[dict]:
    """Which controls ended up somewhere else, and where."""
    changes = []
    for index in after.gen.index:
        was = float(before.gen.at[index, "vm_pu"])
        now = float(after.gen.at[index, "vm_pu"])
        if abs(now - was) > 1e-9:
            changes.append(dict(control="gen", index=int(index),
                                at=int(after.gen.at[index, "bus"]),
                                was=was, now=now))
    for index in after.ext_grid.index:
        was = float(before.ext_grid.at[index, "vm_pu"])
        now = float(after.ext_grid.at[index, "vm_pu"])
        if abs(now - was) > 1e-9:
            changes.append(dict(control="ext", index=int(index),
                                at=int(after.ext_grid.at[index, "bus"]),
                                was=was, now=now))
    for index in after.trafo.index:
        was = before.trafo.at[index, "tap_pos"]
        now = after.trafo.at[index, "tap_pos"]
        if was == was and now == now and was != now:
            changes.append(dict(control="tap", index=int(index),
                                at=int(after.trafo.at[index, "hv_bus"]),
                                was=float(was), now=float(now)))
    return changes
