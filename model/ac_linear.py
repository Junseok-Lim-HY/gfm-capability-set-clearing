#!/usr/bin/env python3
"""First-order ac sensitivities, measured rather than derived.

The clearing is a linear programme and its duals are the prices, so anything
the ac network has to say to it must arrive as linear rows. That means
linearising the power flow about an operating point: how each bus voltage,
each machine's reactive output, each branch flow and the slack's active
correction move when the clearing moves an injection.

Those derivatives are in the power-flow Jacobian, and pandapower will hand it
over. This module does not take it. Reading the Jacobian means reproducing
pandapower's internal bus ordering and its PV/PQ classification, which changes
when a machine hits a reactive limit, and a mistake there is silent: the rows
would still be linear, still be feasible, and still be wrong. Perturbing an
injection and re-solving has no index to get wrong, costs about 45 ms a
solve, and can be checked against a power flow at a point it was not fitted
to. That check is `validate` below, and it is the reason for the choice.

The quantities are the four the review names -- voltage, generator reactive
output, branch loading, slack mismatch -- and they are returned about one
operating point, for one period.
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pandapower as pp

STEP_MW = 10.0        # perturbation size for active injections
STEP_MVAR = 10.0      # and for reactive


@dataclasses.dataclass
class Point:
    """The operating point a set of sensitivities is taken about."""
    v: np.ndarray             # bus voltage magnitudes, pu
    q_gen: np.ndarray         # generator reactive output, MVAr
    p_branch: np.ndarray      # branch active flow at the from end, MW
    q_branch: np.ndarray      # branch reactive flow at the from end, MVAr
    losses: float             # total active losses, MW
    slack: float              # slack active injection, MW


@dataclasses.dataclass
class Sensitivity:
    """d(quantity)/d(injection) for every injection the clearing controls.

    Rows are the perturbed injections in the order given; columns are buses,
    generators or branches. Active and reactive perturbations are kept apart
    because the clearing controls them separately.
    """
    point: Point
    dv_dp: np.ndarray
    dv_dq: np.ndarray
    dq_gen_dp: np.ndarray
    dq_gen_dq: np.ndarray
    dp_branch_dp: np.ndarray
    dp_branch_dq: np.ndarray
    dq_branch_dp: np.ndarray
    dq_branch_dq: np.ndarray
    dlosses_dp: np.ndarray
    dlosses_dq: np.ndarray


# Reactive limits are NOT enforced inside the power flow, and that is the
# whole point. Letting the solver enforce them makes a machine switch from PV
# to PQ the moment it saturates, and the sensitivity is discontinuous across
# that switch; on the repaired RTS-24 seven machines of ten sit at a limit
# already, so the discontinuity is the normal case rather than a corner. With
# the limits off, every machine holds its terminal voltage and its reactive
# output is a smooth function of the injections -- measured error falls from
# 86 per cent of the movement to 5e-5 per unit. The limits then belong to the
# clearing, as explicit rows on that smooth function, which is where the
# review asks for them.
def _solve(net) -> Point:
    pp.runpp(net, enforce_q_lims=False, init="results" if "res_bus" in net
             and len(net.res_bus) else "auto")
    branch_p = np.concatenate([net.res_line.p_from_mw.to_numpy(),
                               net.res_trafo.p_hv_mw.to_numpy()])
    branch_q = np.concatenate([net.res_line.q_from_mvar.to_numpy(),
                               net.res_trafo.q_hv_mvar.to_numpy()])
    losses = float(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum())
    slack = float(net.res_ext_grid.p_mw.sum()) if len(net.res_ext_grid) else 0.0
    return Point(v=net.res_bus.vm_pu.to_numpy().copy(),
                 q_gen=net.res_gen.q_mvar.to_numpy().copy(),
                 p_branch=branch_p, q_branch=branch_q,
                 losses=losses, slack=slack)


def _nudge(net, kind: str, bus: int, amount: float):
    """Return a copy of the network with one injection moved.

    The injection is applied as a static generator so that it does not disturb
    the voltage control of the machine at that bus. Perturbing a voltage-
    controlled generator's active power leaves its terminal voltage pinned,
    which is what the clearing does too; perturbing its reactive power would
    not, and that is why reactive perturbations are applied here as an
    independent injection rather than as a change to a setpoint.
    """
    trial = copy.deepcopy(net)
    row = {"bus": bus, "p_mw": 0.0, "q_mvar": 0.0, "name": "probe"}
    row["p_mw" if kind == "p" else "q_mvar"] = amount
    pp.create_sgen(trial, **row)
    return trial


def sensitivities(net, buses, step_mw: float = STEP_MW,
                  step_mvar: float = STEP_MVAR) -> Sensitivity:
    """Central differences about the network's present operating point.

    `buses` are the buses whose injections the clearing controls. Central
    rather than one-sided differences because the power flow is not linear and
    a one-sided difference carries the curvature into the slope.
    """
    base = _solve(copy.deepcopy(net))
    shape = (len(buses),)

    def sweep(kind: str, step: float):
        rows = {k: [] for k in ("v", "q_gen", "p_branch", "q_branch", "loss")}
        for bus in buses:
            up = _solve(_nudge(net, kind, bus, step))
            down = _solve(_nudge(net, kind, bus, -step))
            scale = 2.0 * step
            rows["v"].append((up.v - down.v) / scale)
            rows["q_gen"].append((up.q_gen - down.q_gen) / scale)
            rows["p_branch"].append((up.p_branch - down.p_branch) / scale)
            rows["q_branch"].append((up.q_branch - down.q_branch) / scale)
            rows["loss"].append((up.losses - down.losses) / scale)
        return {k: np.array(v) for k, v in rows.items()}

    active = sweep("p", step_mw)
    reactive = sweep("q", step_mvar)

    return Sensitivity(
        point=base,
        dv_dp=active["v"], dv_dq=reactive["v"],
        dq_gen_dp=active["q_gen"], dq_gen_dq=reactive["q_gen"],
        dp_branch_dp=active["p_branch"], dp_branch_dq=reactive["p_branch"],
        dq_branch_dp=active["q_branch"], dq_branch_dq=reactive["q_branch"],
        dlosses_dp=active["loss"].reshape(shape),
        dlosses_dq=reactive["loss"].reshape(shape))


def predict(sens: Sensitivity, delta_p: np.ndarray,
            delta_q: np.ndarray) -> Point:
    """What the linearisation says the network does under a given move."""
    return Point(
        v=sens.point.v + delta_p @ sens.dv_dp + delta_q @ sens.dv_dq,
        q_gen=sens.point.q_gen + delta_p @ sens.dq_gen_dp
        + delta_q @ sens.dq_gen_dq,
        p_branch=sens.point.p_branch + delta_p @ sens.dp_branch_dp
        + delta_q @ sens.dp_branch_dq,
        q_branch=sens.point.q_branch + delta_p @ sens.dq_branch_dp
        + delta_q @ sens.dq_branch_dq,
        losses=sens.point.losses + float(delta_p @ sens.dlosses_dp
                                         + delta_q @ sens.dlosses_dq),
        slack=sens.point.slack)


def validate(net, buses, delta_p: np.ndarray, delta_q: np.ndarray,
             step_mw: float = STEP_MW, step_mvar: float = STEP_MVAR) -> dict:
    """Compare the linearisation against a power flow it was not fitted to.

    A linearisation is only worth putting into a clearing if it is accurate
    over the range the clearing will move things. Fitting it at one point and
    testing it at that same point says nothing; this moves the injections by
    `delta_p` and `delta_q`, which the caller should make as large as the
    clearing plausibly would, and reports the error in each quantity.
    """
    sens = sensitivities(net, buses, step_mw, step_mvar)
    guess = predict(sens, delta_p, delta_q)

    moved = copy.deepcopy(net)
    for bus, dp, dq in zip(buses, delta_p, delta_q):
        pp.create_sgen(moved, bus=bus, p_mw=float(dp), q_mvar=float(dq),
                       name="move")
    truth = _solve(moved)

    return dict(
        v_max_err=float(np.abs(guess.v - truth.v).max()),
        v_move=float(np.abs(truth.v - sens.point.v).max()),
        q_gen_max_err=float(np.abs(guess.q_gen - truth.q_gen).max()),
        q_gen_move=float(np.abs(truth.q_gen - sens.point.q_gen).max()),
        p_branch_max_err=float(np.abs(guess.p_branch - truth.p_branch).max()),
        p_branch_move=float(np.abs(truth.p_branch
                                   - sens.point.p_branch).max()),
        q_branch_max_err=float(np.abs(guess.q_branch - truth.q_branch).max()),
        losses_err=float(abs(guess.losses - truth.losses)),
        losses_move=float(abs(truth.losses - sens.point.losses)))
