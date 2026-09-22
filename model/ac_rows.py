#!/usr/bin/env python3
"""Turn measured ac sensitivities into rows a linear clearing can carry.

model/ac_linear.py measures how the network answers a change in injection.
This turns those answers into inequalities on the clearing's own variables, so
that the four things the review asks for -- bus voltage, generator reactive
output, branch loading and the active balance including losses -- are enforced
where the schedule is decided rather than checked after it.

Every row has the same shape. A network quantity y is written about the point
the sensitivities were taken at,

    y  =  y0 + sum_u  dy/dp_u (P_u - p0_u) + dy/dq_u (Q_u - q0_u) ,

with u running over the injections the clearing controls, and then bounded.
The constant folds into the right-hand side; what is left is linear in the
decision variables, which is the whole reason for doing it this way.

Branch limits are apparent-power limits, so they are discs in (P, Q) and get
the same treatment the capability set gets: an inscribed polygon, which is
exact to cos(pi/m) and stays linear. Using the same device twice is
deliberate. The reader has already been told what an inscribed polygon costs
and how it was measured, and a second approximation of a different kind would
need its own account.
"""
from __future__ import annotations

import numpy as np

BRANCH_FACES = 24          # inscribed polygon for the branch MVA disc
# Twelve faces leave the polygon tighter than the disc by 3.4 per cent,
# which is enough for a schedule sitting near a rating to violate a row
# written at its own operating point -- the clearing is then asked to
# move away from a limit the power flow does not agree is there. At
# twenty-four the gap is 0.9 per cent and no such self-violation occurs
# on either system. It matches the capability set's default for the
# same reason: a step between models should not be a step in
# approximation error.


def _faces(sides: int) -> np.ndarray:
    angle = 2.0 * np.pi * np.arange(sides) / sides
    return np.column_stack((np.cos(angle), np.sin(angle)))


class Injections:
    """Which LP variables move which bus, and where they sat when measured.

    `active[u]` and `reactive[u]` are each a list of (column, coefficient)
    pairs: the clearing's net active injection at a converter bus is a sum of
    discharge, charge and curtailment columns plus a constant, so a single
    injection is not a single variable and cannot be treated as one.
    """

    def __init__(self, size: int):
        self.size = size
        self.active, self.reactive = [], []
        self.p0, self.q0 = [], []

    def add(self, active, reactive, p0: float, q0: float) -> None:
        self.active.append(list(active))
        self.reactive.append(list(reactive))
        self.p0.append(float(p0))
        self.q0.append(float(q0))

    def combine(self, dp: np.ndarray, dq: np.ndarray):
        """Row and constant for sum_u dp_u (P_u - p0_u) + dq_u (Q_u - q0_u)."""
        row = np.zeros(self.size)
        constant = 0.0
        for weight, terms, base in ((dp, self.active, self.p0),
                                    (dq, self.reactive, self.q0)):
            for u, factor in enumerate(weight):
                if factor == 0.0:
                    continue
                for column, coefficient in terms[u]:
                    row[column] += factor * coefficient
                constant -= factor * base[u]
        return row, constant


def bounded(rows, rhs, row: np.ndarray, constant: float, base: float,
            low: float, high: float, slack: float = 0.0) -> None:
    """Impose low <= base + constant + row.x <= high as two <= rows.

    `slack` widens the band. It is not cosmetic: the linearisation carries an
    error of its own, and a band tightened to the last per unit would make the
    clearing chase a limit the power flow does not agree is there. The caller
    sets it from the measured linearisation error, so the tolerance is a
    property of the fit rather than a number someone liked.
    """
    offset = base + constant
    if np.isfinite(high):
        rows.append(row.copy())
        rhs.append(high + slack - offset)
    if np.isfinite(low):
        rows.append(-row)
        rhs.append(-(low - slack) + offset)


def voltage_rows(sens, inj: Injections, low, high, slack: float):
    """Bus voltages inside their declared band."""
    rows, rhs = [], []
    for b in range(len(sens.point.v)):
        row, constant = inj.combine(sens.dv_dp[:, b], sens.dv_dq[:, b])
        bounded(rows, rhs, row, constant, float(sens.point.v[b]),
                float(low[b]), float(high[b]), slack)
    return rows, rhs


def reactive_rows(sens, inj: Injections, low, high, slack: float):
    """Machine reactive output inside its capability.

    This is the row the review asks for and the one that makes the
    linearisation legitimate. With the limits absent from the clearing the
    power flow enforces them itself, by switching a machine from voltage
    control to fixed reactive output, and the sensitivity is discontinuous
    across that switch. Enforced here, the machine never has to switch, and
    the linearisation the rest of these rows rest on stays valid.
    """
    rows, rhs = [], []
    for g in range(len(sens.point.q_gen)):
        row, constant = inj.combine(sens.dq_gen_dp[:, g], sens.dq_gen_dq[:, g])
        bounded(rows, rhs, row, constant, float(sens.point.q_gen[g]),
                float(low[g]), float(high[g]), slack)
    return rows, rhs


def branch_rows(sens, inj: Injections, rating, slack: float,
                sides: int = BRANCH_FACES):
    """Branch apparent power inside its rating, as an inscribed polygon."""
    rows, rhs = [], []
    margin = np.cos(np.pi / sides)
    for b in range(len(sens.point.p_branch)):
        limit = float(rating[b])
        if not np.isfinite(limit) or limit <= 0.0:
            continue
        p_row, p_const = inj.combine(sens.dp_branch_dp[:, b],
                                     sens.dp_branch_dq[:, b])
        q_row, q_const = inj.combine(sens.dq_branch_dp[:, b],
                                     sens.dq_branch_dq[:, b])
        p0 = float(sens.point.p_branch[b]) + p_const
        q0 = float(sens.point.q_branch[b]) + q_const
        for nx, ny in _faces(sides):
            rows.append(nx * p_row + ny * q_row)
            rhs.append(limit * margin + slack - (nx * p0 + ny * q0))
    return rows, rhs


def loss_row(sens, inj: Injections):
    """Active losses as a linear expression, for the energy balance.

    The dc balance carries no losses, which is what leaves the slack machine
    to invent them afterwards -- up to 74.7 MW in the base case. Written here,
    the losses the schedule causes are paid for by the schedule.
    """
    row, constant = inj.combine(sens.dlosses_dp, sens.dlosses_dq)
    return row, float(sens.point.losses) + constant
