#!/usr/bin/env python3
"""Converter capability: the four terms, and their polyhedral form.

The physical limit on a converter is a current, so in per unit the operating
point has to satisfy

    (P + dP)^2 + Q^2  <=  (V * I)^2

for whatever additional active power dP is being asked of it. Three things
follow that a bound on active power alone cannot express.

The limit depends on reactive dispatch and on terminal voltage, because they
draw on the same current. It depends on which current the converter gives up
first when it saturates, which is a design choice with two conventional
settings. And I is not one number: specifications state capability as a level
with a duration, so the current available for a one-second inertial response is
not the current available to hold a reserve for half an hour.

A fourth term comes from the dc side. Short-term overload has to be supplied by
the store, and the power a store can deliver falls as it empties, so the
overload credit is withdrawn below a state-of-charge threshold.

Market clearing runs on linear programmes, so the disc is represented by an
inscribed regular polygon. Inscribed rather than circumscribed, so the
approximation never permits an operating point the current limit forbids; the
cost is a bounded conservatism, reported by ``polygon_conservatism``.
"""
from __future__ import annotations

import numpy as np

F0_HZ = 60.0


def polygon_conservatism(sides: int) -> float:
    """Worst-case radial shortfall of an inscribed regular polygon."""
    return 1.0 - np.cos(np.pi / sides)


def polygon_rows(sides: int) -> np.ndarray:
    """Unit normals of the inscribed polygon, one row per face."""
    angles = 2.0 * np.pi * np.arange(sides) / sides
    return np.column_stack((np.cos(angles), np.sin(angles)))


def current_radius(voltage_pu: float, current_pu: float, rating_mva: float,
                   sides: int) -> float:
    """Right-hand side of a polygon face, in MVA."""
    return voltage_pu * current_pu * rating_mva * np.cos(np.pi / sides)


def soc_availability(soc: np.ndarray, threshold: float) -> np.ndarray:
    """Fraction of the overload credit the store can actually support.

    Discharge power falls as the store empties. A linear ramp to zero below the
    threshold is the simplest form that carries the effect; the threshold is a
    parameter to be swept, not a constant to be believed.
    """
    if threshold <= 0.0:
        return np.ones_like(soc)
    return np.clip(soc / threshold, 0.0, 1.0)


def inertial_power_from_h(h_const_s: np.ndarray, rating_mva: np.ndarray,
                          rocof_hz_s: float, f0_hz: float = F0_HZ) -> np.ndarray:
    """Active power a machine of inertia constant H releases at a given RoCoF.

    H is on the machine's own base, so the rating multiplies here and nowhere
    else. Summing inertia constants without this weight is a common slip.
    """
    return 2.0 * h_const_s * rating_mva * rocof_hz_s / f0_hz


def headroom_closed_form(p_pu, q_pu, v_pu, i_max, priority: str):
    """Instantaneous headroom, per unit of rating, for reference and tests."""
    limit = v_pu * i_max
    if priority == "reactive":
        return np.sqrt(np.maximum(limit**2 - np.asarray(q_pu) ** 2, 0.0)) - p_pu
    if priority == "active":
        return limit - np.asarray(p_pu)
    raise ValueError(f"unknown current priority: {priority}")


if __name__ == "__main__":
    print("inscribed-polygon conservatism")
    for sides in (6, 8, 12, 16, 24, 36, 48):
        print(f"  {sides:3d} faces -> {100 * polygon_conservatism(sides):6.3f} %")
    print()
    print("state-of-charge availability, threshold 0.20")
    soc = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.50])
    print("  " + "  ".join(
        f"{s:.0%}->{a:.2f}" for s, a in zip(soc, soc_availability(soc, 0.20))
    ))
