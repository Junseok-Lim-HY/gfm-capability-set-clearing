#!/usr/bin/env python3
r"""The condition for the sign of the active-power bound's error, checked.

At a sustained operating point that both representations admit, the two differ
only in how much inertial power they leave room for. The bound leaves

    H_ap  = Pbar - (P + R),

and the capability set, with the overload band admitted in proportion a,

    H_set = sqrt(L^2 - Qt^2) - (P + R),   L = V (Ic + a (Is - Ic)) S,

so the sustained term cancels and

    H_set - H_ap = sqrt(L^2 - Qt^2) - Pbar.

The sign is therefore piecewise in L against Pbar, not a single threshold:

    L >  Pbar :  exact at |Qt| = sqrt(L^2 - Pbar^2), the bound conservative
                 below it and optimistic above it;
    L == Pbar :  exact only at Qt = 0, optimistic otherwise;
    L <  Pbar :  optimistic for every admissible Qt.

An earlier version of this test wrote the threshold as
S sqrt(max{(V Is)^2 - 1, 0}) and declared equality wherever it was attained.
That is wrong in the third region: clamping the threshold to zero does not make
the two representations agree there, it only hides that the bound is
overstating. At V Is = 0.9 and Qt = 0 the bound overstates by 0.1 S while the
clamped threshold claims exactness.

Two things are checked here, separately, because they can fail separately:
the sign rule against the exact circle, and the polyhedral set that the
clearing actually solves against that same circle. The second is where the
inscribed polygon shows up, and it is checked by solving the face constraints
rather than by substituting a circle of the inscribed radius, which is what
the earlier version did.
"""
from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "model"))

import capability as cap  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

RATING = 100.0          # S
SUSTAINED = 60.0        # P + R, which cancels and so must not matter
SIDES = 192


def limit(voltage: float, i_cont: float, i_short: float, available: float,
          rating: float) -> float:
    """The radius L the short-term constraint allows at availability a."""
    return voltage * (i_cont + available * (i_short - i_cont)) * rating


def headroom_exact(L: float, reactive: float, bound: float) -> float:
    """H_set - H_ap on the exact circle, negative where the bound overstates."""
    if abs(reactive) >= L:
        return -np.inf
    return float(np.sqrt(L ** 2 - reactive ** 2) - bound)


TOL = 1e-6   # in MW on a 100 MVA rating, so a millionth of per unit


def predicted(L: float, reactive: float, bound: float) -> str:
    if L > bound + TOL:
        crossing = np.sqrt(L ** 2 - bound ** 2)
        if abs(reactive) < crossing - TOL:
            return "conservative"
        if abs(reactive) > crossing + TOL:
            return "optimistic"
        return "exact"
    if abs(L - bound) <= TOL:
        return "exact" if abs(reactive) <= TOL else "optimistic"
    return "optimistic"


def observed(value: float) -> str:
    if value > TOL:
        return "conservative"
    if value < -TOL:
        return "optimistic"
    return "exact"


def polygon_active(reactive: float, radius: float, sides: int) -> float:
    """Largest active coordinate the inscribed polygon admits at this Q.

    Solves the face constraints, which is what the clearing does. Substituting
    a circle of the inscribed radius is a different set and agrees only on the
    axes.
    """
    best = np.inf
    for nx, ny in cap.polygon_rows(sides):
        if nx > 1e-12:
            best = min(best, (radius - reactive * ny) / nx)
    return float(best)


def sign_rule() -> pd.DataFrame:
    """Does the piecewise rule call the sign correctly on the exact circle?"""
    rows = []
    voltages = (0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.08)
    shorts = (1.0, 1.1, 1.2, 1.35, 1.5, 1.8)
    availables = (0.0, 0.5, 1.0)
    bounds = (RATING, 0.9 * RATING)
    for v, i_s, a, bound in itertools.product(voltages, shorts, availables,
                                              bounds):
        L = limit(v, 1.0, i_s, a, RATING)
        crossing = np.sqrt(max(L ** 2 - bound ** 2, 0.0))
        probes = [0.0, 0.25 * L, 0.5 * L, 0.75 * L, 0.95 * L]
        if crossing > 0.0:
            probes += [crossing, 0.5 * crossing, min(1.05 * crossing, 0.99 * L)]
        for reactive in sorted(set(probes)):
            value = headroom_exact(L, reactive, bound)
            rows.append(dict(voltage=v, i_short=i_s, available=a, bound=bound,
                             limit=L, reactive=reactive, difference=value,
                             predicted=predicted(L, reactive, bound),
                             observed=observed(value)))
    return pd.DataFrame(rows)


def polygon_gap() -> pd.DataFrame:
    """How far the inscribed polygon sits inside the circle it approximates."""
    rows = []
    for sides in (8, 12, 24, 48, 96, 192):
        radius = cap.current_radius(1.0, 1.0, RATING, sides)
        worst = 0.0
        for fraction in np.linspace(0.0, 0.95, 20):
            reactive = fraction * RATING
            exact = np.sqrt(max(RATING ** 2 - reactive ** 2, 0.0))
            got = polygon_active(reactive, radius, sides)
            worst = max(worst, exact - got)
        rows.append(dict(sides=sides, radius=radius,
                         worst_shortfall=worst,
                         worst_shortfall_pct=100.0 * worst / RATING,
                         radial_bound_pct=100.0 * (1.0 - np.cos(np.pi / sides))))
    return pd.DataFrame(rows)


def main() -> int:
    rule = sign_rule()
    rule.to_csv(RESULTS / "sign_condition.csv", index=False)
    wrong = rule[rule.predicted != rule.observed]

    print("1. the piecewise sign rule against the exact circle")
    print(f"   {len(rule)} probes over voltage, short-term rating, band "
          f"availability, active-power bound and reactive dispatch")
    print(f"   sign called correctly at {len(rule) - len(wrong)} of {len(rule)}")
    if not wrong.empty:
        print(wrong.head(12).to_string(index=False))

    print("\n   the region the earlier threshold got wrong, L < Pbar:")
    region = rule[(rule.limit < rule.bound) & (rule.reactive == 0.0)].head(4)
    for _, row in region.iterrows():
        print(f"     V={row.voltage:.2f} Is={row.i_short:.2f} a={row.available:.1f}"
              f" Pbar={row.bound:.0f}: L={row.limit:6.2f}, Q=0, "
              f"difference {row.difference:+7.2f} -> {row.observed}")

    gap = polygon_gap()
    gap.to_csv(RESULTS / "polygon_gap.csv", index=False)
    print("\n2. the inscribed polygon against the circle, by solving its faces")
    print(gap.to_string(index=False,
                        float_format=lambda v: f"{v:10.4f}"))
    print("\n   worst shortfall is measured over reactive dispatches from 0 to"
          "\n   0.95 pu, so it is the largest error the clearing can meet, not"
          "\n   the radial error on the axis.")

    print("\nPASS" if wrong.empty else f"\nFAIL ({len(wrong)})")
    return 1 if not wrong.empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
