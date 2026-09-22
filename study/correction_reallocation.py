#!/usr/bin/env python3
"""Where the corrected face moves the inertial obligation to.

Applying Eq. (27) to the short-term face costs nothing and leaves the total
scheduled inertia untouched, but it moves the obligation between converters,
and the aggregate delivery rises. Those two facts sitting next to each other
invite an explanation -- that the correction takes duty off the converters that
could not have delivered it -- and an explanation is not a result until it is
measured. This measures it.

For each cell the delivery before correction is put against the change in that
cell's scheduled inertia. If duty really moves off the converters that deliver
least, the two correlate positively: low delivery, negative change.

It does, on the system where the correction is large enough to matter, and
does not on the system where it is not. Both are reported. A mechanism that
is claimed everywhere and holds in half the cases is worse than one stated
with its range.

Writes results/correction_reallocation.csv.
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
sys.path.insert(0, str(ROOT / "study"))

import corrected_face as cf  # noqa: E402
from multiperiod import (MultiPeriod, build_case,  # noqa: E402
                         common_demand_peak)
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
MOVED = 1e-6          # MW, below which a cell counts as not having moved
RESAMPLES = 10000     # bootstrap draws for the correlation interval
SEED = 20260822       # fixed, so the reported interval does not move


def statistics(frame):
    """Pearson, Spearman and a bootstrap interval for the Pearson value.

    Both coefficients, because nothing says the relation between a cell's
    delivery and the change in its obligation should be linear, and the rank
    coefficient does not need it to be. The interval is a percentile bootstrap
    rather than a normal approximation: these samples run from 21 to 82 cells,
    which is not where the approximation belongs.
    """
    if len(frame) < 3 or frame.dh.std() == 0 or frame.energy.std() == 0:
        return np.nan, np.nan, np.nan, np.nan
    x = frame.energy.to_numpy()
    y = frame.dh.to_numpy()
    pearson = float(np.corrcoef(x, y)[0, 1])
    rank_x = pd.Series(x).rank().to_numpy()
    rank_y = pd.Series(y).rank().to_numpy()
    spearman = float(np.corrcoef(rank_x, rank_y)[0, 1])
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(RESAMPLES):
        take = rng.integers(0, len(x), len(x))
        xs, ys = x[take], y[take]
        if xs.std() == 0 or ys.std() == 0:
            continue
        draws.append(np.corrcoef(xs, ys)[0, 1])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return pearson, spearman, float(lo), float(hi)


def main() -> None:
    settings = Capability(i_short_term=1.5, priority="reactive",
                          soc_threshold=0.20)
    rows = []
    for name, share in itertools.product(cf.CASES, cf.SHARES):
        base = build_case(name, converter_share=share,
                          demand_peak_mw=common_demand_peak(name))
        case = dataclasses.replace(
            base, conv_injection_pu=np.full(len(base.demand_mw), cf.DISPATCH))
        plain = MultiPeriod(case, settings, "current").solve()
        reactance = cf.thevenin(case, plain, settings)
        corrected = cf.event_voltage(case, plain, reactance, settings)
        fixed = dataclasses.replace(case, conv_voltage_short_pu=corrected)
        after = MultiPeriod(fixed, settings, "current").solve()

        cells = []
        for t, i in itertools.product(range(len(case.demand_mw)),
                                      range(len(case.conv_rating_mva))):
            h0, h1 = float(plain["h"][t, i]), float(after["h"][t, i])
            if h0 <= 1e-9 and h1 <= 1e-9:
                continue
            got = cf.delivered(case, plain, reactance, settings, t, i)
            if not np.isfinite(got.get("energy_ratio", np.nan)):
                continue
            cells.append(dict(energy=got["energy_ratio"], dh=h1 - h0))
        frame = pd.DataFrame(cells)
        down = frame[frame.dh < -MOVED]
        up = frame[frame.dh > MOVED]
        still = frame[np.abs(frame.dh) <= MOVED]
        correlation, spearman, lo, hi = statistics(frame)

        rows.append(dict(
            case=name, share=share, cells=len(frame),
            spearman=spearman, ci_lo=lo, ci_hi=hi,
            v_drop_pct=float(100.0 * (case.conv_voltage_pu - corrected).max()
                             / case.conv_voltage_pu.max()),
            v_short_min=float(corrected.min()),
            v_short_max=float(corrected.max()),
            x_max=float(reactance.max()),
            correlation=correlation,
            n_down=len(down), n_up=len(up), n_still=len(still),
            energy_down=float(down.energy.median()) if len(down) else np.nan,
            energy_up=float(up.energy.median()) if len(up) else np.nan,
            energy_still=float(still.energy.median()) if len(still) else np.nan))
        row = rows[-1]
        print(f"{name} {share:.0%}: 단기전압 강하 {row['v_drop_pct']:5.2f}%  "
              f"상관 {correlation:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}] n={len(frame)} "
              f"rho {spearman:+.3f}  "
              f"H 줄어든 칸 {row['n_down']:3d}(이행 중앙 "
              f"{row['energy_down']:.3f})  "
              f"늘어난 칸 {row['n_up']:3d}({row['energy_up']:.3f})  "
              f"안 움직인 칸 {row['n_still']:3d}")

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "correction_reallocation.csv", index=False)
    print(f"\nwrote {RESULTS / 'correction_reallocation.csv'}")


if __name__ == "__main__":
    sys.exit(main())
