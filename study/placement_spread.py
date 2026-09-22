#!/usr/bin/env python3
"""Does the answer depend on which machines are the converters?

Everywhere else in this paper the converter fleet is chosen by one rule: take
the smallest units first until they reach the target share of installed
capacity. The rule was picked to avoid choosing the units that flatter the
result, but picking a rule is still picking, and a difference reported from one
placement is a difference reported from one draw. This asks what the spread
around that draw looks like.

Placement is varied and nothing else. Each draw walks the fleet in a random
order and takes units until the same capacity share is reached, so the share is
matched to within one unit's rating and the comparison is between placements
rather than between fleet sizes. Draws that land outside a narrow band are
rejected rather than adjusted, and how many were rejected is reported: silently
keeping them would let fleet size back into the experiment through the door
marked placement.

The band is centred on the share the smallest-first rule actually reaches, not
on the nominal target. The rule takes units until the target is passed, so the
last unit overshoots, and on IEEE 39-bus at a nominal \SI{20}{\percent} the
baseline sits at \num{0.2254}. Measuring draws against the nominal figure
would apply a test the baseline itself fails, and would reject the draws
closest to it.

Demand is held at the peak the baseline case uses, so the systems being cleared
differ only in which machines carry the capability set.

Two things are read off. Whether delta keeps its sign across placements is the
question that matters, because the paper's claim is about a direction. Where
the smallest-first baseline sits inside the distribution is the second, because
a baseline at the extreme would mean the rule was doing the work after all.

Writes results/placement_spread.csv.
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

from multiperiod import (MultiPeriod, build_case,  # noqa: E402
                         common_demand_peak, fleet, load_network)
from prices import price  # noqa: E402
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
CASES = ("ieee39", "rts24")
SHARES = (0.2, 0.4)
DISPATCH = 0.8
DRAWS = 400
TOLERANCE = 0.02          # of installed capacity, either side of the anchor
SEED = 20260818


def draw_mask(rating: np.ndarray, share: float, rng) -> np.ndarray:
    """One random placement reaching the target share of installed capacity.

    Same stopping rule as the baseline, applied to a random order instead of
    ascending rating. That keeps the only difference between a draw and the
    baseline the identity of the machines.
    """
    total = float(rating.sum())
    order = rng.permutation(len(rating))
    mask = np.zeros(len(rating), dtype=bool)
    taken = 0.0
    for k in order:
        if taken >= share * total:
            break
        mask[k] = True
        taken += float(rating[k])
    return mask


def delta_of(case, settings_kwargs) -> dict:
    """Cost and inertia price under both representations, and Eq. (20)."""
    out = {}
    for form in ("box", "current"):
        settings = Capability(**settings_kwargs)
        out[form] = MultiPeriod(case, settings, form).solve()
    if not all(out[f]["feasible"] for f in out):
        return dict(feasible=False)
    return dict(feasible=True,
                cost_box=out["box"]["cost"], cost_set=out["current"]["cost"],
                price_box=price(out["box"]["prices"], "inertia"),
                price_set=price(out["current"]["prices"], "inertia"),
                delta_pct=100.0 * (out["box"]["cost"] - out["current"]["cost"])
                / out["box"]["cost"])


def main() -> None:
    kwargs = dict(i_short_term=1.5, priority="reactive", soc_threshold=0.20)
    peaks = {n: common_demand_peak(n, SHARES) for n in CASES}
    rng = np.random.default_rng(SEED)
    rows = []

    for name, share in itertools.product(CASES, SHARES):
        rating = fleet(load_network(name)).rating.to_numpy(dtype=float)
        total = float(rating.sum())

        def cleared(mask, label, index):
            case = build_case(name, converter_share=share,
                              demand_peak_mw=peaks[name],
                              converter_mask=mask)
            loaded = dataclasses.replace(
                case,
                conv_injection_pu=np.full(len(case.demand_mw), DISPATCH))
            row = dict(case=name, share=share, placement=label, draw=index,
                       n_converters=int(np.sum(mask)) if mask is not None
                       else len(case.conv_rating_mva),
                       actual_share=float(case.conv_rating_mva.sum() / total))
            row.update(delta_of(loaded, kwargs))
            return row

        base = cleared(None, "smallest-first", -1)
        rows.append(base)
        anchor = base["actual_share"]
        print(f"\n=== {name} {share:.0%}  기준(작은 것부터): "
              f"{base['n_converters']}기, "
              f"실제 몫 {base['actual_share']:.4f}, "
              f"delta {base.get('delta_pct', float('nan')):+.4f}%")

        kept, rejected = 0, 0
        for index in range(DRAWS):
            mask = draw_mask(rating, share, rng)
            hit = float(rating[mask].sum() / total)
            if abs(hit - anchor) > TOLERANCE:
                rejected += 1
                continue
            row = cleared(mask, "random", index)
            rows.append(row)
            kept += 1
            mark = (f"{row['delta_pct']:+.4f}%" if row.get("feasible")
                    else "해 없음")
            print(f"  draw {index:2d}: {row['n_converters']:2d}기  "
                  f"몫 {hit:.4f}  {mark}")
        print(f"  채택 {kept}, 몫이 기준({anchor:.4f})에서 {TOLERANCE:.0%} "
              f"넘게 벗어나 버린 draw {rejected}")

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "placement_spread.csv", index=False)

    print("\n=== 배치에 따른 delta 의 흩어짐")
    for name, share in itertools.product(CASES, SHARES):
        block = frame[(frame.case == name) & (frame.share == share)]
        good = block[block.feasible.fillna(False)]
        drawn = good[good.placement == "random"]
        anchor = good[good.placement == "smallest-first"]
        if not len(drawn) or not len(anchor):
            print(f"{name} {share:.0%}: 비교할 draw 가 없습니다")
            continue
        d = drawn.delta_pct
        base = float(anchor.delta_pct.iloc[0])
        below = int((d < base).sum())
        print(f"{name} {share:.0%}: n={len(d)}  "
              f"중앙값 {d.median():+.4f}%  "
              f"범위 {d.min():+.4f} ~ {d.max():+.4f}%  "
              f"기준 {base:+.4f}% (draw 중 {below}/{len(d)} 개가 그보다 작음)  "
              f"부호 양수 {int((d > 0).sum())}/{len(d)}  "
              f"해 없음 {int((~block.feasible.fillna(False)).sum())}")

    print(f"\nwrote {RESULTS / 'placement_spread.csv'}")


if __name__ == "__main__":
    sys.exit(main())
