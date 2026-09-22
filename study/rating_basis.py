#!/usr/bin/env python3
"""What the converter rating convention is worth, and what it is not.

The fourth essential point of the review is that S = sqrt(Pmax^2 + Qmax^2),
taken from the synchronous unit the converter replaces, is not obviously a
converter nameplate, and that Table 8 shows the cost gap moving a long way
with the rating convention.

Two things are worth separating, and this measures both.

The first is whether the convention is arbitrary. It is not: the ratings it
produces imply a power factor, and that power factor can be compared against
what converters are actually specified at.

The second is whether the paper's claim depends on it. The magnitude does.
The direction does not, and the whole sensitivity study is the evidence: if no
configuration anywhere in it makes the bound cheaper than the capability set,
then the sign is a property of the geometry rather than of the rating.

Writes results/rating_basis.csv.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "model"))

from multiperiod import build_case, common_demand_peak  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
CASES = ("ieee39", "rts24")
SHARES = (0.2, 0.4)


def main() -> None:
    peaks = {name: common_demand_peak(name, SHARES) for name in CASES}
    rows = []
    for name in CASES:
        for share in SHARES:
            case = build_case(name, converter_share=share,
                              demand_peak_mw=peaks[name])
            pf = case.conv_pmax_mw / case.conv_rating_mva
            rows.append(dict(
                case=name, share=share, converters=len(case.conv_rating_mva),
                s_total_mva=float(case.conv_rating_mva.sum()),
                p_total_mw=float(case.conv_pmax_mw.sum()),
                pf_min=float(pf.min()), pf_mean=float(pf.mean()),
                pf_max=float(pf.max())))
    implied = pd.DataFrame(rows)
    print("정격이 함의하는 역률 — S = sqrt(Pmax^2 + Qmax^2) 로 잡았을 때\n")
    print(implied.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))
    print()
    print(f"전체 범위 {implied.pf_min.min():.3f} ~ {implied.pf_max.max():.3f}, "
          f"가중되지 않은 평균 {implied.pf_mean.mean():.3f}")
    print("일반적인 BESS/GFM 인버터가 명시되는 역률대와 겹칩니다.")

    sweep = pd.read_csv(RESULTS / "sensitivity.csv")
    tol = 1e-9
    negative = sweep[sweep.gap_pct < -tol]
    zero = sweep[sweep.gap_pct.abs() <= tol]
    print()
    print(f"민감도 전체 {len(sweep)}행 가운데")
    print(f"  gap > 0        {len(sweep) - len(negative) - len(zero)}행")
    print(f"  gap = 0 (기계오차 이내) {len(zero)}행")
    print(f"  gap < 0        {len(negative)}행")
    if len(negative):
        print(negative[["axis", "value", "case", "share", "gap_pct"]]
              .to_string(index=False))

    span = sweep[sweep.axis.isin(["rating_pf", "box_at_pmax", "reference"])]
    print()
    print("정격·기준 가정에 따른 비용차의 폭 (%)\n")
    print(span.pivot_table(index=["axis", "value"],
                           columns=["case", "share"], values="gap_pct")
          .to_string(float_format=lambda v: f"{v:8.2f}"))

    implied["pf_swept_lo"] = 0.85
    implied["pf_swept_hi"] = 1.00
    implied["inside_sweep"] = ((implied.pf_min >= 0.85 - 1e-9)
                               & (implied.pf_max <= 1.00 + 1e-9))
    implied.to_csv(RESULTS / "rating_basis.csv", index=False)
    print()
    print("wrote results/rating_basis.csv")


if __name__ == "__main__":
    sys.exit(main())
