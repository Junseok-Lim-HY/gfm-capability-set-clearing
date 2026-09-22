#!/usr/bin/env python3
"""How wide are the inertia price brackets, and does 13.7 survive them?

The professor's second essential point is that the paper reports one dual for
a requirement whose subdifferential may be an interval. The dual test already
computes a left and a right difference per row, so the material to answer him
is on disk; what has never been done is to carry those brackets through the
per-period sum that becomes the reported price.

This reads the existing dual_check.csv and reports, per configuration, the
price the paper quotes together with the price that the left and right
brackets give. If the bracket for the bound and the bracket for the capability
set overlap, the ratio between them is not a number the paper can claim.

Nothing is written. This is a look before any decision.
"""
from __future__ import annotations

import collections
import pathlib
import re
import sys

import pandas as pd

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
PERIOD = re.compile(r"_t(\d+)")
HOURS = 3.0


def period_of(name: str) -> int:
    found = PERIOD.search(name)
    return int(found.group(1)) if found else -1


def per_period_sum(rows: pd.DataFrame, column: str) -> float:
    """The paper's rule: sum the duals within a period, then average periods."""
    by_period: dict[int, float] = collections.defaultdict(float)
    for name, value in zip(rows.requirement, rows[column]):
        by_period[period_of(name)] += float(value)
    if not by_period:
        return float("nan")
    return sum(by_period.values()) / len(by_period) / HOURS


def main() -> None:
    frame = pd.read_csv(RESULTS / "dual_check.csv")
    inertia = frame[frame.kind == "inertia"]

    print("행 수", len(inertia))
    degenerate = inertia[inertia.left != inertia.right]
    print(f"좌우 차분이 다른 행: {len(degenerate)} / {len(inertia)}")
    print()

    print(f"{'case':<8}{'share':<7}{'form':<9}"
          f"{'quoted':>9}{'left':>9}{'right':>9}{'width':>9}")
    quoted: dict[tuple, dict] = {}
    for key, rows in inertia.groupby(["case", "share", "form"]):
        one = {c: per_period_sum(rows, c) for c in ("dual", "left", "right")}
        quoted[key] = one
        width = one["right"] - one["left"]
        print(f"{key[0]:<8}{key[1]:<7}{key[2]:<9}"
              f"{one['dual']:>9.3f}{one['left']:>9.3f}"
              f"{one['right']:>9.3f}{width:>9.3f}")

    print()
    print("비율 box / current — 점 추정과, 구간이 주어졌을 때의 최악·최선")
    print(f"{'case':<8}{'share':<7}{'quoted':>9}{'lowest':>9}{'highest':>9}"
          f"   {'구간 겹침':<10}")
    for case in ("ieee39", "rts24"):
        for share in (0.2, 0.4):
            box = quoted.get((case, share, "box"))
            cur = quoted.get((case, share, "current"))
            if not box or not cur:
                continue
            point = box["dual"] / cur["dual"] if cur["dual"] else float("inf")
            lo = box["left"] / cur["right"] if cur["right"] else float("inf")
            hi = box["right"] / cur["left"] if cur["left"] else float("inf")
            overlap = not (box["left"] > cur["right"] or cur["left"] > box["right"])
            print(f"{case:<8}{share:<7}{point:>9.2f}{lo:>9.2f}{hi:>9.2f}"
                  f"   {'겹침' if overlap else '분리됨':<10}")


if __name__ == "__main__":
    sys.exit(main())
