#!/usr/bin/env python3
"""Are the reported prices actually the derivatives they are claimed to be?

The paper prices every service as the multiplier of the requirement it clears.
That is only worth saying if the multiplier really is the rate at which the
cleared cost moves with the requirement, and the way to know is to move the
requirement and look. An earlier version of this study asserted the check had
been done; it had been done on a single-interval prototype that no longer
produces any published number, which is not the same thing.

The subtlety is degeneracy. A linear program's value function is convex and
piecewise linear in the right-hand side, so at a breakpoint it has no
derivative, only a subdifferential, and the simplex returns one subgradient
from that interval. Testing a dual against a single one-sided difference will
therefore report false failures wherever a requirement sits exactly on a
breakpoint --- which is common, because that is what a binding constraint at a
vertex means.

So the test is the correct one for a linear program: compute the left and
right differences and require

    lambda_left - tol  <=  lambda  <=  lambda_right + tol,

which is the definition of a subgradient. A dual outside that bracket is
wrong; a dual inside it is the derivative wherever one exists and a valid
subgradient where one does not. The bracket width is reported so a reader can
see which requirements are degenerate and which are not.
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "model"))

from multiperiod import (MultiPeriod, build_case,  # noqa: E402
                         common_demand_peak)
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"

STEP = 1.0        # MW; small against the requirements, large against solver noise
TOLERANCE = 1e-4  # relative, against the size of the bracket or the dual


def kind_of(name: str) -> str:
    return name.split("_t", 1)[0]


def check(case, settings, form: str) -> list[dict]:
    """Bracket every requirement's dual between its one-sided differences."""
    model = MultiPeriod(case, settings, form)
    base = model.solve()
    if not base["feasible"]:
        return []

    rows = []
    for name, dual in base["prices"].items():
        up = model.solve(perturb=(name, STEP))
        down = model.solve(perturb=(name, -STEP))
        if not (up["feasible"] and down["feasible"]):
            # A requirement that cannot be raised is at a feasibility edge and
            # its dual is not a two-sided derivative. Recorded, not silently
            # dropped, because the count matters when reporting coverage.
            rows.append(dict(requirement=name, kind=kind_of(name), dual=dual,
                             left=np.nan, right=np.nan, slack=np.nan,
                             ok=False, note="perturbed problem infeasible"))
            continue
        right = (up["cost"] - base["cost"]) / STEP
        left = (base["cost"] - down["cost"]) / STEP
        # Convexity puts left below right; solver noise can invert them by a
        # hair at a flat requirement, so the bracket is widened by the tolerance
        # rather than trusted to be ordered.
        scale = max(1.0, abs(dual), abs(left), abs(right))
        low, high = min(left, right), max(left, right)
        ok = (low - TOLERANCE * scale) <= dual <= (high + TOLERANCE * scale)
        rows.append(dict(requirement=name, kind=kind_of(name), dual=dual,
                         left=left, right=right, slack=high - low, ok=ok,
                         note="" if ok else "dual outside the subdifferential"))
    return rows


def main() -> int:
    settings = Capability(i_short_term=1.5, priority="reactive",
                          soc_threshold=0.20)
    rows = []
    for name in ("ieee39", "rts24"):
        peak = common_demand_peak(name)
        for share in (0.2, 0.4):
            base = build_case(name, converter_share=share,
                              demand_peak_mw=peak)
            case = dataclasses.replace(
                base, conv_injection_pu=np.full(len(base.demand_mw), 0.8))
            for form in ("box", "current"):
                for row in check(case, settings, form):
                    rows.append(dict(case=name, share=share, form=form, **row))

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "dual_check.csv", index=False)

    print("finite-difference check of every requirement dual, step "
          f"{STEP} MW\n")
    summary = frame.groupby("kind").agg(
        requirements=("ok", "size"),
        passed=("ok", "sum"),
        degenerate=("slack", lambda s: int((s > 1e-6).sum())),
        worst_bracket=("slack", "max"),
    )
    print(summary.to_string(float_format=lambda v: f"{v:10.4f}"))

    bad = frame[~frame.ok]
    print(f"\n{len(frame) - len(bad)} of {len(frame)} duals lie in the "
          f"subdifferential of the value function.")
    if len(bad):
        print("\nfailures:")
        print(bad[["case", "share", "form", "requirement", "dual",
                   "left", "right", "note"]]
              .to_string(index=False, float_format=lambda v: f"{v:10.4f}"))
    print(f"\nwrote {RESULTS / 'dual_check.csv'} ({len(frame)} rows)")
    print("PASS" if not len(bad) else f"FAIL ({len(bad)})")
    return 1 if len(bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
