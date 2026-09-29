#!/usr/bin/env python3
"""Regenerate every result file the manuscript rests on, in dependency order.

    python reproduce.py --quick   # tests and the number check against the shipped csv files (minutes)
    python reproduce.py           # full regeneration (several hours on a desktop; needs ~8 GB RAM)

Run the scripts one at a time: the constraint matrices of the AC-feasible
clearing are large, and two of them side by side can exhaust memory.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
PYTHON = sys.executable

# (script, quick?, what it establishes)
STEPS = [
    ("study/test_reduction.py", True, "with the three omissions removed the two formulations coincide"),
    ("study/test_sign_condition.py", True, "the Proposition against the exact disc"),
    ("study/test_current_safety.py", True, "cleared schedules respect the continuous current limit"),
    ("study/test_duals.py", False, "reported prices are subgradients of the value function"),
    ("study/run_case_study.py", False, "main sweeps (case_study.csv)"),
    ("study/run_sign_flip.py", False, "each omission with the other two removed (sign_flip.csv)"),
    ("study/run_sensitivity.py", False, "one-at-a-time parameter sweep (sensitivity.csv)"),
    ("study/run_day_schedule.py", False, "one day, period by period (day_schedule.csv)"),
    ("study/rating_cross_table.py", False, "rating convention x comparison bound (rating_cross.csv)"),
    ("study/comparison_ladder.py", False, "active-power bound, apparent-power, derating (comparison_ladder.csv)"),
    ("study/exact_cone.py", False, "polygon against the exact second-order cone (exact_cone.csv)"),
    ("study/placement_spread.py", False, "randomised converter placements (placement_spread.csv)"),
    ("study/price_bracket_dispatch.py", False, "left and right limits of the inertia price (price_brackets.csv)"),
    ("study/scale_check_118.py", False, "IEEE 118-bus check (scale_check_118.csv)"),
    ("study/ac_feasible_clearing.py", False, "successive linearisation to ac-feasible schedules (ac_feasible.csv)"),
    ("study/secured_outages.py", False, "post-contingency deliverability constraint (secured_outages.csv)"),
    ("study/post_contingency.py", False, "post-contingency screen of the lossless schedules (post_contingency.csv)"),
    ("study/dynamic_check.py", False, "RMS verification, 360 runs (dynamic_check.csv, dynamic_traces.csv)"),
    ("study/corrected_face.py", False, "short-term face corrected to the event voltage (corrected_face.csv)"),
    ("study/correction_reallocation.py", False, "reallocation under the corrected face"),
    ("manuscript/figures/make_figures.py", False, "Figs. 2-6 and 9 from the csv files"),
    ("manuscript/figures/fig_placement_sensitivity.py", False, "Figs. 7 and 8"),
    ("manuscript/figures/fig1_overview_r2.py", False, "Fig. 1"),
    ("study/check_r2_numbers.py", True, "headline numbers of the manuscript against the csv files"),
    ("study/audit_reruns_260929.py", False, "audit re-runs: repair effect, corrected event voltage, secured fraction to 1e-3"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="tests and the number check only")
    args = parser.parse_args()
    failed = []
    for script, quick, what in STEPS:
        if args.quick and not quick:
            continue
        print(f"\n=== {script}: {what}", flush=True)
        t0 = time.time()
        code = subprocess.call([PYTHON, str(HERE / script)], cwd=HERE)
        print(f"    exit {code} after {time.time() - t0:.0f} s", flush=True)
        if code:
            failed.append(script)
    print("\nFAILED:" if failed else "\nall steps passed", *failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
