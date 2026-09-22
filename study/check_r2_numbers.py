#!/usr/bin/env python3
"""Check the headline numbers of the r2 manuscript against the result files.

Each entry names a quantity stated in manuscript/r2, where it comes from, and
the value the text prints. The check fails if any stored value disagrees with
the printed one beyond the printed precision.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

R = pathlib.Path(__file__).resolve().parent / "results"
fails = []


def check(name, printed, actual, tol):
    ok = abs(float(printed) - float(actual)) <= tol
    print(f"{'ok ' if ok else 'BAD'} {name}: printed {printed} actual {actual:.4f}")
    if not ok:
        fails.append(name)


# base-case cost differences (sensitivity reference row)
se = pd.read_csv(R / "sensitivity.csv")
ref = se[se.axis == "reference"].set_index(["case", "share"]).gap_pct
for (c, s), v in {("ieee39", 0.2): 0.96, ("ieee39", 0.4): 0.34,
                  ("rts24", 0.2): 0.68, ("rts24", 0.4): 1.08}.items():
    check(f"delta base {c} {s}", v, ref[(c, s)], 0.005)
for axis, val, col, printed in [("rocof", 0.5, ("ieee39", 0.2), 8.51),
                                ("sync_inertia", 3.0, ("ieee39", 0.2), 4.62),
                                ("storage_hours", 1.0, ("ieee39", 0.2), 2.52),
                                ("offer_ratio", 4.0, ("rts24", 0.4), 3.83),
                                ("box_at_pmax", 1.0, ("rts24", 0.4), 7.94)]:
    row = se[(se.axis == axis) & (np.isclose(se.value, val)) & (se.case == col[0]) & (se.share == col[1])]
    check(f"sens {axis}={val} {col}", printed, float(row.gap_pct.iloc[0]), 0.005)

# sign flip ranges
sf = pd.read_csv(R / "sign_flip.csv")
check("overload max", 1.1, sf[sf.omission == "overload"].error_pct.max(), 0.05)
check("reactive min", -2.6, sf[sf.omission == "reactive"].error_pct.min(), 0.05)
check("voltage min", -2.7, sf[sf.omission == "voltage"].error_pct.min(), 0.05)
check("voltage max", 0.9, sf[sf.omission == "voltage"].error_pct.max(), 0.05)

# rating cross table
rc = pd.read_csv(R / "rating_cross.csv")
check("rating cross all positive delta", 1.0, float((-rc.gap_pct > 0).all()), 0)
check("rating cross min |delta| (S)", 0.26, (-rc[rc.bound == "S"].gap_pct).min(), 0.005)
check("rating cross max |delta| (S)", 5.44, (-rc[rc.bound == "S"].gap_pct).max(), 0.005)
check("rating cross max |delta| (Psch)", 8.25, (-rc[rc.bound == "P_sch"].gap_pct).max(), 0.005)
check("rating cross box price max", 41.0, rc.price_inertia_box.max(), 1e-6)
check("rating cross set price", 3.0, rc.price_inertia_set.max(), 0.005)

# comparison ladder at full dispatch
cl = pd.read_csv(R / "comparison_ladder.csv")
full = cl[np.isclose(cl.dispatch, 1.0)]
for col, lo, hi in [("box", 1.17, 14.87), ("apparent", 1.60, 16.74), ("derated", 0.65, 22.49)]:
    key = [c for c in cl.columns if c.endswith(col) and ("delta" in c or "gap" in c or "pct" in c)]
    if not key:
        continue
    vals = -full[key[0]]
    check(f"ladder {col} min", lo, vals.min(), 0.01)
    check(f"ladder {col} max", hi, vals.max(), 0.01)

# exact cone
ec = pd.read_csv(R / "exact_cone.csv")
check("cone gap m24 max", 0.066, ec.gap_m24.max(), 0.0005)
check("cone gap m96 max", 0.0045, ec.gap_m96.max(), 0.0001)
check("delta_m24 <= delta_cone (as |delta|)", 1.0, float((ec.delta_m24 >= ec.delta_cone - 1e-9).all()), 0)

# placement
pl = pd.read_csv(R / "placement_spread.csv")
pl = pl[pl.placement == "random"]
feas = pl[pl.feasible]
check("placement feasible draws", 605, len(feas), 0)
check("placement all positive", 1.0, float((feas.iloc[:, [c for c in range(len(feas.columns)) if "delta" in feas.columns[c] or "gap" in feas.columns[c]][0]] > 0).all()), 0)

# AC final
ac = pd.read_csv(R / "ac_feasible.csv")
check("ac all pass", 1.0, float(ac.passes.all()), 0)
check("ac loading max", 99.9, ac.loading_max.max(), 0.05)
check("ac balance max", 0.96, ac.balance_mw.max(), 0.005)
check("ac rounds max", 13, ac.rounds.max(), 0)
w = ac.pivot_table(index=["case", "share"], columns="form", values="cost")
d_ac = 100 * (w["box"] - w["current"]) / w["box"]
for key, v in {("ieee39", 0.2): 0.85, ("ieee39", 0.4): 0.07, ("rts24", 0.2): 0.59, ("rts24", 0.4): 1.00}.items():
    check(f"ac-feasible delta {key}", v, d_ac[key], 0.005)
check("ac cost change min", 1.0, ac.cost_change_pct.min(), 0.05)
check("ac cost change max", 2.4, ac.cost_change_pct.max(), 0.05)
box_cur = ac[ac.form == "box"].conv_current_any_form
check("box current max", 1.40, box_cur.max(), 0.005)

# secured outages
so = pd.read_csv(R / "secured_outages.csv")
check("secured all feasible", 1.0, float(so.feasible_secured.all()), 0)
check("secured cost increase max", 0.06, so.cost_increase_pct.max(), 0.005)
check("secured worst loading after max", 100.1, so.screen_worst_secured.max(), 0.05)

# price brackets unique
pb = pd.read_csv(R / "price_brackets.csv")
check("price brackets coincide", 1.0, float(((pb.box_left - pb.box_right).abs() < 1e-6).all() and ((pb.current_left - pb.current_right).abs() < 1e-6).all()), 0)

# corrected face energy
cf = pd.read_csv(R / "corrected_face.csv").set_index(["case", "share"])
for key, b, a in [(("ieee39", 0.2), 86.3, 89.9), (("ieee39", 0.4), 80.7, 83.4), (("rts24", 0.2), 92.8, 93.3), (("rts24", 0.4), 97.7, 97.0)]:
    check(f"energy before {key}", b, 100 * cf.loc[key, "energy_before"], 0.06)
    check(f"energy after {key}", a, 100 * cf.loc[key, "energy_after"], 0.06)
check("v drop max", 12.4, cf.v_drop_pct.max(), 0.06)

# 118
s118 = pd.read_csv(R / "scale_check_118.csv")
check("118 max delta", 0.12, s118.delta_pct.max(), 0.005)
check("118 min delta", -0.68, s118.delta_pct.min(), 0.005)

print("\nFAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
