#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SEGAN r2 numbers check, version 2 (2026-09-29).

Differences from check_r2_numbers.py:
  * every cost difference is recomputed from the cost columns with Eq. (22),
    delta = 100 (C_cmp - C_set) / C_cmp, never read from a gap column;
  * the printed strings are read from the compiled PDFs (main + supplementary)
    and compared, so a stale sentence is caught even when the table is right;
  * retired wordings and values are listed as must-NOT-contain checks.

    python check_r2_numbers_v2.py <Manuscript.pdf> <Supplementary.pdf>
"""
from __future__ import annotations

import io
import pathlib
import re
import sys

import pymupdf as fitz
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = pathlib.Path(__file__).resolve().parent
R = HERE / "results"
fails, notes = [], []


def pdf_text(path):
    t = " ".join(p.get_text() for p in fitz.open(path))
    return re.sub(r"\s+%", "%", re.sub(r"\s+", " ", t)).replace("−", "-").replace("–", "-")


MAIN = pdf_text(sys.argv[1]) if len(sys.argv) > 1 else ""
SUPP = pdf_text(sys.argv[2]) if len(sys.argv) > 2 else ""


def check(name, printed, actual, tol):
    ok = abs(float(printed) - float(actual)) <= tol
    print(f"{'ok ' if ok else 'BAD'} {name}: printed {printed} actual {actual:.4f}")
    if not ok:
        fails.append(name)


def expect(name, s, where=None, present=True):
    text = MAIN if where == "main" else SUPP if where == "supp" else MAIN + " " + SUPP
    if not text:
        notes.append(f"skip (no pdf): {name}")
        return
    dehyph = re.sub(r"(?<=\w)- (?=\w)", "", text)          # "per- centage" -> "percentage"
    joined = text.replace("- ", "-")                          # "0.53- 8.25" -> "0.53-8.25"
    found = (s in text) or (s in dehyph) or (s in joined)
    ok = found == present
    print(f"{'ok ' if ok else 'BAD'} {name}: {'expects' if present else 'must NOT contain'} '{s}'")
    if not ok:
        fails.append(name)


def delta(c_cmp, c_set):  # Eq. (22)
    return 100.0 * (c_cmp - c_set) / c_cmp


# ------------------------------------------------------------------ Table 4 / Sec. 5.2
rc = pd.read_csv(R / "rating_cross.csv")
rc["delta"] = delta(rc.cost_box, rc.cost_set)
check("T4 all positive", 1.0, float((rc.delta > 0).all()), 0)
check("T4 distinct cells = 28 (P/S=1 row shared by both bounds)", 28,
      len(rc.drop_duplicates(subset=["case", "share", "p_over_s", "delta"])), 0)
s_rows, p_rows = rc[rc.bound == "S"], rc[rc.bound == "P_sch"]
check("T4 min delta, Pbar=S", 0.26, s_rows.delta.min(), 0.005)
check("T4 max delta, Pbar=S", 5.44, s_rows.delta.max(), 0.005)
check("T4 min delta, Pbar=Psch (incl. P/S=1 row) = 0.53", 0.53, p_rows.delta.min(), 0.005)
check("T4 max delta, Pbar=Psch", 8.25, p_rows.delta.max(), 0.005)
expect("5.2 Psch range must not say 0.62-8.25", "from 0.62-8.25%", "main", present=False)
expect("5.2 Psch range 0.53-8.25", "0.53-8.25%", "main")
for (c, s, ps, b), v in {("ieee39", 0.2, 1.0, "S"): 1.02, ("ieee39", 0.4, 0.9, "P_sch"): 4.49,
                         ("rts24", 0.4, 0.95, "P_sch"): 6.89, ("rts24", 0.4, "case", "S"): 1.08}.items():
    row = rc[(rc.case == c) & (rc.share == s) & (rc.p_over_s.astype(str) == str(ps)) & (rc.bound == b)]
    check(f"T4 cell {c} {s} {ps} {b}", v, float(row.delta.iloc[0]), 0.005)
check("set price within 0.1% of offer (rating cross, 3.003)", 3.003, rc.price_inertia_set.max(), 0.0005)
ref = rc[rc.p_over_s.astype(str) == "case"]
check("Psch reference price min 25.6", 25.6, ref[ref.bound == "P_sch"].price_inertia_box.min(), 0.05)
check("Psch reference price max 41.0", 41.0, ref[ref.bound == "P_sch"].price_inertia_box.max(), 0.05)

# ------------------------------------------------------------------ Table 5 / Sec. 5.3 (ladder)
cl = pd.read_csv(R / "comparison_ladder.csv")
w = cl.pivot_table(index=["case", "share", "dispatch"], columns="form", values="cost")
for f, lo, hi in [("box", 1.17, 14.87), ("apparent", 1.60, 16.74), ("derated", 0.65, 22.49)]:
    d = delta(w[f], w["current"])
    full = d.xs(1.0, level="dispatch")
    check(f"ladder {f} min @e=1", lo, full.min(), 0.005)
    check(f"ladder {f} max @e=1", hi, full.max(), 0.005)
    check(f"ladder {f} never negative beyond 1e-3", 1.0, float((d > -1e-3).all()), 0)
check("box/set cost ratio ieee39 0.4 e=1", 1.175, (w["box"] / w["current"]).loc[("ieee39", 0.4, 1.0)], 0.0005)
check("box/set cost ratio rts24 0.4 e=1", 1.020, (w["box"] / w["current"]).loc[("rts24", 0.4, 1.0)], 0.0005)
pr = cl.pivot_table(index=["case", "share", "dispatch"], columns="form", values="price_inertia")
check("ratio e=0.8 min 2.3", 2.25, (pr["box"] / pr["current"]).xs(0.8, level="dispatch").min(), 0.01)
check("ratio e=1.0 min 7.8", 7.75, (pr["box"] / pr["current"]).xs(1.0, level="dispatch").min(), 0.01)
check("ratio e=1.0 max 13.7", 13.67, (pr["box"] / pr["current"]).xs(1.0, level="dispatch").max(), 0.01)
cs = pd.read_csv(R / "current_safety.csv")
i_box = cs[(cs.form == "box") & (cs.dispatch == 0.8)].i_after.max()
check("lossless bound current max (rts24) 1.38", 1.38, i_box, 0.005)
expect("5.3 must not cite Table 6 for 1.38", "1.38 pu of continuous current on RTS-24 (Table 6", "main", present=False)

# ------------------------------------------------------------------ sensitivity / Fig. 7 / S6
se = pd.read_csv(R / "sensitivity.csv")
se["delta"] = delta(se.cost_box, se.cost_set)
ref = se[se.axis == "reference"].set_index(["case", "share"]).delta
for (c, s), v in {("ieee39", 0.2): 0.96, ("ieee39", 0.4): 0.34, ("rts24", 0.2): 0.68, ("rts24", 0.4): 1.08}.items():
    check(f"base delta {c} {s}", v, ref[(c, s)], 0.005)
for axis, val, key, printed in [("rocof", 0.5, ("ieee39", 0.2), 8.51), ("sync_inertia", 3.0, ("ieee39", 0.2), 4.62),
                                ("storage_hours", 1.0, ("ieee39", 0.2), 2.52), ("offer_ratio", 4.0, ("rts24", 0.4), 3.83),
                                ("box_at_pmax", 1.0, ("rts24", 0.4), 7.94), ("polygon", 8.0, ("rts24", 0.2), 0.27),
                                ("rating_pf", 0.85, ("rts24", 0.4), 0.28), ("voltage_shift", -0.05, ("rts24", 0.2), 0.39)]:
    row = se[(se.axis == axis) & np.isclose(se.value, val) & (se.case == key[0]) & (se.share == key[1])]
    check(f"sens {axis}={val} {key}", printed, float(row.delta.iloc[0]), 0.005)
rest = se[se.axis.isin(["polygon", "cycling_cost", "reserve_overlap", "voltage_shift"])].copy()
rest["shift"] = rest.apply(lambda r: r.delta - ref[(r.case, r.share)], axis=1)
check("remaining-parameter shift max is 0.41 pp (text says <0.4)", 0.41, rest["shift"].abs().max(), 0.005)
expect("5.3 remaining parameters wording", "less than 0.4 percentage points", "main", present=False)
expect("S1 gate cost movement < 0.01 pp", "less than 0.01 percentage", "supp")

# ------------------------------------------------------------------ Fig. 4 / Sec. 5.1
sf = pd.read_csv(R / "sign_flip.csv")
sf["delta"] = delta(sf.cost_box, sf.cost_circle)
check("overload max +1.1", 1.1, sf[sf.omission == "overload"].delta.max(), 0.05)
check("reactive min -2.6", -2.6, sf[sf.omission == "reactive"].delta.min(), 0.05)
check("voltage min -2.7", -2.7, sf[sf.omission == "voltage"].delta.min(), 0.05)
check("voltage max +0.9", 0.9, sf[sf.omission == "voltage"].delta.max(), 0.05)
nan12 = sf[(sf.omission == "reactive") & (sf.value == 1.2) & (sf.share == 0.4)]
check("reactive 1.2 @0.4: both forms unrecorded (script drops the pair)", 1.0,
      float(nan12.cost_box.isna().all() and nan12.cost_circle.isna().all()), 0)
expect("5.1 must not claim the bound clears at 120%", "while the bound still clears", "main", present=False)
# re-run 2026-09-29: box AND current are infeasible on both systems at share 0.4 (m = 24 and 192)

# ------------------------------------------------------------------ exact cone / S2
ec = pd.read_csv(R / "exact_cone.csv")
for m in (24, 96):
    ec[f"gap{m}"] = 100 * (ec[f"cost_m{m}"] - ec.cost_cone) / ec.cost_cone
ec["d24"] = delta(ec.cost_box, ec.cost_m24)
ec["dcone"] = delta(ec.cost_box, ec.cost_cone)
check("cone gap m24 max 0.066", 0.066, ec.gap24.max(), 0.0005)
check("cone gap m96 max 0.0045", 0.0045, ec.gap96.max(), 0.0001)
check("|d24 - dcone| max 0.056", 0.056, (ec.d24 - ec.dcone).abs().max(), 0.0005)
check("d24 <= dcone always", 1.0, float((ec.d24 <= ec.dcone + 1e-9).all()), 0)

# ------------------------------------------------------------------ placement / Fig. 8 / 118
pl = pd.read_csv(R / "placement_spread.csv")
pl["delta"] = delta(pl.cost_box, pl.cost_set)
rnd = pl[pl.placement == "random"]
check("accepted draws 666", 666, len(rnd), 0)
check("feasible draws 605", 605, int(rnd.feasible.sum()), 0)
check("infeasible all rts24", 1.0, float((rnd[~rnd.feasible].case == "rts24").all()), 0)
fe = rnd[rnd.feasible]
check("all delta > 0", 1.0, float((fe.delta > 0).all()), 0)
g = fe.groupby(["case", "share"]).delta
check("spread factor min 2.6", 2.6, (g.max() / g.min()).min(), 0.05)
check("spread factor max 34.5", 34.5, (g.max() / g.min()).max(), 0.05)
s118 = pd.read_csv(R / "scale_check_118.csv")
check("118 max +0.12", 0.12, s118.delta_pct.max(), 0.005)
check("118 min -0.68", -0.68, s118.delta_pct.min(), 0.005)

# ------------------------------------------------------------------ Table 6 / Sec. 5.4
ac = pd.read_csv(R / "ac_feasible.csv")
check("ac all pass", 1.0, float(ac.passes.all()), 0)
w = ac.pivot_table(index=["case", "share"], columns="form", values="cost")
d_ac = delta(w["box"], w["current"])
for key, v in {("ieee39", 0.2): 0.85, ("ieee39", 0.4): 0.07, ("rts24", 0.2): 0.59, ("rts24", 0.4): 1.00}.items():
    check(f"ac delta {key}", v, d_ac[key], 0.005)
check("ac cost change min 1.0", 1.0, (100 * (ac.cost - ac.dc_cost) / ac.dc_cost).min(), 0.05)
check("ac cost change max 2.4", 2.4, (100 * (ac.cost - ac.dc_cost) / ac.dc_cost).max(), 0.05)
check("ac rounds 5..13", 13, ac.rounds.max(), 0)
check("ac loading max 99.9", 99.9, ac.loading_max.max(), 0.05)
check("ac balance < 1 MW", 0.96, ac.balance_mw.max(), 0.005)
check("bound current 1.40 (ac)", 1.40, ac[ac.form == "box"].conv_current_any_form.max(), 0.005)
check("set price ac max 3.03", 3.03, ac[ac.form == "current"].price_inertia.max(), 0.005)
check("bound price ac min 6.8", 6.76, ac[ac.form == "box"].price_inertia.min(), 0.01)
check("bound price ac max 8.5", 8.55, ac[ac.form == "box"].price_inertia.max(), 0.01)
rounds = pd.read_csv(R / "ac_feasible_rounds.csv")
r0 = rounds[rounds["round"] == 0]
check("lossless slack min 33 (text says 32)", 32.6, r0.balance_mw.min(), 0.05)
check("lossless slack max 76", 76.3, r0.balance_mw.max(), 0.05)
check("lossless loading max 103.2", 103.2, r0.loading_max.max(), 0.05)

# ------------------------------------------------------------------ Table 7 / Sec. 5.5
so = pd.read_csv(R / "secured_outages.csv")
check("secured all feasible", 1.0, float(so.feasible_secured.all()), 0)
check("secured over plain max 29", 29, so.screen_over_plain.max(), 0)
check("secured checked at that row 47", 47, int(so.loc[so.screen_over_plain.idxmax(), "screen_checked"]), 0)
check("secured worst plain 129", 129.2, so.screen_worst_plain.max(), 0.05)
check("secured worst after max 100.03 (r40 rerun)", 100.03, so.screen_worst_secured.max(), 0.01)
check("secured worst after min is 90.1 (not 100.0)", 90.07, so.screen_worst_secured.min(), 0.05)
expect("5.5 must not say 100.0-100.1 in all eight", "in all eight clearings", "main", present=False)
expect("5.5 within 0.03% wording", "within 0.03% of the emergency rating", "main")
check("secured cost increase max 0.06", 0.06, so.cost_increase_pct.max(), 0.005)
check("secured fraction converged (<1e-3) in all eight", 1.0, float((so.fraction_move < 1e-3).all()), 0)
wp = so.pivot_table(index=["case", "share"], columns="form", values="cost_plain")
ws = so.pivot_table(index=["case", "share"], columns="form", values="cost_secured")
check("rts24 0.4 delta plain 1.08 (text says 1.09)", 1.08, delta(wp["box"], wp["current"])[("rts24", 0.4)], 0.005)
check("rts24 0.4 delta secured 1.14", 1.14, delta(ws["box"], ws["current"])[("rts24", 0.4)], 0.005)
expect("5.5 must not say from 1.09%", "from 1.09% to 1.14%", "main", present=False)
expect("5.5 says 1.08% to 1.14%", "from 1.08% to 1.14%", "main")

# ------------------------------------------------------------------ Table 8 / Sec. 5.6
dc = pd.read_csv(R / "dynamic_check.csv")
check("dynamic runs 360", 360, len(dc), 0)
check("limiter engaged all", 1.0, float(dc.limiter_engaged.astype(str).isin(["T", "True"]).all()), 0)
check("t_lim min 0.07", 0.07, dc.rise_time.min(), 0.005)
check("t_lim max 0.34", 0.34, dc.rise_time.max(), 0.005)
check("current never above limit", 0.0, (dc.peak_current - dc.i_short).max(), 1e-9)
hold = dc[dc.reactive_mode == "hold schedule"]
const = hold[hold.voltage == "grid voltage holds"]
fall = hold[hold.voltage != "grid voltage holds"]
check("worst delivery falling grid 54.0", 54.0, 100 * fall.settled_frac.min(), 0.05)
check("corr(delivery, x_g) -0.93", -0.93, np.corrcoef(const.settled_frac, const.x_grid)[0, 1], 0.005)
droop = dc[(dc.reactive_mode != "hold schedule") & (dc.voltage != "grid voltage holds")]
check("droop lost responses 16 of 90", 16, int((droop.settled_frac <= 1e-6).sum()), 0)
check("droop falling runs 90", 90, len(droop), 0)
cons = const[const.region == "conservative"]
check("T8 conservative rows pool Is=1.1 and 1.5", 2, cons.i_short.nunique(), 0)
for xg, lo, hi in [(0.05, 99.5, 99.7), (0.15, 95.1, 97.7), (0.30, 79.9, 90.6)]:
    s = cons[np.isclose(cons.x_grid, xg)]
    check(f"T8 conservative delivered lo xg={xg}", lo, 100 * s.settled_frac.min(), 0.05)
    check(f"T8 conservative delivered hi xg={xg}", hi, 100 * s.settled_frac.max(), 0.05)
cf = pd.read_csv(R / "corrected_face.csv").set_index(["case", "share"])
check("v drop max 12.4", 12.4, cf.v_drop_pct.max(), 0.05)
check("h total unchanged", 0.0, (cf.h_after - cf.h_before).abs().max(), 1e-6)
for key, b, a in [(("ieee39", 0.2), 86.3, 89.9), (("ieee39", 0.4), 80.7, 83.4), (("rts24", 0.2), 92.8, 93.3), (("rts24", 0.4), 97.7, 97.0)]:
    check(f"energy before {key}", b, 100 * cf.loc[key, "energy_before"], 0.06)
    check(f"energy after {key}", a, 100 * cf.loc[key, "energy_after"], 0.06)
check("p10 54.0 -> 75.9", 75.9, 100 * cf.energy_p10_after.loc[("ieee39", 0.4)], 0.06)

# ------------------------------------------------------------------ Sec. 4.1 / S10 base cases
bc = pd.read_csv(R / "base_case_repair.csv").set_index(["case", "stage"])
check("ieee39 shipped vmax 1.0636", 1.0636, bc.loc[("ieee39", "shipped"), "v_max"], 5e-5)
check("rts24 shipped vmin 0.919", 0.919, bc.loc[("rts24", "shipped"), "v_min"], 5e-4)
check("rts24 shipped vmax 1.058", 1.058, bc.loc[("rts24", "shipped"), "v_max"], 5e-4)
check("repaired loading ieee39 = 77 (text says 73; 73 is lines only)", 77.0, bc.loc[("ieee39", "repaired"), "worst_loading"], 0.5)
expect("4.1 says 77% and 90%", "loading is 77% and 90%", "main")
be = pd.read_csv(R / "base_case_repair_effect.csv")
check("repair effect < 0.001 pp", 0.0, be.change_pp.abs().max(), 0.001)
expect("4.1 repair effect wording", "below 0.001 percentage points", "main")
vs = pd.read_csv(R / "corrected_face_vshort.csv")
check("V_sh min 0.88", 0.88, vs.v_short_min.min(), 0.005)
check("face reduction max 13.0", 13.0, vs.face_reduction_max_pct.max(), 0.05)
expect("5.6 face 13.0% wording", "lowers the short-term face by up to 13.0%", "main")
expect("5.6 must not say 12.4%", "face by up to 12.4%", "main", present=False)
check("repaired loading rts24 = 90", 90.0, bc.loc[("rts24", "repaired"), "worst_loading"], 0.5)
expect("4.1 must not say 73%", "loading is 73% and 90%", "main", present=False)

# ------------------------------------------------------------------ Table 3 / P/S
rb = pd.read_csv(R / "rating_basis.csv")
check("P/S min 0.857", 0.857, rb.pf_min.min(), 0.0005)
check("P/S max 0.952", 0.952, rb.pf_max.max(), 0.0005)
check("P/S mean of configuration means 0.921", 0.921, rb.pf_mean.mean(), 0.0005)
notes.append("P/S 'mean 0.921' is the mean of the four configuration means; the per-unit mean is 0.916 (pooled) / 0.915 (unique units)")

# ------------------------------------------------------------------ S1 gate table
cs_ = pd.read_csv(R / "case_study.csv")
gate = cs_[cs_.axis == "threshold"].pivot_table(index=["case", "share"], columns="soc_threshold", values="mean_availability")
print(gate.round(3).to_string())
check("S1 availability at base 0.20 is NOT 1.00 (ieee39 0.2 = 0.71)", 0.705, gate.loc[("ieee39", 0.2), 0.2], 0.005)
check("S1 availability at 0.30, ieee39 0.4 = 0.51 (table says 0.55)", 0.506, gate.loc[("ieee39", 0.4), 0.3], 0.005)
check("S1 availability at 0.30, rts24 0.2 = 0.35 (table says 0.42)", 0.349, gate.loc[("rts24", 0.2), 0.3], 0.005)
check("S1 availability at 0.30, rts24 0.4 = 0.43 (table says 0.35)", 0.435, gate.loc[("rts24", 0.4), 0.3], 0.005)
expect("S1 must not print 0.30 row as 0.62/0.55/0.42/0.35", "0.30 0.96 0.62 0.34 0.55 0.68 0.42 1.08 0.35", "supp", present=False)

# ------------------------------------------------------------------ S4 / S5 / S7 / S9
du = pd.read_csv(R / "dual_check.csv")
check("dual rows 1648", 1648, len(du), 0)
check("dual all ok", 1.0, float(du.ok.all()), 0)
ine = du[du.kind == "inertia"]
check("degenerate inertia rows 64", 64, int(((ine.left - ine.right).abs() > 1e-6).sum()), 0)
pb = pd.read_csv(R / "price_brackets.csv")
check("brackets coincide", 1.0, float(((pb.box_left - pb.box_right).abs() < 1e-6).all() and ((pb.current_left - pb.current_right).abs() < 1e-6).all()), 0)
pc = pd.read_csv(R / "post_contingency.csv")
check("S7 worst 161.2", 161.2, pc.loading_post_max.max(), 0.05)
check("S7 over max 200", 200, pc.over_rating.max(), 0)
for V in (0.95, 1.05):
    check(f"S9 crossing V={V} Is=1.1", {0.95: 0.303, 1.05: 0.578}[V], np.sqrt((V * 1.1) ** 2 - 1), 0.0005)

# ------------------------------------------------------------------ cross-references in the supplementary (hard-coded numbers)
expect("S3 cites Eq. (16) for stored energy (not 11)", "Eq. (11) counts the stored energy", "supp", present=False)
expect("S7 cites Eq. (21) for the secured rows (not 20)", "before Eq. (20) is imposed", "supp", present=False)

# ------------------------------------------------------------------ wording that the data do not support
expect("abstract 'about 1%' (base is 0.3-1.1%)", "about 1% at the base case", "main", present=False)
expect("abstract 0.3 to 1.1%", "0.3 to 1.1% at the base", "main")
expect("S1 base row 0.71/0.54/0.41/0.47", "0.20 (base) 0.96 0.71 0.34 0.54 0.68 0.41 1.08 0.47", "supp")
expect("S3 cites Eq. (16)", "Eq. (16) counts the stored energy", "supp")
expect("S7 cites Eq. (21)", "before Eq. (21) is imposed", "supp")
expect("5.1 neither representation clears", "neither representation clears at 40% converter share", "main")
expect("Table 8 label 1.1, 1.5 pu", "(1.1, 1.5 pu)", "main")
expect("5.2 '32 cases' (28 distinct)", "positive in all 32 cases", "main", present=False)
expect("5.1 'not on the converter share'", "not on the converter share", "main", present=False)

print("\nNOTES:")
for n in notes:
    print(" -", n)
print("\nFAILED:", len(fails))
for f in fails:
    print(" -", f)
sys.exit(1 if fails else 0)
