#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-runs requested by the 2026-09-29 audit of the r2 manuscript (items E2, E1, B6).

(1) E2  cost-difference change caused by the base-case voltage repair: clear the four base
        configurations on the *unrepaired* shipped cases and compare delta with the repaired run.
(2) E1  minimum corrected short-term terminal voltage V^sh (Eq. 23) behind the 'as low as 0.88 pu'
        sentence of the conclusion; writes results/corrected_face_vshort.csv.
(3) B6  secured-outage fraction iteration for IEEE 39 / 20 % / capability set with the round cap
        raised from 14 to 40, so the 1e-3 stopping rule is actually met; writes
        results/secured_outages_ieee39_02_current_r40.csv.
Run with the SEGAN venv python. Nothing in results/ is overwritten.
"""
from __future__ import annotations

import dataclasses
import io
import itertools
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "study"))
import multiperiod as mp  # noqa: E402
from procurement_lp import Capability  # noqa: E402
import prices  # noqa: E402

R = ROOT / "study" / "results"
SET = Capability(i_short_term=1.5, priority="reactive", soc_threshold=0.20)
DISPATCH = 0.8


def base_case(name, share):
    base = mp.build_case(name, converter_share=share, demand_peak_mw=mp.common_demand_peak(name))
    return dataclasses.replace(base, conv_injection_pu=np.full(len(base.demand_mw), DISPATCH))


def part_e2():
    print("== (1) E2: repaired vs unrepaired base cases", flush=True)
    rep = pd.read_csv(R / "sensitivity.csv")
    rep = rep[rep.axis == "reference"].set_index(["case", "share"])
    saved = mp.BASE_CASE_REPAIR
    rows = []
    try:
        mp.BASE_CASE_REPAIR = {k: {"gen": {}, "tap": {}} for k in saved}
        for name, share in itertools.product(("ieee39", "rts24"), (0.2, 0.4)):
            t0 = time.time()
            case = base_case(name, share)
            out = {f: mp.MultiPeriod(case, SET, f).solve() for f in ("box", "current")}
            d_unrep = 100.0 * (out["box"]["cost"] - out["current"]["cost"]) / out["box"]["cost"]
            d_rep = 100.0 * (rep.loc[(name, share), "cost_box"] - rep.loc[(name, share), "cost_set"]) / rep.loc[(name, share), "cost_box"]
            rows.append(dict(case=name, share=share, delta_unrepaired=d_unrep, delta_repaired=d_rep,
                             change_pp=d_rep - d_unrep,
                             price_box_unrep=prices.price(out["box"]["prices"], "inertia"),
                             price_set_unrep=prices.price(out["current"]["prices"], "inertia")))
            print(f"  {name} {share}: unrepaired {d_unrep:.4f} repaired {d_rep:.4f} change {d_rep - d_unrep:+.4f} pp  ({time.time()-t0:.0f}s)", flush=True)
    finally:
        mp.BASE_CASE_REPAIR = saved
    df = pd.DataFrame(rows); df.to_csv(R / "base_case_repair_effect.csv", index=False)
    print("  max |change| = %.4f pp" % df.change_pp.abs().max(), flush=True)


def part_e1():
    print("== (2) E1: minimum corrected short-term voltage", flush=True)
    import corrected_face as cf
    rows = []
    for name, share in itertools.product(("ieee39", "rts24"), (0.2, 0.4)):
        t0 = time.time()
        case = base_case(name, share)
        plain = mp.MultiPeriod(case, SET, "current").solve()
        reactance = cf.thevenin(case, plain, SET)
        corrected = cf.event_voltage(case, plain, reactance, SET)
        v = np.asarray(case.conv_voltage_pu)
        rows.append(dict(case=name, share=share, v_pre_min=float(v.min()), v_pre_max=float(v.max()),
                         v_short_min=float(corrected.min()), v_short_max=float(corrected.max()),
                         drop_max_pu=float((v - corrected).max()),
                         face_reduction_max_pct=float(100.0 * ((v - corrected) / v).max())))
        print(f"  {name} {share}: V_sh min {corrected.min():.4f} (pre {v.min():.4f}-{v.max():.4f}), face -{100*((v-corrected)/v).max():.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(R / "corrected_face_vshort.csv", index=False)


def part_b6():
    print("== (3) B6: secured fraction iteration, ieee39 0.2 current, 40 rounds", flush=True)
    import secured_outages as so
    case = base_case("ieee39", 0.2)
    order = [int(w) for w in np.argsort(-case.sync_pmax_mw)[:so.SECURED]]
    plain_model = mp.MultiPeriod(case, SET, "current")
    plain = plain_model.solve()
    start = so.fractions(case, plain_model, plain, order)
    t0 = time.time()
    model, tight, rounds, move = so.solve_secured(case, SET, "current", order, rounds=40, start=start)
    screen = so.screen(case, model, tight, only=set(order))
    row = dict(case="ieee39", share=0.2, form="current", secured_units=len(order), fraction_rounds=rounds,
               fraction_move=move, cost_plain=plain["cost"], cost_secured=tight["cost"],
               cost_increase_pct=100.0 * (tight["cost"] - plain["cost"]) / plain["cost"],
               price_inertia_plain=prices.price(plain["prices"], "inertia"),
               price_inertia_secured=prices.price(tight["prices"], "inertia"),
               price_reserve_plain=prices.price(plain["prices"], "reserve"),
               price_reserve_secured=prices.price(tight["prices"], "reserve"),
               screen_over_secured=screen["over"], screen_worst_secured=screen["worst"],
               screen_checked=screen["checked"], seconds=time.time() - t0)
    pd.DataFrame([row]).to_csv(R / "secured_outages_ieee39_02_current_r40.csv", index=False)
    print("  ", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["e2", "e1", "b6"]
    for w in which:
        {"e2": part_e2, "e1": part_e1, "b6": part_b6}[w]()
    print("done", flush=True)
