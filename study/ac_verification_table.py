#!/usr/bin/env python3
"""The ac verification table the review asks for, in one place.

The instruction is explicit: run an ac power flow on the final solution and
present, as a table, whether bus voltage, generator reactive output, branch
rating and the active balance are inside tolerance. The manuscript has all of
those numbers and none of that table -- they are scattered through the prose
of Section 7, which is where a reader who wants to check the claim cannot use
them.

Scattering them also lets a gap hide. Writing the columns out forces every
configuration to answer every question, and a configuration that has no
answer shows up as a blank rather than as a sentence nobody wrote.

Two things are reported that the instruction does not name and the review
does. The correction is iterative, so the round count and the residual it
stopped at belong with the result: a schedule that is inside the band after
one round and one that took five are not the same claim. And the converter
representation is a column, because the screen is run both ways and a table
that silently picked one would be picking it after seeing the answer.

Writes results/ac_verification.csv and the LaTeX table body.
"""
from __future__ import annotations

import copy
import dataclasses
import itertools
import pathlib
import sys

import numpy as np
import pandapower as pp
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import voltage_control  # noqa: E402
from multiperiod import (MultiPeriod, ac_voltage_pass,  # noqa: E402
                         build_case, common_demand_peak)
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
CASES = ("ieee39", "rts24")
SHARES = (0.2, 0.4)
FORMS = ("box", "current")
DISPATCH = 0.8
ROUNDS = 5


def measure(case, outcome, controls, model) -> dict:
    """Every quantity the instruction names, on one schedule."""
    networks: list = []
    ac_voltage_pass(case, outcome, solved=networks, controls=controls)

    worst = dict(v_min=np.inf, v_max=-np.inf, v_slack_lo=0.0, v_slack_hi=0.0,
                 q_at_limit=0, loading_max=0.0, balance_mw=0.0,
                 diverged=0)
    reference = next((j for j, (table, _) in enumerate(case.sync_gen_index)
                      if table == "ext_grid"), None)
    if reference is None:
        reference = int(np.argmax(case.sync_rating_mva))

    for period, net in enumerate(networks):
        try:
            pp.runpp(net, enforce_q_lims=True, numba=False)
        except Exception:                                # noqa: BLE001
            worst["diverged"] += 1
            continue
        low = net.bus.min_vm_pu.fillna(0.94).to_numpy()
        high = net.bus.max_vm_pu.fillna(1.06).to_numpy()
        v = net.res_bus.vm_pu.to_numpy()
        worst["v_min"] = min(worst["v_min"], float(v.min()))
        worst["v_max"] = max(worst["v_max"], float(v.max()))
        worst["v_slack_lo"] = max(worst["v_slack_lo"],
                                  float(np.maximum(low - v, 0.0).max()))
        worst["v_slack_hi"] = max(worst["v_slack_hi"],
                                  float(np.maximum(v - high, 0.0).max()))

        if len(net.gen):
            q = net.res_gen.q_mvar.to_numpy()
            lo = net.gen.min_q_mvar.to_numpy(dtype=float)
            hi = net.gen.max_q_mvar.to_numpy(dtype=float)
            at = ((np.abs(q - lo) < 1e-6) | (np.abs(q - hi) < 1e-6)).sum()
            worst["q_at_limit"] = max(worst["q_at_limit"], int(at))

        loading = [float(net.res_line.loading_percent.max())]
        if len(net.res_trafo):
            loading.append(float(net.res_trafo.loading_percent.max()))
        worst["loading_max"] = max(worst["loading_max"], max(loading))

        cleared = float(outcome["pg"][period, reference])
        table, index = case.sync_gen_index[reference]
        if table == "ext_grid" and len(net.res_ext_grid):
            actual = float(net.res_ext_grid.p_mw.sum())
        elif index in net.res_gen.index:
            actual = float(net.res_gen.at[index, "p_mw"])
        else:
            actual = cleared
        worst["balance_mw"] = max(worst["balance_mw"], abs(actual - cleared))
    return worst


def run(name: str, share: float, form: str, settings) -> dict:
    base = build_case(name, converter_share=share,
                      demand_peak_mw=common_demand_peak(name))
    case = dataclasses.replace(
        base, conv_injection_pu=np.full(len(base.demand_mw), DISPATCH))

    controls, residual, rounds = None, float("nan"), 0
    for rounds in range(1, ROUNDS + 1):
        model = MultiPeriod(case, settings, form)
        outcome = model.solve()
        if not outcome["feasible"]:
            return dict(case=name, share=share, form=form, feasible=False)
        networks: list = []
        ac_voltage_pass(case, outcome, solved=networks, controls=controls)
        carried, residual = [], 0.0
        for period, net in enumerate(networks):
            before = copy.deepcopy(net)
            fixed, left = voltage_control.correct(net)
            residual = max(residual, left)
            keep = (copy.deepcopy(controls[period]) if controls
                    else {"gen": {}, "tap": {}})
            for change in voltage_control.moved(before, fixed):
                keep.setdefault(change["control"], {})[change["index"]] = \n                    change["now"]
            carried.append(keep)
        controls = carried
        resulting = ac_voltage_pass(case, outcome, controls=controls)
        move = float(np.abs(resulting - case.conv_voltage_pu).max())
        case = dataclasses.replace(
            case, conv_voltage_pu=case.conv_voltage_pu
            + 0.5 * (resulting - case.conv_voltage_pu))
        if residual <= 1e-9 and move < 1e-3:
            break

    row = dict(case=name, share=share, form=form, feasible=True,
               rounds=rounds, residual_pu2=residual)
    row.update(measure(case, outcome, controls, model))
    return row


def main() -> None:
    settings = Capability(i_short_term=1.5, priority="reactive",
                          soc_threshold=0.20)
    rows = [run(name, share, form, settings)
            for name, share, form in itertools.product(CASES, SHARES, FORMS)]
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "ac_verification.csv", index=False)

    show = ["case", "share", "form", "rounds", "v_min", "v_max",
            "v_slack_hi", "q_at_limit", "loading_max", "balance_mw",
            "diverged"]
    print(frame.reindex(columns=show)
          .to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
    live = frame[frame.feasible]
    clean = live[(live.v_slack_lo <= 1e-6) & (live.v_slack_hi <= 1e-6)
                 & (live.loading_max <= 100.0 + 1e-6)]
    print(f"\n전압과 선로를 모두 만족하는 설정 {len(clean)}/{len(live)}")
    print(f"발산한 기간이 있는 설정 {int((live.diverged > 0).sum())}")
    print(f"\nwrote {RESULTS / 'ac_verification.csv'}")


if __name__ == "__main__":
    sys.exit(main())
