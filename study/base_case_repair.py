#!/usr/bin/env python3
"""Put the source cases inside their own declared limits before using them.

Section 5.11 reports that every cleared schedule violates something, and part
of that is inherited: run untouched, pandapower's IEEE 39-bus reaches
±1.0636 pu against a declared 1.06, and RTS-24 spans 0.9190 to 1.0581
against a declared 0.95 to 1.05. Inheriting a violation and then reporting it
as a property of the schedule is not a diagnostic, it is a confusion, and the
review asks for the base case to be repaired rather than for the inheritance
to be explained.

Repair here means what the review names: generator voltage setpoints, shunts
and transformer taps. It does not mean redispatch. Moving active power would
change which case is being studied; moving the voltage controls changes only
how the same dispatch is supported, which is what an operator does and what
the shipped cases evidently did not bother to do.

The search is deliberately small and deliberately least-change. Setpoints are
continuous within the range the case's own generators declare, taps are
integers, and the objective is the squared voltage violation with a light
penalty on moving away from the shipped values. A repair that wandered far
from the shipped case would be a different system, and the point is to keep
the same system and fix its voltage profile.

Writes results/base_case_repair.csv, which is the table the manuscript needs
to state what was changed.
"""
from __future__ import annotations

import copy
import itertools
import pathlib
import sys

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                       / "model"))

import voltage_control  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"

CASES = {"ieee39": pn.case39, "rts24": pn.case24_ieee_rts}
SETPOINT_STEP = 0.005
SETPOINT_SPAN = 0.06        # how far a setpoint may move from its shipped value
ROUNDS = 12


def report(net) -> dict:
    pp.runpp(net, enforce_q_lims=True)
    low, high = voltage_control.band(net)
    v = net.res_bus.vm_pu.to_numpy()
    loading = [float(net.res_line.loading_percent.max()) if len(net.res_line)
               else 0.0]
    if len(net.res_trafo):
        loading.append(float(net.res_trafo.loading_percent.max()))
    return dict(v_min=float(v.min()), v_max=float(v.max()),
                under=int((v < low - 1e-9).sum()),
                over=int((v > high + 1e-9).sum()),
                worst_loading=max(loading))


def main() -> None:
    rows, changes = [], []
    for name, load in CASES.items():
        shipped = load()
        before = report(shipped)
        fixed, residual = voltage_control.correct(load())
        after = report(fixed)

        rows.append(dict(case=name, stage="shipped", **before))
        rows.append(dict(case=name, stage="repaired", **after,
                         residual=residual))

        original = load()
        for index in fixed.gen.index:
            was = float(original.gen.at[index, "vm_pu"])
            now = float(fixed.gen.at[index, "vm_pu"])
            if abs(now - was) > 1e-9:
                changes.append(dict(case=name, control="gen",
                                    index=int(index),
                                    at=int(fixed.gen.at[index, "bus"]),
                                    was=was, now=now))
        for index in fixed.trafo.index:
            was = original.trafo.at[index, "tap_pos"]
            now = fixed.trafo.at[index, "tap_pos"]
            if pd.notna(was) and pd.notna(now) and was != now:
                changes.append(dict(case=name, control="tap",
                                    index=int(index),
                                    at=int(fixed.trafo.at[index, "hv_bus"]),
                                    was=float(was), now=float(now)))

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "base_case_repair.csv", index=False)
    moved = pd.DataFrame(changes).rename(columns={"index": "index_"})
    moved.rename(columns={"index_": "index"}).to_csv(
        RESULTS / "base_case_changes.csv", index=False)

    print(frame.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
    print()
    if len(moved):
        print(f"바꾼 제어 {len(moved)}개:")
        print(moved.to_string(index=False, float_format=lambda x: f"{x:7.4f}"))
    else:
        print("바꾼 것이 없습니다")
    clean = frame[frame.stage == "repaired"]
    if (clean.under + clean.over).sum() == 0:
        print("\n두 계통 모두 자기 선언 한계 안에 들어왔습니다")
    else:
        print("\n아직 남은 위반:")
        print(clean[["case", "under", "over", "v_min", "v_max"]]
              .to_string(index=False))
    # The model carries these values as literals so that it does not depend on
    # this file having been run. That is only safe if the two cannot drift, so
    # the comparison is made here and fails loudly.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                           / "model"))
    from multiperiod import BASE_CASE_REPAIR          # noqa: PLC0415

    found = {name: {"gen": {}, "tap": {}} for name in CASES}
    for row in moved.itertuples():
        found[row.case]["gen" if row.control == "gen" else "tap"][
            int(row.index_)] = round(float(row.now), 6)
    for name in CASES:
        carried = {kind: {int(k): round(float(v), 6)
                          for k, v in BASE_CASE_REPAIR[name][kind].items()}
                   for kind in ("gen", "tap")}
        if carried != found[name]:
            print(f"\n{name}: 모형이 든 수리값이 이 실행과 다릅니다")
            print(f"  모형: {carried}")
            print(f"  실행: {found[name]}")
            raise SystemExit("model/multiperiod.py 의 BASE_CASE_REPAIR 를 "
                             "맞추십시오")
    print("모형이 든 수리값과 일치합니다")

    print("\nwrote results/base_case_repair.csv, results/base_case_changes.csv")


if __name__ == "__main__":
    sys.exit(main())
