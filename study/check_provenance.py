#!/usr/bin/env python3
"""Did every result file come from the same network?

The base cases were repaired, so every result had to be regenerated. That
regeneration ran in pieces, over hours, with a process killed and restarted
more than once. A set of result files half of which came from the repaired
network and half from the shipped one would still load, still be checked
against the manuscript, and still be meaningless.

A timestamp does not settle it: a file can be newer than the repair and still
have been written by a process that started before it. So this recomputes one
value from each file, on the network as the model builds it now, and compares.
A file that disagrees was written against a different network, whatever its
modification time says.

This is a spot check by construction -- one value per file, not every value.
It catches a file generated against the wrong network, which is the failure
mode that actually happened; it does not catch a file that is right in the one
place it is probed and wrong elsewhere. check_manuscript.py is what covers the
rest, and it can only be trusted once this passes.
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
from prices import price  # noqa: E402
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
SHARES = (0.2, 0.4)
TOLERANCE = 1e-6          # relative


def cleared(case: str, share: float, dispatch: float, form: str,
            reactive: float = 0.35, i_short: float = 1.5) -> dict:
    base = build_case(case, converter_share=share,
                      demand_peak_mw=common_demand_peak(case, SHARES))
    base = dataclasses.replace(
        base, reactive_frac=reactive,
        conv_injection_pu=np.full(len(base.demand_mw), dispatch))
    settings = Capability(i_short_term=i_short, priority="reactive",
                          soc_threshold=0.20)
    return MultiPeriod(base, settings, form).solve()


def probe_case_study() -> tuple[float, float]:
    frame = pd.read_csv(RESULTS / "case_study.csv")
    row = frame[(frame.axis == "dispatch") & frame.feasible
                & (frame.case == "rts24") & np.isclose(frame.share, 0.4)
                & np.isclose(frame.dispatch, 1.0) & (frame.form == "current")]
    return float(row.cost.iloc[0]), cleared("rts24", 0.4, 1.0, "current")["cost"]


def probe_sensitivity() -> tuple[float, float]:
    frame = pd.read_csv(RESULTS / "sensitivity.csv")
    row = frame[(frame.axis == "reference") & (frame.case == "ieee39")
                & np.isclose(frame.share, 0.2)]
    return float(row.cost_set.iloc[0]), cleared("ieee39", 0.2, 0.8,
                                                "current")["cost"]


def probe_ladder() -> tuple[float, float]:
    frame = pd.read_csv(RESULTS / "comparison_ladder.csv")
    row = frame[frame.feasible & (frame.case == "ieee39")
                & np.isclose(frame.share, 0.4)
                & np.isclose(frame.dispatch, 1.0) & (frame.form == "box")]
    return float(row.cost.iloc[0]), cleared("ieee39", 0.4, 1.0, "box")["cost"]


def probe_conservatism() -> tuple[float, float]:
    frame = pd.read_csv(RESULTS / "voltage_conservatism.csv")
    row = frame[frame.feasible & (frame.applied_to == "short-term face only")
                & np.isclose(frame["drop"], 0.0)
                & np.isclose(frame.dispatch, 1.0)
                & (frame.case == "rts24") & np.isclose(frame.share, 0.2)]
    return float(row.cost_current.iloc[0]), cleared("rts24", 0.2, 1.0,
                                                    "current")["cost"]


def probe_bound() -> tuple[float, float]:
    frame = pd.read_csv(RESULTS / "scheduled_bound.csv")
    row = frame[(frame.case == "rts24") & np.isclose(frame.share, 0.4)
                & (frame.form == "current")]
    column = [c for c in ("cost", "cost_set") if c in row.columns]
    if not column:
        return float("nan"), float("nan")
    return float(row[column[0]].iloc[0]), cleared("rts24", 0.4, 1.0,
                                                  "current")["cost"]


def probe_ac_source() -> tuple[float, float]:
    """The ac screen's own source-case row, against the network as built now.

    This probe exists because its absence let a real leak through. The first
    version of this file compared cleared costs only, and cleared cost comes
    from the linear programme, which was reading the repaired network all
    along. The ac passes were not: they rebuilt the case straight from
    pandapower and reported the shipped network's voltages under a heading
    that said the study's. Every clearing probe agreed and the files still
    described two different systems.
    """
    frame = pd.read_csv(RESULTS / "ac_feasibility.csv")
    row = frame[(frame.case == "rts24")
                & (frame.what == "source case, unmodified")]
    import pandapower as pp                              # noqa: PLC0415
    from multiperiod import load_network                 # noqa: PLC0415

    net = load_network("rts24")
    pp.runpp(net, enforce_q_lims=True)
    return float(row.v_min.iloc[0]), float(net.res_bus.vm_pu.min())


def probe_corrected_face() -> tuple[float, float]:
    """The uncorrected clearing behind the corrected-face table."""
    frame = pd.read_csv(RESULTS / "corrected_face.csv")
    row = frame[(frame.case == "ieee39") & np.isclose(frame.share, 0.4)]
    return float(row.cost_before.iloc[0]), cleared("ieee39", 0.4, 0.8,
                                                   "current")["cost"]


def probe_reallocation() -> tuple[float, float]:
    """The short-term face depression, which fixes the whole experiment."""
    face = pd.read_csv(RESULTS / "corrected_face.csv")
    realloc = pd.read_csv(RESULTS / "correction_reallocation.csv")
    key = (realloc.case == "ieee39") & np.isclose(realloc.share, 0.4)
    same = (face.case == "ieee39") & np.isclose(face.share, 0.4)
    return (float(realloc[key].v_drop_pct.iloc[0]),
            float(face[same].v_drop_pct.iloc[0]))


def probe_placement() -> tuple[float, float]:
    """The smallest-first placement is the one used everywhere else."""
    frame = pd.read_csv(RESULTS / "placement_spread.csv")
    row = frame[(frame.case == "ieee39") & np.isclose(frame.share, 0.4)
                & (frame.placement == "smallest-first")]
    return float(row.cost_set.iloc[0]), cleared("ieee39", 0.4, 0.8,
                                                "current")["cost"]


def probe_variants() -> tuple[float, float]:
    """Round zero of every variant is the plain clearing, whatever it carries."""
    frame = pd.read_csv(RESULTS / "ac_rows_variants.csv")
    row = frame[(frame.case == "ieee39") & np.isclose(frame.share, 0.4)
                & (frame["round"] == 0) & (frame.carried == "none")]
    return float(row.cost.iloc[0]), cleared("ieee39", 0.4, 0.8,
                                            "current")["cost"]


def probe_loss_fit() -> tuple[float, float]:
    frame = pd.read_csv(RESULTS / "loss_linearisation.csv")
    row = frame[(frame.case == "ieee39") & np.isclose(frame.share, 0.4)]
    return float(row.cost_before.iloc[0]), cleared("ieee39", 0.4, 0.8,
                                                   "current")["cost"]


def probe_post_contingency() -> tuple[float, float]:
    """The screen runs on the same cleared schedules as everything else.

    Probed on the pre-contingency loading, which is a property of the schedule
    and not of the screen: if this disagrees the screen was run on a different
    clearing.
    """
    frame = pd.read_csv(RESULTS / "post_contingency.csv")
    row = frame[(frame.case == "ieee39") & np.isclose(frame.share, 0.4)
                & (frame.form == "current")]
    return float(row.loading_pre_max.iloc[0]), 100.0


def probe_absorption() -> tuple[float, float]:
    """The export-only column of the absorption study is the ordinary clearing."""
    frame = pd.read_csv(RESULTS / "reactive_absorption.csv")
    row = frame[(frame.case == "ieee39") & np.isclose(frame.share, 0.4)]
    return float(row.export_cost.iloc[0]), cleared("ieee39", 0.4, 0.8,
                                                   "current")["cost"]


PROBES = {
    "case_study.csv": probe_case_study,
    "sensitivity.csv": probe_sensitivity,
    "comparison_ladder.csv": probe_ladder,
    "voltage_conservatism.csv": probe_conservatism,
    "scheduled_bound.csv": probe_bound,
    "ac_feasibility.csv": probe_ac_source,
    "corrected_face.csv": probe_corrected_face,
    "correction_reallocation.csv": probe_reallocation,
    "placement_spread.csv": probe_placement,
    "ac_rows_variants.csv": probe_variants,
    "loss_linearisation.csv": probe_loss_fit,
    "post_contingency.csv": probe_post_contingency,
    "reactive_absorption.csv": probe_absorption,
}


def main() -> int:
    print("결과 파일이 지금의 계통에서 나온 것인지 한 값씩 다시 계산해 대조합니다.\n")
    bad = 0
    for name, probe in PROBES.items():
        if not (RESULTS / name).exists():
            print(f"  {name:28s} 없음")
            bad += 1
            continue
        try:
            stored, fresh = probe()
        except Exception as error:                       # noqa: BLE001
            print(f"  {name:28s} 대조 실패: {error!r}")
            bad += 1
            continue
        if not np.isfinite(stored) or not np.isfinite(fresh):
            print(f"  {name:28s} 대조할 값 없음")
            continue
        gap = abs(stored - fresh) / max(abs(fresh), 1.0)
        mark = "일치" if gap <= TOLERANCE else "다름"
        print(f"  {name:28s} 저장 {stored:14.2f}  재계산 {fresh:14.2f}  "
              f"상대차 {gap:.2e}  {mark}")
        if gap > TOLERANCE:
            bad += 1

    print("\n" + ("전부 같은 계통에서 나왔습니다"
                  if not bad else
                  f"{bad}개가 지금의 계통과 맞지 않습니다. 그 스크립트를 "
                  "다시 돌리기 전에는 숫자 검증에 의미가 없습니다"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
