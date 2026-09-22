#!/usr/bin/env python3
"""Clear, check against the ac network, re-clear -- until the check passes.

The review's first requirement is that the final schedules be ac-feasible, and
that the table which says so be a validation and not a diagnosis. The earlier
loop (run_ac_clearing.py) got close and stopped short, for three reasons that
are each repaired here.

It had no trust region. Each round replaced the schedule outright, so a loss
term or a branch row written about one operating point was applied hundreds of
megawatts away from it. On IEEE 39-bus the loss-charged clearing moved far
enough in one round for the power flow to diverge. Successive linear
programming needs a step bound for exactly this reason: the linearisation is a
local statement. The bound here is on each controlled bus's active and
reactive injection, shrinks geometrically, and is a row like any other, so the
programme is still linear and its duals are still prices -- of the final
subproblem, which is what Section 7 says they are.

It did not carry losses in the balance in the run that fed the table, so the
reference machine invented 33--76 MW after the fact and the "schedule" that was
checked was not the schedule that was cleared. Losses are charged here in every
round, from the same linearisation as the rows.

And its pass/fail looked at bus voltage and branch loading only. The check
below looks at everything the review names and the audit added: every bus,
every branch at the worse of its two ends, every machine's reactive output with
the limits *not* enforced by the solver (so a violation shows as megavars
rather than as a silently switched bus), the reference machine's active output
against its own limits and against the reserve it sold, and each converter's
current recomputed at the voltage the power flow returns.

Branch rows are in apparent power at the sending end and at nominal voltage,
while the rating is a current and the power flow reports the worse end. The gap
is closed adaptively: a branch found above rating has its row tightened by the
measured ratio, which is the standard correction and converges because the
ratio tends to one.

Writes results/ac_feasible.csv (one row per configuration, the final check)
and results/ac_feasible_rounds.csv (the path).
"""
from __future__ import annotations

import copy
import dataclasses
import pathlib
import sys
import time

import numpy as np
import pandapower as pp
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import ac_clearing  # noqa: E402
import ac_linear  # noqa: E402
import ac_rows  # noqa: E402
import prices  # noqa: E402
from multiperiod import (MultiPeriod, ac_voltage_pass,  # noqa: E402
                         build_case, common_demand_peak)
from procurement_lp import Capability  # noqa: E402
import voltage_control  # noqa: E402
from run_ac_clearing import corrected  # noqa: E402


def combined_violation(net, enforce_q_lims: bool = False) -> float:
    """Voltage band and machine reactive limits, as one score to descend.

    The control layer owns both. A setpoint moved to clear a voltage spends
    reactive power at that machine, so correcting voltage with the solver
    enforcing the reactive limits hides the cost: the machine is quietly
    switched to a PQ bus and the voltage it was holding is gone. Here the
    limits are not enforced by the solver; a machine past its capability shows
    up as megavars in the score, and the search lowers its setpoint instead.
    Reactive excess is scaled so that 100 MVAr weighs like 0.01 pu.
    """
    try:
        pp.runpp(net, enforce_q_lims=False, numba=False)
    except Exception:                                    # noqa: BLE001
        return float("inf")
    low = net.bus.min_vm_pu.fillna(0.94).to_numpy()
    high = net.bus.max_vm_pu.fillna(1.06).to_numpy()
    v = net.res_bus.vm_pu.to_numpy()
    score = float(np.sum(np.maximum(low - v, 0.0) ** 2
                         + np.maximum(v - high, 0.0) ** 2))
    for frame, result in ((net.gen, net.res_gen),
                          (net.ext_grid, net.res_ext_grid)):
        if not len(frame) or "max_q_mvar" not in frame:
            continue
        q = result.q_mvar.to_numpy(dtype=float)
        hi = frame.max_q_mvar.to_numpy(dtype=float)
        lo = frame.min_q_mvar.to_numpy(dtype=float)
        over = np.where(np.isfinite(hi), np.maximum(q - hi, 0.0), 0.0)
        under = np.where(np.isfinite(lo), np.maximum(lo - q, 0.0), 0.0)
        score += float(np.sum(((over + under) * 1e-4) ** 2))
    return score


# The corrector in model/voltage_control.py descends whatever `violation`
# measures. Pointing it at the combined score is the whole change; the search
# itself, its step sizes and its least-change rule are untouched.
voltage_control.violation = combined_violation

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
CASES = ("ieee39", "rts24")
SHARES = (0.2, 0.4)
FORMS = ("box", "current")
DISPATCH = 0.8

ROUNDS = 16
STEP_FRACTION = 0.04      # first trust region, as a fraction of peak demand
STEP_SHRINK = 0.7
STEP_FLOOR_MW = 2.0

# What counts as inside a limit. Each is a measurement tolerance, not a
# relaxation: the power flow itself converges to about 1e-6 pu.
TOL_V = 1e-4              # pu
TOL_LOADING = 0.05        # per cent of rating
TOL_Q = 0.5               # MVAr
TOL_BALANCE = 1.0         # MW the reference machine moves from its schedule
TOL_CURRENT = 1e-3        # pu

# How large a residual the clearing is asked to close outright.
Q_REACH = 20.0            # MVAr
V_REACH = 3e-3            # pu


def reference_unit(case) -> int:
    ref = next((j for j, (table, _) in enumerate(case.sync_gen_index)
                if table == "ext_grid"), None)
    return int(np.argmax(case.sync_rating_mva)) if ref is None else ref


def certify(case, model, outcome, controls, settings) -> dict:
    """Every limit, on the schedule as cleared, with nothing enforced for it."""
    networks: list = []
    ac_voltage_pass(case, outcome, solved=networks, controls=controls)
    ref = reference_unit(case)
    table, index = case.sync_gen_index[ref]
    rg = outcome["x"][model.index["rg"]]

    worst = dict(v_under=0.0, v_over=0.0, v_min=np.inf, v_max=-np.inf,
                 loading_max=0.0, q_excess_mvar=0.0, balance_mw=0.0,
                 ref_p_excess_mw=0.0, ref_headroom_short_mw=0.0,
                 conv_current_max=0.0, diverged=0)
    v_conv = []
    for period, net in enumerate(networks):
        try:
            pp.runpp(net, enforce_q_lims=False, numba=False)
        except Exception:                                # noqa: BLE001
            worst["diverged"] += 1
            v_conv.append(np.full(len(case.conv_rating_mva), np.nan))
            continue
        low = net.bus.min_vm_pu.fillna(0.94).to_numpy()
        high = net.bus.max_vm_pu.fillna(1.06).to_numpy()
        v = net.res_bus.vm_pu.to_numpy()
        worst["v_min"] = min(worst["v_min"], float(v.min()))
        worst["v_max"] = max(worst["v_max"], float(v.max()))
        worst["v_under"] = max(worst["v_under"],
                               float(np.maximum(low - v, 0.0).max()))
        worst["v_over"] = max(worst["v_over"],
                              float(np.maximum(v - high, 0.0).max()))

        loading = [float(net.res_line.loading_percent.max())]
        if len(net.res_trafo):
            loading.append(float(net.res_trafo.loading_percent.max()))
        worst["loading_max"] = max(worst["loading_max"], max(loading))

        for frame, result in ((net.gen, net.res_gen),
                              (net.ext_grid, net.res_ext_grid)):
            if not len(frame) or "max_q_mvar" not in frame:
                continue
            q = result.q_mvar.to_numpy(dtype=float)
            hi = frame.max_q_mvar.to_numpy(dtype=float)
            lo = frame.min_q_mvar.to_numpy(dtype=float)
            over = np.where(np.isfinite(hi), q - hi, 0.0)
            under = np.where(np.isfinite(lo), lo - q, 0.0)
            worst["q_excess_mvar"] = max(worst["q_excess_mvar"],
                                         float(np.maximum(over, under).max()),
                                         0.0)

        cleared = float(outcome["pg"][period, ref])
        if table == "ext_grid" and len(net.res_ext_grid):
            actual = float(net.res_ext_grid.p_mw.sum())
        elif index in net.res_gen.index:
            actual = float(net.res_gen.at[index, "p_mw"])
        else:
            actual = cleared
        worst["balance_mw"] = max(worst["balance_mw"], abs(actual - cleared))
        p_max = float(case.sync_pmax_mw[ref])
        p_min = float(case.sync_pmin_mw[ref])
        worst["ref_p_excess_mw"] = max(worst["ref_p_excess_mw"],
                                       actual - p_max, p_min - actual, 0.0)
        worst["ref_headroom_short_mw"] = max(
            worst["ref_headroom_short_mw"],
            actual + float(rg[period, ref]) - p_max, 0.0)
        v_conv.append(np.array([float(net.res_bus.at[int(b), "vm_pu"])
                                for b in case.conv_bus_pp]))

    v_conv = np.array(v_conv)
    injection = model.injection
    net_p = outcome["pd"] - outcome["pc"] - outcome["curt"] + injection
    base = v_conv * case.conv_rating_mva
    with np.errstate(invalid="ignore"):
        before = np.hypot(net_p, outcome["q"]) / base
        after = np.hypot(net_p + outcome["r"], outcome["q"]) / base
    if outcome.get("form") in (None, "current"):
        worst["conv_current_max"] = float(np.nanmax(np.maximum(before, after)))
    worst["conv_current_any_form"] = float(
        np.nanmax(np.maximum(before, after)))

    worst["v_conv"] = v_conv
    worst["passes"] = bool(
        worst["diverged"] == 0
        and worst["v_under"] <= TOL_V and worst["v_over"] <= TOL_V
        and worst["loading_max"] <= 100.0 + TOL_LOADING
        and worst["q_excess_mvar"] <= TOL_Q
        and worst["balance_mw"] <= TOL_BALANCE
        and worst["ref_p_excess_mw"] <= TOL_BALANCE
        and worst["ref_headroom_short_mw"] <= TOL_BALANCE
        # The proposed set claims the bridge current; the comparators do not,
        # and what they do to it is reported rather than required.
        and worst["conv_current_max"] <= settings.i_continuous + TOL_CURRENT)
    return worst


def branch_loading(net) -> np.ndarray:
    parts = [net.res_line.loading_percent.to_numpy(dtype=float)]
    if len(net.res_trafo):
        parts.append(net.res_trafo.loading_percent.to_numpy(dtype=float))
    return np.concatenate(parts)


def trust_rows(inj: ac_rows.Injections, step_p: float, step_q: float):
    """|injection - where it was measured| <= step, per bus, P and Q."""
    rows, rhs = [], []
    for terms, base, step in ((inj.active, inj.p0, step_p),
                              (inj.reactive, inj.q0, step_q)):
        for u, columns in enumerate(terms):
            if not columns:
                continue
            row = np.zeros(inj.size)
            for column, coefficient in columns:
                row[column] += coefficient
            rows.append(row.copy())
            rhs.append(base[u] + step)
            rows.append(-row)
            rhs.append(-(base[u] - step))
    return rows, rhs


def one(name: str, share: float, form: str, settings) -> tuple:
    base = build_case(name, converter_share=share,
                      demand_peak_mw=common_demand_peak(name))
    case = dataclasses.replace(
        base, conv_injection_pu=np.full(len(base.demand_mw), DISPATCH))
    peak = float(np.max(case.demand_mw))
    step = STEP_FRACTION * peak

    path, controls, extra, losses = [], None, None, None
    tighten = None            # per period, per branch, multiplicative
    fallback = None
    best = None
    dc_cost = None
    for index in range(ROUNDS):
        started = time.perf_counter()
        model = MultiPeriod(case, settings, form, extra=extra, losses=losses)
        outcome = model.solve()
        if not outcome["feasible"] and fallback is not None:
            # The rows that push on voltage and reactive output could not be
            # met inside this step; hold them where they are instead.
            extra, fallback = fallback, None
            model = MultiPeriod(case, settings, form, extra=extra,
                                losses=losses)
            outcome = model.solve()
        if not outcome["feasible"]:
            # The step was too tight for the rows to be met from where the
            # schedule stands; widen once rather than stop.
            path.append(dict(case=name, share=share, form=form, round=index,
                             feasible=False, step_mw=step))
            print(f"  r{index:02d} no feasible step at {step:6.1f} MW, widening",
                  flush=True)
            step /= STEP_SHRINK ** 2
            if extra is None or step > STEP_FRACTION * peak * 4:
                break
            # rebuild the trust rows only, about the same point
            t_rows, t_rhs = [], []
            for inj in injections:
                more, more_rhs = trust_rows(inj, step, step)
                t_rows += more
                t_rhs += more_rhs
            extra = (core_rows + t_rows, core_rhs + t_rhs)
            continue
        outcome["form"] = form
        if dc_cost is None:
            dc_cost = outcome["cost"]

        try:
            controls, residual, networks = corrected(case, outcome, controls)
        except Exception as error:                       # noqa: BLE001
            path.append(dict(case=name, share=share, form=form, round=index,
                             feasible=True, converged=False,
                             note=type(error).__name__, step_mw=step))
            step *= 0.5
            if best is None:
                break
            continue

        check = certify(case, model, outcome, controls, settings)
        check_v = check.pop("v_conv")
        path.append(dict(case=name, share=share, form=form, round=index,
                         feasible=True, converged=True, cost=outcome["cost"],
                         step_mw=step, residual_pu2=residual,
                         seconds=time.perf_counter() - started, **check))
        print(f"  r{index:02d} cost {outcome['cost']:>12,.0f} step {step:6.1f} "
              f"V {check['v_min']:.4f}~{check['v_max']:.4f} "
              f"load {check['loading_max']:6.2f}% dQ {check['q_excess_mvar']:6.1f} "
              f"bal {check['balance_mw']:6.2f} "
              f"{time.perf_counter() - started:5.0f}s "
              f"{'PASS' if check['passes'] else ''}", flush=True)
        if check["passes"]:
            best = dict(outcome=outcome, check=check, round=index,
                        controls=controls)
            if step <= max(STEP_FLOOR_MW, 0.002 * peak) or index >= 3:
                break

        # Feed the voltages the power flow returned into the capability set.
        case = dataclasses.replace(case, conv_voltage_pu=check_v)

        # Linearise about this round's corrected networks.
        buses = ac_clearing.controlled_buses(case)
        core_rows, core_rhs, injections, fresh_losses = [], [], [], []
        push_rows, push_rhs, hold_rows, hold_rhs = [], [], [], []
        if tighten is None:
            tighten = [None] * len(networks)
        for period, net in enumerate(networks):
            sens = ac_linear.sensitivities(net, buses)
            inj = ac_clearing.injections_at(model, case, buses, outcome,
                                            period)
            injections.append(inj)
            low, high, q_low, q_high, rating = ac_clearing.limits(net)

            pp.runpp(net, enforce_q_lims=False, numba=False)
            measured = branch_loading(net)
            if tighten[period] is None:
                tighten[period] = np.ones_like(rating)
            implied = np.hypot(sens.point.p_branch, sens.point.q_branch) \
                / np.where(rating > 0, rating, np.inf) * 100.0
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(implied > 1.0, measured / implied, 1.0)
            # how much more current the worse end carries than the sending-end
            # apparent power at nominal voltage suggests
            tighten[period] = np.where(np.isfinite(ratio) & (ratio > 1.0),
                                       1.0 / ratio, 1.0)

            more, more_rhs = ac_rows.branch_rows(
                sens, inj, rating * tighten[period] * (1.0 - 1e-3), 0.0)
            core_rows += more
            core_rhs += more_rhs
            # Voltage and machine reactive output belong to the control
            # layer, which has just been given its turn. The clearing is asked
            # only not to undo that: where a quantity is inside its limit the
            # limit is the row, and where the controls left a residual the row
            # is that residual, so the programme is never infeasible for a
            # reason it has no lever on.
            # Where the controls left a small residual the clearing is asked
            # to close it, because a few megavars or half a millivolt is
            # within reach of converter reactive dispatch; where the residual
            # is large the row asks for half of it, so the programme is not
            # made infeasible by something it has little lever on. If even
            # that cannot be met the round falls back to "no worse".
            q_now = sens.point.q_gen
            v_now = sens.point.v
            q_hi_push = np.where(q_now - q_high <= Q_REACH, q_high,
                                 q_now - 0.5 * (q_now - q_high))
            q_lo_push = np.where(q_low - q_now <= Q_REACH, q_low,
                                 q_now + 0.5 * (q_low - q_now))
            v_hi_push = np.where(v_now - high <= V_REACH, high,
                                 v_now - 0.5 * (v_now - high))
            v_lo_push = np.where(low - v_now <= V_REACH, low,
                                 v_now + 0.5 * (low - v_now))
            for target_rows, target_rhs, ql, qh, vl, vh in (
                    (push_rows, push_rhs, q_lo_push, q_hi_push,
                     v_lo_push, v_hi_push),
                    (hold_rows, hold_rhs, np.minimum(q_low, q_now),
                     np.maximum(q_high, q_now), np.minimum(low, v_now),
                     np.maximum(high, v_now))):
                more, more_rhs = ac_rows.reactive_rows(sens, inj, ql, qh, 0.0)
                target_rows += more
                target_rhs += more_rhs
                more, more_rhs = ac_rows.voltage_rows(sens, inj, vl, vh, 0.0)
                target_rows += more
                target_rhs += more_rhs
            fresh_losses.append(ac_rows.loss_row(sens, inj))
        losses = fresh_losses

        t_rows, t_rhs = [], []
        for inj in injections:
            more, more_rhs = trust_rows(inj, step, step)
            t_rows += more
            t_rhs += more_rhs
        branch_rows_, branch_rhs_ = core_rows, core_rhs
        core_rows, core_rhs = branch_rows_ + push_rows, branch_rhs_ + push_rhs
        fallback = (branch_rows_ + hold_rows + t_rows,
                    branch_rhs_ + hold_rhs + t_rhs)
        extra = (core_rows + t_rows, core_rhs + t_rhs)
        step = max(step * STEP_SHRINK, STEP_FLOOR_MW)

    final = dict(case=name, share=share, form=form, dc_cost=dc_cost)
    if best is not None:
        outcome = best["outcome"]
        final.update(passes=True, rounds=best["round"] + 1,
                     cost=outcome["cost"],
                     cost_change_pct=100.0 * (outcome["cost"] - dc_cost)
                     / dc_cost,
                     price_inertia=prices.price(outcome["prices"], "inertia"),
                     price_reserve=prices.price(outcome["prices"], "reserve"),
                     curtailed_mwh=float(outcome["curt"].sum() * 3.0),
                     reserve_conv_mw=float(outcome["r"].sum()),
                     inertia_conv_mw=float(outcome["h"].sum()),
                     **{k: v for k, v in best["check"].items()
                        if k != "passes"})
    else:
        last = next((p for p in reversed(path) if p.get("converged")), {})
        final.update({k: v for k, v in last.items()
                      if k not in ("case", "share", "form", "round")})
        final.update(passes=False, rounds=len(path))
    return final, path


def main() -> None:
    settings = Capability(i_short_term=1.5, priority="reactive",
                          soc_threshold=0.20)
    targets = sys.argv[1:] or [f"{c}:{s}:{f}" for c in CASES for s in SHARES
                               for f in FORMS]
    finals, paths = [], []
    for target in targets:
        name, share, form = target.split(":")
        print(f"=== {name} share {share} {form}", flush=True)
        final, path = one(name, float(share), form, settings)
        finals.append(final)
        paths += path
        tag = "" if len(targets) == 8 else "_partial"
        pd.DataFrame(finals).to_csv(RESULTS / f"ac_feasible{tag}.csv",
                                    index=False)
        pd.DataFrame(paths).to_csv(RESULTS / f"ac_feasible_rounds{tag}.csv",
                                   index=False)
    print(pd.DataFrame(finals).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
