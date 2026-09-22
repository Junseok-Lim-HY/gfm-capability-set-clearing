#!/usr/bin/env python3
"""Does a grid-forming converter actually deliver the inertial power it was
scheduled, inside the current limit the schedule assumed?

The capability set of Section 2 is an envelope written at the operating point
before the disturbance. It says a schedule (P, Q, R, H) is admissible if the
short-term constraint holds at the pre-contingency terminal voltage. What it
does not say is whether a converter running that schedule, given a frequency
event and a current limiter, gets the inertial power out. That is the gap the
review asks to close, and the one the manuscript's own limitations name.

The model is device-level, because the claim is device-level. One converter
against a Thevenin grid whose frequency ramps, a virtual synchronous machine
outer loop with integral reactive control, and a current limiter carrying
the same two limits and the same priority rule the scheduling constraint
uses.

    E angle delta --- j X_f --- V_t --- j X_g --- V_g angle 0
    2 H_v dw/dt = P_ref - P_out
    ddelta/dt   = w_base (w - w_grid(t))
    dE/dt       = k_q (Q_ref - Q_out)
    |I|        <= I_lim(t),  allocated by priority

Three things are deliberate.

The terminal voltage is an outcome, not an input. The converter is a source
behind its filter reactance and the grid is a source behind its own, so V_t is
where they meet. This matters because the assumption under test is precisely
that the pre-contingency V_t stands in for the value during the event; forcing
V_t by hand would assume the answer.

The virtual inertia is not free. H_v is set so that at the design RoCoF the
machine's inertial contribution is exactly the H the clearing scheduled, the
same relation the inertia requirement of Section 4.2 is built on. The
converter is asked for what it sold and nothing more.

The limiter has anti-windup. Once the limit binds, delivered power stops
depending on the angle, so the angle loses its restoring term and becomes a
free integrator; without a hold the machine walks out of synchronism within
the second. That is a real and well documented failure of current-limited
grid-forming control, but it is a property of the limiter, not of the
capability set, and letting it run would answer a different question.

Two questions are asked at every operating point: is the scheduled H
delivered, and is the answer still yes when the grid voltage falls during the
event instead of holding at its pre-contingency value.

What this does not establish, stated because the result is easy to overread.

It is one converter in RMS. There is no network, no fault current, no inner
current loop, no phase-locked loop and no blocking, so it is not a
fault-ride-through study and does not replace an electromagnetic-transient
one.

Half of the headline agreement is definitional. The limiter here enforces the
same inequality the scheduling constraint states, so at a given voltage the
two cannot disagree; that is arithmetic, not evidence. The half that is
evidence is that the converter reaches the boundary at all rather than being
held off it by the angle ceiling, the reactive loop or the swing, and that the
voltage at which it lands is not the one the constraint was written at.
swing_self_check() tests the part of the model the limiter is not involved in.

The rise time is a property of the assumed virtual inertia and damping, not of
the capability set, and is reported separately for that reason rather than
folded into the delivered figure.

The five per cent fall in grid voltage is a stipulated scenario, not a
contingency computed from a network.

Writes results/dynamic_check.csv, results/dynamic_traces.csv and
results/dynamic_self_check.csv.
"""
from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

RESULTS = pathlib.Path(__file__).resolve().parent / "results"

F0 = 60.0              # Hz
W_BASE = 2.0 * np.pi * F0
X_FILTER = 0.05        # pu on converter base, converter to terminal
DAMPING = 20.0         # pu power per pu frequency, VSM damping
GAIN_Q = 20.0          # pu volts per pu reactive power per second
DROOP_V = 0.04         # pu voltage per pu reactive, voltage-droop mode
ROCOF = 1.0            # Hz/s, the design value the schedule was built at
EVENT_AT = 0.5         # s
CONTAINMENT = 1.0      # s, how long the inertial response is asked for
I_CONT = 1.0           # pu, continuous current limit
BAND = 1.0             # s, how long the short-term limit is available
STEP = 1e-4            # s, integration step
# Every run is integrated at STEP; only every THIN-th sample is written,
# and only for the operating point the figure draws. Writing all of them
# at full resolution costs a gigabyte and says nothing the summary does
# not.
THIN = 10
# Two operating points are kept, not one. The first is the figure's active
# power panel and is unchanged. The second carries a large reactive schedule,
# because the current panel is about how the priority rule splits the limit
# between the axes and at a small reactive dispatch there is no split to see.
TRACE_CASES = (dict(v_pre=1.00, q_case="low", i_short=1.1,
                    priority="reactive"),
               dict(v_pre=1.00, q_case="high", i_short=1.1,
                    priority="reactive"))

# The operating points the review asks for.
VOLTAGES = (0.95, 1.00, 1.05)
REACTIVE = {"low": 0.20, "high": 0.60}     # pu of rating, scheduled Q
# The third region of the Proposition is the crossing itself, where the
# two models agree exactly. It is a single reactive value for each
# (voltage, short-term rating) pair and a grid sweep never lands on it, so
# it is placed rather than searched for.
CRITICAL = "critical"
SHORT_TERM = (1.1, 1.5)                    # pu, I^s
PRIORITIES = ("reactive", "active")
P_SCHED = 0.60                             # pu, sustained active power
RESERVE = 0.10                             # pu, held reserve, called at the event
OVERLAP = 1.0                              # chi of Eq. (4)
SAGS = ((0.0, "grid voltage holds"), (0.05, "grid voltage falls 5%"))
# Reactance from the terminal to the Thevenin source. The short-circuit
# ratio is roughly its reciprocal, so these are a stiff bus, an ordinary
# one, and a weak one.
GRIDS = {"strong": 0.05, "ordinary": 0.15, "weak": 0.30}
# How the converter uses its reactive axis. Holding the scheduled value is
# what Eq. (4) assumes; a voltage droop is what much of the fleet actually
# runs, and it spends current on voltage support that the schedule counted
# on for active power.
REACTIVE_MODES = ("hold schedule", "voltage droop")



def short_term_limit(i_short: float, t: float,
                     i_cont: float = I_CONT) -> float:
    """I^c until the event, then the short-term rating for one band."""
    if t < EVENT_AT or t > EVENT_AT + BAND:
        return i_cont
    return i_short


def grid_frequency(t: float) -> float:
    """Per-unit grid frequency: flat, then a ramp down at the design RoCoF."""
    if t < EVENT_AT:
        return 1.0
    return 1.0 - min(t - EVENT_AT, CONTAINMENT) * ROCOF / F0


def allocate(i_d: float, i_q: float, limit: float, priority: str):
    """Cut the current to the limit, keeping whichever axis has priority.

    This is the rule the capability set encodes. Reactive priority holds the
    reactive current and gives up active; active priority does the reverse.
    """
    if float(np.hypot(i_d, i_q)) <= limit:
        return i_d, i_q, False
    if priority == "reactive":
        kept = float(np.clip(i_q, -limit, limit))
        room = float(np.sqrt(max(limit ** 2 - kept ** 2, 0.0)))
        return float(np.sign(i_d) * min(abs(i_d), room)), kept, True
    kept = float(np.clip(i_d, -limit, limit))
    room = float(np.sqrt(max(limit ** 2 - kept ** 2, 0.0)))
    return kept, float(np.sign(i_q) * min(abs(i_q), room)), True


def network(delta: float, e_mag: float, v_grid: float,
            x_grid: float):
    """Current, terminal voltage and power for one converter angle."""
    e_phasor = e_mag * np.exp(1j * delta)
    current = (e_phasor - v_grid) / (1j * (X_FILTER + x_grid))
    v_term = e_phasor - 1j * X_FILTER * current
    apparent = v_term * np.conj(current)
    return current, abs(v_term), float(apparent.real), float(apparent.imag)


def initial_point(p_set: float, q_set: float, v_term: float,
                  x_grid: float):
    """Source magnitudes that put exactly (p_set, q_set) at a terminal of V.

    The terminal voltage is what the schedule is written at, so it is the
    quantity held fixed when the case is set up; E and V_grid follow.
    """
    current = np.conj((p_set + 1j * q_set) / v_term)
    e_phasor = v_term + 1j * X_FILTER * current
    grid_phasor = v_term - 1j * x_grid * current
    return (float(abs(e_phasor)), float(np.angle(e_phasor / grid_phasor)),
            float(abs(grid_phasor)))


def headroom(v: float, q: float, i_short: float) -> float:
    """The H the short-term constraint of Eq. (4) admits at this point."""
    reach = (v * i_short) ** 2 - q ** 2
    if reach <= 0.0:
        return 0.0
    return float(np.sqrt(reach) - P_SCHED - OVERLAP * RESERVE)


def run(v_pre: float, q_set: float, i_short: float, priority: str,
        h_sched: float, sag: float, x_grid: float,
        reactive_mode: str, i_cont: float = I_CONT,
        containment: float = CONTAINMENT,
        p_sched: float = P_SCHED,
        reserve: float = RESERVE) -> pd.DataFrame:
    """One operating point through one event."""
    e_init, delta0, v_grid_pre = initial_point(p_sched, q_set, v_pre,
                                               x_grid)
    h_virtual = h_sched * F0 / (2.0 * ROCOF)

    def frequency(t: float) -> float:
        if t < EVENT_AT:
            return 1.0
        return 1.0 - min(t - EVENT_AT, containment) * ROCOF / F0

    def grid_voltage(t: float) -> float:
        return v_grid_pre if t < EVENT_AT else v_grid_pre * (1.0 - sag)

    def output(t: float, delta: float, e_mag: float) -> dict:
        v_grid = grid_voltage(t)
        _, v_term, p_raw, q_raw = network(delta, e_mag, v_grid, x_grid)
        i_d, i_q = p_raw / v_term, q_raw / v_term
        limit = short_term_limit(i_short, t, i_cont)
        i_d_lim, i_q_lim, _ = allocate(i_d, i_q, limit, priority)
        # At the angle ceiling the current equals the limit exactly
        # and allocate() has nothing to cut, so asking whether it
        # cut under-reports the runs that are actually held at the
        # limit. Ask the magnitude instead.
        clipped = float(np.hypot(i_d_lim, i_q_lim)) >= limit * (1.0 - 1e-9)
        return dict(t=t, v_term=v_term, limit=limit, clipped=clipped,
                    i_mag=float(np.hypot(i_d, i_q)),
                    i_delivered=float(np.hypot(i_d_lim, i_q_lim)),
                    p_out=i_d_lim * v_term, q_out=i_q_lim * v_term,
                    p_unlimited=p_raw)

    def angle_ceiling(t: float, e_mag: float) -> float:
        """The largest angle the current limit permits at this instant."""
        v_grid = grid_voltage(t)
        limit = short_term_limit(i_short, t, i_cont)
        reach = ((X_FILTER + x_grid) * limit) ** 2
        cosine = (e_mag ** 2 + v_grid ** 2 - reach) \
            / (2.0 * e_mag * v_grid)
        return float(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def derivatives(t, state):
        delta, omega, e_mag = state
        shown = output(t, delta, e_mag)
        p_ref = p_sched + (OVERLAP * reserve if t >= EVENT_AT else 0.0)
        slip = omega - frequency(t)
        d_delta = W_BASE * slip
        d_omega = (p_ref - shown["p_out"] - DAMPING * slip) / (2.0 * h_virtual)
        # Integral reactive control, so the converter holds the reactive power
        # it was scheduled rather than drifting off it. A droop would leave a
        # steady-state error, and that error would eat current budget the
        # active response needs -- which would make the run a test of the
        # reactive controller rather than of the capability set.
        if reactive_mode == "voltage droop":
            target = q_set + (v_pre - shown["v_term"]) / DROOP_V
        else:
            target = q_set
        d_e = GAIN_Q * (target - shown["q_out"])
        # Clamping anti-windup, on both loops. At the ceiling, stop advancing
        # the angle and stop the frequency loop charging further into the
        # limit; while the current is saturated, stop integrating reactive
        # error the converter has no current left to answer.
        if delta >= angle_ceiling(t, e_mag) and d_delta > 0.0:
            d_delta, d_omega = 0.0, min(d_omega, 0.0)
        if shown["clipped"] and d_e > 0.0:
            d_e = 0.0
        return [d_delta, d_omega, d_e]

    horizon = EVENT_AT + containment + 0.5
    steps = int(round(horizon / STEP))
    state = np.array([delta0, 1.0, e_init])
    rows = []
    for index in range(steps + 1):
        t = index * STEP
        delta, _, e_mag = state
        rows.append(output(t, min(float(delta), angle_ceiling(t, e_mag)),
                           float(e_mag)))
        if index == steps:
            break
        k1 = np.array(derivatives(t, state))
        k2 = np.array(derivatives(t + STEP / 2, state + STEP / 2 * k1))
        k3 = np.array(derivatives(t + STEP / 2, state + STEP / 2 * k2))
        k4 = np.array(derivatives(t + STEP, state + STEP * k3))
        state = state + STEP / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        # The clamp is a state constraint, not just a rate one: hold the angle
        # at the ceiling rather than letting a stage of the step carry it past.
        state[0] = min(state[0], angle_ceiling(t + STEP, state[2]))
    return pd.DataFrame(rows)


def summarise(trace: pd.DataFrame, h_sched: float, q_sched: float,
              v_pre: float, i_short: float,
              p_sched: float = P_SCHED,
              reserve: float = RESERVE) -> dict:
    """What the run says about the schedule it was given.

    Two different numbers are wanted and the difference matters. The mean over
    the containment window carries the time the outer loop needs to build the
    response, which is a property of the virtual inertia and the damping, not
    of the capability set. The settled value -- the last fifth of the window --
    is what the converter can actually hold, and that is the one the constraint
    is a claim about. Reporting only the mean would charge the capability set
    for a tuning choice.

    Both are returned, because neither answers the other's question and one
    of them stops meaning anything outside this sweep. Here the limiter
    binds in every run and pins the output flat, so the settled value is a
    level. Where nothing pins it the machine rings at a damping ratio near
    0.11 and has not decayed inside the window, and the last fifth is then
    a phase of that ring rather than a level. The mean is what frequency
    containment receives and is well defined either way.
    """
    during = trace[(trace.t >= EVENT_AT) & (trace.t <= EVENT_AT + CONTAINMENT)]
    settling = during[during.t >= EVENT_AT + 0.8 * CONTAINMENT]
    baseline = p_sched + OVERLAP * reserve
    settled = float(settling.p_out.mean() - baseline)
    v_low = float(during.v_term.min())
    v_settled = float(settling.v_term.mean())

    # What Eq. (4) would have admitted had it been written at the voltage that
    # actually obtains rather than the one before the event. Both sides are
    # read over the settling window: comparing a settled power against the
    # lowest voltage of the whole window would compare two different instants,
    # and where the voltage dips and recovers it makes the converter look like
    # it beat its own limit.
    at_actual = headroom(v_settled, float(settling.q_out.mean()), i_short)

    reached = during[during.p_out >= baseline + 0.95 * settled] \
        if settled > 0 else during.iloc[:0]
    engaged = during[during.clipped]
    return dict(
        h_scheduled=h_sched,
        h_settled=settled,
        h_mean=float(during.p_out.mean() - baseline),
        settled_frac=float(settled / h_sched) if h_sched else np.nan,
        mean_frac=(float((during.p_out.mean() - baseline) / h_sched)
                   if h_sched else np.nan),
        h_at_actual_voltage=at_actual,
        against_actual=float(settled / at_actual) if at_actual > 0 else np.nan,
        q_scheduled=q_sched,
        q_settled=float(settling.q_out.mean()),
        v_pre=v_pre,
        v_min=v_low,
        v_settled=v_settled,
        v_settled_drop_pct=100.0 * (v_pre - v_settled) / v_pre,
        v_drop_pct=100.0 * (v_pre - v_low) / v_pre,
        rise_time=float(reached.t.iloc[0] - EVENT_AT) if len(reached) else np.nan,
        peak_current=float(during.i_delivered.max()),
        limiter_engaged=bool(len(engaged)))


def critical_reactive(v_pre: float, i_short: float) -> float:
    """The reactive dispatch at which the two models agree exactly.

    Qtilde_crit = sqrt(L^2 - Pbar^2) with L = V I^s S the short-term reach at
    full band availability and Pbar = S the comparison bound, both per unit of
    rating. Returns nan where L <= Pbar, since the crossing does not exist
    there and the bound overstates at every reactive dispatch.

    It also returns nan where the crossing exists but is not a schedule. The
    capability set has two constraints and the crossing comes out of one of
    them: it is where the short-term face meets the active-power bound, and it
    owes the continuous face nothing. At I^s = 1.5 it asks for 1.02 to 1.22
    per unit of reactive power, which needs 1.30 to 1.34 per unit of current
    before the disturbance arrives, against a continuous limit of 1.0. A
    converter cannot sit there and wait for an event; it is already outside
    its own sustained limit. Placing such a point and calling it an operating
    point puts the run somewhere the schedule could never be.
    """
    reach = (v_pre * i_short) ** 2 - 1.0
    if reach <= 0.0:
        return float("nan")
    critical = float(np.sqrt(reach))
    sustained = float(np.hypot(P_SCHED + RESERVE, critical))
    if sustained > v_pre * I_CONT + 1e-9:
        return float("nan")
    return critical


def proposition_region(v_pre: float, q_set: float, i_short: float) -> str:
    """Which region of the Proposition an operating point sits in.

    The sign of the error the active-power bound makes is fixed by the
    scheduled reactive power against the critical value, so a dynamic study
    that sweeps voltage and reactive dispatch is only representative if the
    points it lands on actually straddle that value. This says where each one
    falls rather than leaving it to be assumed.

    The comparison bound is drawn at the rating, as in the case study, and the
    band is fully available, which is the condition the short-term face is
    written at.
    """
    reach = (v_pre * i_short) ** 2 - 1.0
    if reach <= 0.0:
        return "optimistic"          # L <= Pbar: case (iii)
    critical = float(np.sqrt(reach))
    if abs(q_set - critical) < 1e-9:
        return "crossing"
    return "conservative" if q_set < critical else "optimistic"


def swing_self_check() -> pd.DataFrame:
    """Does the machine produce the inertial power the swing equation says?

    The headline agreement in this file is half definitional: the model's
    limiter enforces the same inequality Eq. (4) states, so at a given voltage
    they cannot disagree. What is not definitional is whether the outer loop
    delivers 2 H_v df/dt in the first place, and that has to be checked where
    the limiter is not involved at all.

    So the limit is lifted out of reach and the settled response is compared
    against the textbook value. H_v was chosen to make that value equal the
    scheduled H, so the target is the schedule itself. If this disagrees, the
    agreement reported everywhere else is an artefact of the limiter and means
    nothing.
    """
    rows = []
    # A long ramp, because the unlimited response is lightly damped and has
    # not settled inside one second; and both limits lifted, because a check
    # that leaves the continuous limit at 1.0 pu measures the limiter again.
    ramp = 15.0
    for x_grid in GRIDS.values():
        for h_sched in (0.10, 0.25, 0.40):
            trace = run(1.00, 0.20, 1.0e3, "reactive", h_sched, 0.0, x_grid,
                        "hold schedule", i_cont=1.0e3, containment=ramp)
            window = trace[(trace.t >= EVENT_AT + 0.8 * ramp)
                           & (trace.t <= EVENT_AT + ramp)]
            delivered = float(window.p_out.mean()
                              - P_SCHED - OVERLAP * RESERVE)
            rows.append(dict(x_grid=x_grid, h_sched=h_sched,
                             delivered=delivered,
                             clipped=bool(trace.clipped.any()),
                             error=delivered - h_sched))
    return pd.DataFrame(rows)


def main() -> None:
    records, traces = [], []
    reactive_cases = list(REACTIVE.items()) + [(CRITICAL, None)]
    for (grid, x_grid), v, (q_name, q), i_short, priority, mode in \
            itertools.product(GRIDS.items(), VOLTAGES, reactive_cases,
                              SHORT_TERM, PRIORITIES, REACTIVE_MODES):
        if q_name == CRITICAL:
            q = critical_reactive(v, i_short)
            if not np.isfinite(q):
                continue
        h_sched = headroom(v, q, i_short)
        if h_sched <= 0.0:
            continue
        for sag, label in SAGS:
            trace = run(v, q, i_short, priority, h_sched, sag, x_grid, mode)
            row = dict(grid=grid, x_grid=x_grid, v_pre=v, q_case=q_name,
                       q_set=q, i_short=i_short, priority=priority,
                       reactive_mode=mode, voltage=label,
                       region=proposition_region(v, q, i_short))
            row.update(summarise(trace, h_sched, q, v, i_short))
            records.append(row)
            here = (v, q_name, i_short, priority)
            if any(here == tuple(case.values()) for case in TRACE_CASES):
                traces.append(trace.iloc[::THIN].assign(
                    grid=grid, v_pre=v, q_case=q_name, i_short=i_short,
                    priority=priority, reactive_mode=mode, voltage=label))

    frame = pd.DataFrame(records)
    frame.to_csv(RESULTS / "dynamic_check.csv", index=False)
    pd.concat(traces).to_csv(RESULTS / "dynamic_traces.csv", index=False)

    def band(values):
        return f"{values.min():6.3f} .. {values.max():6.3f}"

    print("이행률은 정착값 기준입니다. 괄호 안은 실제 전압에서 다시 쓴 "
          "Eq. (4) 대비입니다.\n")
    header = (f"{'grid':>9} {'event':>22} {'settled/sched':>22} "
              f"{'settled/at-actual':>22} {'V drop %':>16} {'rise s':>14}")
    for mode in REACTIVE_MODES:
        print(f"\n[무효 {mode}]")
        print(header)
        chunk = frame[frame.reactive_mode == mode]
        for (grid, label), part in chunk.groupby(["grid", "voltage"],
                                                 sort=False):
            print(f"{grid:>9} {label:>22} {band(part.settled_frac):>22} "
                  f"{band(part.against_actual):>22} "
                  f"{band(part.v_drop_pct):>16} {band(part.rise_time):>14}")

    print("\nProposition 영역별 운전점 수:")
    for region, part in frame.groupby("region"):
        points = part[["v_pre", "q_case", "i_short"]].drop_duplicates()
        print(f"  {region:13s} {len(part):3d}개  "
              f"(설정 {len(points)}가지)")
    for region in ("conservative", "crossing", "optimistic"):
        if region not in set(frame.region):
            print(f"  {region:13s}   0개  -- 이 영역은 비어 있습니다")

    print("\n전류 제한이 걸린 운전점 "
          f"{int(frame.limiter_engaged.sum())}/{len(frame)}, "
          f"최대 전류 {frame.peak_current.max():.4f} pu "
          f"(허용 {max(SHORT_TERM):.2f})")
    worst = frame.loc[frame.settled_frac.idxmin()]
    print(f"최악 운전점: {worst.grid} 계통, {worst.voltage}, "
          f"V={worst.v_pre}, Q={worst.q_case}, I^s={worst.i_short} "
          f"-> 이행률 {worst.settled_frac:.3f}")
    checks = swing_self_check()
    checks.to_csv(RESULTS / "dynamic_self_check.csv", index=False)
    worst = checks.error.abs().max()
    print(f"\n스윙 자체 검사 {len(checks)}개: 전류 제한을 풀면 전달 관성전력이 "
          f"2 H_v df/dt 와 최대 {worst:.2e} pu 차이 "
          f"(제한기 작동 {int(checks.clipped.sum())}회)")
    if worst > 1e-4 or checks.clipped.any():
        raise SystemExit("스윙 구현이 교과서 값과 안 맞습니다. 다른 결과를 "
                         "믿으면 안 됩니다.")

    print("\nwrote results/dynamic_check.csv, results/dynamic_traces.csv, "
          "results/dynamic_self_check.csv")


if __name__ == "__main__":
    sys.exit(main())
