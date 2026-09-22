#!/usr/bin/env python3
"""Is a cleared schedule actually deliverable on the ac network?

Convergence of the voltage fixed point is not an answer to this. It says the
voltages the clearing assumed are the voltages its own schedule produces, and
nothing more: not that those voltages are inside the case's limits, not that
the machines can supply the reactive power the flow needs, not that every
branch is within its rating, and not how much the slack had to invent to close
a balance the dc model wrote without losses.

So each is tested, with reactive limits enforced on the power flow.

The result has to be read against a baseline, because two of the three
failures are not the clearing's doing. The pandapower cases violate their own
declared bus-voltage columns at their own default operating points --- IEEE
39-bus reaches 1.064 pu against a declared 1.06, RTS-24 spans 0.919 to 1.058
against 0.95 to 1.05 --- so those columns are optimisation bounds rather than
a property of the shipped solution. Reporting a voltage excursion as though
the clearing caused it would be wrong. Branch overload and slack absorption
are different: the base cases are inside their ratings, so anything past 100
per cent belongs to the schedule and to the dc approximation that produced it.
"""
from __future__ import annotations

import dataclasses
import itertools
import pathlib
import sys

import numpy as np
import pandapower as pp
import pandapower.networks as nw
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "model"))

from multiperiod import (PROFILE, ac_feasibility, build_case,  # noqa: E402
                         load_network,
                         common_demand_peak, solve_to_voltage_fixed_point)
from procurement_lp import Capability  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
# The source rows must come through the same repair every other result does.
# Reading pandapower directly here would report the shipped case's voltage
# excursion as the study's baseline, and the study no longer uses that case.
SOURCE = {name: (lambda n=name: load_network(n)) for name in ("ieee39", "rts24")}


def baseline(name: str) -> dict:
    """What the untouched case does against its own declared limits."""
    net = SOURCE[name]()
    pp.runpp(net, numba=False, enforce_q_lims=True)
    return dict(
        case=name, what="source case, unmodified",
        v_min=float(net.res_bus.vm_pu.min()),
        v_max=float(net.res_bus.vm_pu.max()),
        v_limit_lo=float(net.bus.min_vm_pu.min()),
        v_limit_hi=float(net.bus.max_vm_pu.max()),
        loading_max=float(net.res_line.loading_percent.max()),
        # The cleared rows report a correction; the source case has nothing to
        # correct, so this column is left empty there rather than filled with a
        # number that means something else.
        slack_mw=float("nan"),
    )


def main() -> int:
    settings = Capability(i_short_term=1.5, priority="reactive",
                          soc_threshold=0.20)
    rows = [baseline(name) for name in SOURCE]

    for name, share, form in itertools.product(SOURCE, (0.2, 0.4),
                                               ("box", "current")):
        base = build_case(name, converter_share=share,
                          demand_peak_mw=common_demand_peak(name))
        case = dataclasses.replace(
            base, conv_injection_pu=np.full(len(base.demand_mw), 0.8))
        outcome = solve_to_voltage_fixed_point(case, settings, form)
        if not outcome["feasible"]:
            continue
        # Both converter representations, because choosing one would be
        # choosing it after seeing which flatters the schedule. They answer
        # different questions and the difference between them is itself a
        # result: as constant injections the converters hold no bus, and the
        # voltage excursion that produces is largely an artefact of a
        # representation this paper's own premise contradicts.
        for model in ("injection", "voltage"):
            checked = ac_feasibility(case, outcome, converter_model=model)
            rows.append(dict(case=name, share=share, form=form,
                             converter_model=model,
                             what="cleared schedule", **checked))

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "ac_feasibility.csv", index=False)

    columns = ["case", "share", "form", "converter_model", "what",
               "v_min", "v_max", "loading_max", "slack_mw", "q_at_limit",
               "periods_failed", "feasible"]
    print(frame.reindex(columns=columns)
          .to_string(index=False, float_format=lambda v: f"{v:9.3f}"))

    cleared = frame[frame.what == "cleared schedule"]
    passed = int(cleared.feasible.sum()) if len(cleared) else 0
    print(f"\n{passed} of {len(cleared)} cleared schedules satisfy every ac "
          f"limit the case declares.")

    # The source cases breach their own declared voltage columns, so an
    # excursion cannot simply be attributed to the schedule. It cannot simply
    # be attributed to the case either: the amount by which the limit is
    # exceeded grows, and by how much is the number that says so.
    for name in SOURCE:
        source = frame[(frame.case == name) & (frame.what != "cleared schedule")]
        here = cleared[cleared.case == name]
        if not len(source) or not len(here):
            continue
        hi = float(source.v_limit_hi.iloc[0])
        print(f"{name}: overvoltage above the declared column is "
              f"{float(source.v_max.iloc[0]) - hi:+.4f} pu in the source case "
              f"and up to {float(here.v_max.max()) - hi:+.4f} pu when cleared.")
    print("Branch loading past 100 per cent, the slack correction and any "
          "machine at its\nreactive limit are not inherited and belong to the "
          "schedule and the dc approximation.")
    print(f"\nwrote {RESULTS / 'ac_feasibility.csv'} ({len(frame)} rows)")
    # Reports rather than gates: this is a measurement of what the
    # approximation costs, not a test the study is expected to pass.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
