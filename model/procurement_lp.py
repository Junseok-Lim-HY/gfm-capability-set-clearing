#!/usr/bin/env python3
"""Single-interval prototype, kept only for the Capability dataclass.

The clearing the paper reports is multiperiod.py. This module was the first
version of it and its System/Model pair is no longer exercised by any study
script. It also still carries the inertia requirement in the form that counts
the tripped machine's own stored energy, which multiperiod.py now corrects, so
its numbers should not be quoted. Capability lives here because both models
share it; the rest is history and is retained only so the prototype in the
project record can still be run.
"""


from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pandas as pd
from scipy.optimize import linprog

import capability as cap

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parent / "feasibility" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


@dataclasses.dataclass
class System:
    """One clearing interval."""
    demand_mw: float
    reactive_mvar: float
    reserve_mw: float
    loss_mw: float                 # largest single contingency
    rocof_hz_s: float

    conv_rating_mva: np.ndarray
    conv_cost_energy: np.ndarray
    conv_cost_reserve: np.ndarray
    conv_cost_inertia: np.ndarray
    conv_voltage_pu: np.ndarray
    conv_soc: np.ndarray

    sync_rating_mva: np.ndarray
    sync_cost_energy: np.ndarray
    sync_cost_reserve: np.ndarray
    sync_h_s: np.ndarray
    sync_pmax_mw: np.ndarray
    sync_rmax_mw: np.ndarray


@dataclasses.dataclass
class Capability:
    i_continuous: float = 1.00     # pu of rated current, sustained
    i_short_term: float = 1.50     # pu, for the inertial second
    box_margin: float = 1.00       # what the active-power bound is drawn at
    # Which quantity the active-power bound Pbar is drawn against.
    #
    # "rating" is the default, and the reduction test is why. The capability
    # disc projected onto Q = 0 at V = 1 with no overload is P + R + H <= S,
    # so that is the bound the set must reproduce when the three omissions
    # are removed. Drawing Pbar at the active limit instead makes the
    # baseline a different constraint -- the projection *and* a tighter
    # active limit -- and the two models then differ by two things at once,
    # which is not the comparison this paper claims to make. The active
    # limit is real, so it is imposed on the net injection of both models
    # separately, where it belongs.
    #
    # "pmax" keeps the tighter reading available as a sensitivity. It is the
    # more flattering one here, roughly quadrupling the reported difference,
    # which is a further reason not to make it the default.
    box_bound: str = "rating"
    priority: str = "reactive"     # which current is given up first

    @property
    def reactive_alpha(self) -> float:
        """Fraction of the reactive dispatch held through the excursion.

        The current-priority policy is a converter setting, so it enters the
        clearing as a parameter. Reactive priority holds all of it; ideal
        active priority releases all of it; anything between is a real
        controller. Writing it as a bound and letting the optimiser choose was
        letting the market pick the physics.
        """
        return {"reactive": 1.0, "active": 0.0}.get(self.priority, 1.0)
    soc_threshold: float = 0.20    # below this the overload credit fades
    polygon_sides: int = 24


class Model:
    """Column layout, constraint assembly and solve for one formulation."""

    def __init__(self, system: System, capability: Capability, form: str):
        self.sys = system
        self.cap = capability
        self.form = form
        self.n = len(system.conv_rating_mva)
        self.g = len(system.sync_rating_mva)

        blocks = ["p", "q", "r", "h", "qe"]
        self.index = {}
        cursor = 0
        for name in blocks:
            self.index[name] = np.arange(cursor, cursor + self.n)
            cursor += self.n
        for name in ("pg", "rg"):
            self.index[name] = np.arange(cursor, cursor + self.g)
            cursor += self.g
        self.size = cursor

    # ---------------------------------------------------------------- costs
    def objective(self) -> np.ndarray:
        c = np.zeros(self.size)
        c[self.index["p"]] = self.sys.conv_cost_energy
        c[self.index["r"]] = self.sys.conv_cost_reserve
        c[self.index["h"]] = self.sys.conv_cost_inertia
        c[self.index["pg"]] = self.sys.sync_cost_energy
        c[self.index["rg"]] = self.sys.sync_cost_reserve
        return c

    # ------------------------------------------------------- requirements
    def requirements(self):
        """Equality for balance, inequalities for reserve and inertia.

        Written as -Ax <= -b so that the sign of every dual is comparable.
        """
        a_eq = np.zeros((1, self.size))
        a_eq[0, self.index["p"]] = 1.0
        a_eq[0, self.index["pg"]] = 1.0
        b_eq = np.array([self.sys.demand_mw])

        rows, rhs, names = [], [], []

        row = np.zeros(self.size)
        row[self.index["r"]] = -1.0
        row[self.index["rg"]] = -1.0
        rows.append(row)
        rhs.append(-self.sys.reserve_mw)
        names.append("reserve")

        sync_inertial = cap.inertial_power_from_h(
            self.sys.sync_h_s, self.sys.sync_rating_mva, self.sys.rocof_hz_s
        ).sum()
        row = np.zeros(self.size)
        row[self.index["h"]] = -1.0
        rows.append(row)
        rhs.append(-(self.sys.loss_mw - sync_inertial))
        names.append("inertia")

        row = np.zeros(self.size)
        row[self.index["q"]] = -1.0
        rows.append(row)
        rhs.append(-self.sys.reactive_mvar)
        names.append("reactive")

        return a_eq, b_eq, np.array(rows), np.array(rhs), names

    # -------------------------------------------------------- capability
    def capability_rows(self):
        rows, rhs = [], []
        rating = self.sys.conv_rating_mva
        voltage = self.sys.conv_voltage_pu

        if self.form == "box":
            for i in range(self.n):
                row = np.zeros(self.size)
                row[self.index["p"][i]] = 1.0
                row[self.index["r"][i]] = 1.0
                row[self.index["h"][i]] = 1.0
                rows.append(row)
                rhs.append(self.cap.box_margin * rating[i])
            return np.array(rows), np.array(rhs)

        faces = cap.polygon_rows(self.cap.polygon_sides)
        available = cap.soc_availability(
            self.sys.conv_soc, self.cap.soc_threshold
        )
        for i in range(self.n):
            r_cont = cap.current_radius(
                voltage[i], self.cap.i_continuous, rating[i],
                self.cap.polygon_sides,
            )
            # Only the band above the continuous rating needs the store to
            # support it, so the state of charge scales that band and nothing
            # else. Backing off energy to make room for inertia inside the
            # continuous circle is always available.
            r_short_full = cap.current_radius(
                voltage[i], self.cap.i_short_term, rating[i],
                self.cap.polygon_sides,
            )
            r_short = r_cont + available[i] * max(r_short_full - r_cont, 0.0)
            for nx, ny in faces:
                # sustained services against the continuous current
                row = np.zeros(self.size)
                row[self.index["p"][i]] = nx
                row[self.index["r"][i]] = nx
                row[self.index["q"][i]] = ny
                rows.append(row)
                rhs.append(r_cont)

                # the inertial second against the short-term current, with the
                # reactive output the converter actually holds during the event
                row = np.zeros(self.size)
                row[self.index["p"][i]] = nx
                row[self.index["r"][i]] = nx
                row[self.index["h"][i]] = nx
                row[self.index["qe"][i]] = ny
                rows.append(row)
                rhs.append(r_short)

        # which current is given up first
        for i in range(self.n):
            row = np.zeros(self.size)
            if self.cap.priority == "reactive":
                row[self.index["qe"][i]] = 1.0
                row[self.index["q"][i]] = -1.0
                rows.append(row.copy())
                rhs.append(0.0)
                rows.append(-row)
                rhs.append(0.0)
            else:
                row[self.index["qe"][i]] = 1.0
                row[self.index["q"][i]] = -1.0
                rows.append(row)
                rhs.append(0.0)
        return np.array(rows), np.array(rhs)

    # ------------------------------------------------------------- bounds
    def bounds(self):
        rating = self.sys.conv_rating_mva
        limits = []
        for i in range(self.n):
            limits.append((0.0, rating[i]))                      # p
        for i in range(self.n):
            limits.append((0.0, rating[i]))                      # q
        for i in range(self.n):
            limits.append((0.0, rating[i]))                      # r
        # The bound is deliberately loose in the current-based form, where the
        # polygon carries the limit. In the box form it must be the box, or the
        # baseline would silently inherit a short-term current it does not model.
        h_cap = (self.cap.box_margin if self.form == "box"
                 else self.cap.i_short_term)
        for i in range(self.n):
            limits.append((0.0, h_cap * rating[i]))                   # h
        for i in range(self.n):
            limits.append((0.0, rating[i]))                      # qe
        for j in range(self.g):
            limits.append((0.0, float(self.sys.sync_pmax_mw[j])))
        for j in range(self.g):
            limits.append((0.0, float(self.sys.sync_rmax_mw[j])))
        return limits

    # -------------------------------------------------------------- solve
    def solve(self) -> dict:
        a_eq, b_eq, a_req, b_req, req_names = self.requirements()
        a_cap, b_cap = self.capability_rows()
        a_ub = np.vstack((a_req, a_cap))
        b_ub = np.concatenate((b_req, b_cap))

        result = linprog(
            self.objective(), A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
            bounds=self.bounds(), method="highs",
        )
        if not result.success:
            return {"status": result.message, "feasible": False}

        x = result.x
        # linprog reports marginals for A_ub x <= b_ub; the requirement rows
        # were negated, so negate back to price the requirement itself.
        marginals = result.ineqlin.marginals[: len(req_names)]
        prices = {name: -float(m) for name, m in zip(req_names, marginals)}
        prices["energy"] = float(result.eqlin.marginals[0])

        return {
            "feasible": True,
            "cost": float(result.fun),
            "prices": prices,
            "p": x[self.index["p"]],
            "q": x[self.index["q"]],
            "r": x[self.index["r"]],
            "h": x[self.index["h"]],
            "pg": x[self.index["pg"]],
            "rg": x[self.index["rg"]],
        }


def demo_system(n_conv: int = 4, n_sync: int = 3, seed: int = 0) -> System:
    """A small interval with converters cheap and synchronous units dear.

    The point of the numbers is not realism but separation: the clearing has to
    want converter energy, so that what limits their contribution to reserve
    and inertia is the capability model rather than the merit order.
    """
    rng = np.random.default_rng(seed)
    conv_rating = np.full(n_conv, 200.0)
    sync_rating = np.full(n_sync, 400.0)
    return System(
        demand_mw=1200.0,
        reactive_mvar=260.0,
        reserve_mw=180.0,
        loss_mw=400.0,
        rocof_hz_s=1.0,
        conv_rating_mva=conv_rating,
        conv_cost_energy=np.full(n_conv, 12.0),
        conv_cost_reserve=np.full(n_conv, 4.0),
        conv_cost_inertia=np.full(n_conv, 3.0),
        conv_voltage_pu=np.clip(rng.normal(1.0, 0.02, n_conv), 0.94, 1.05),
        conv_soc=np.array([0.60, 0.35, 0.12, 0.05])[:n_conv],
        sync_rating_mva=sync_rating,
        sync_cost_energy=np.full(n_sync, 38.0),
        sync_cost_reserve=np.full(n_sync, 9.0),
        sync_h_s=np.full(n_sync, 4.0),
        sync_pmax_mw=np.full(n_sync, 380.0),
        sync_rmax_mw=np.full(n_sync, 60.0),
    )


def check_duals(system: System, settings: Capability, form: str) -> None:
    """Perturb each requirement and confirm the dual predicts the change."""
    base = Model(system, settings, form).solve()
    if not base["feasible"]:
        return
    step = 1.0
    for name, field in (("energy", "demand_mw"),
                        ("reserve", "reserve_mw"),
                        ("inertia", "loss_mw")):
        bumped = dataclasses.replace(
            system, **{field: getattr(system, field) + step}
        )
        after = Model(bumped, settings, form).solve()
        if not after["feasible"]:
            print(f"  {name:8s} dual {base['prices'][name]:8.3f}  "
                  f"(perturbed problem infeasible)")
            continue
        empirical = (after["cost"] - base["cost"]) / step
        print(f"  {name:8s} dual {base['prices'][name]:8.3f}  "
              f"observed {empirical:8.3f}  "
              f"{'ok' if abs(empirical - base['prices'][name]) < 1e-6 else 'MISMATCH'}")


def main() -> None:
    system = demo_system()
    print("dual check, current-based model, 1.5 pu short-term current")
    check_duals(system, Capability(i_short_term=1.5), "current")
    print()
    records = []
    for priority in ("reactive", "active"):
        for i_short in (1.0, 1.2, 1.5):
            settings = Capability(i_short_term=i_short, priority=priority)
            for form in ("box", "current"):
                outcome = Model(system, settings, form).solve()
                if not outcome["feasible"]:
                    records.append({
                        "form": form, "priority": priority,
                        "i_short_term": i_short, "feasible": False,
                    })
                    continue
                records.append({
                    "form": form,
                    "priority": priority,
                    "i_short_term": i_short,
                    "feasible": True,
                    "cost": outcome["cost"],
                    "conv_energy_mw": outcome["p"].sum(),
                    "conv_reserve_mw": outcome["r"].sum(),
                    "conv_inertia_mw": outcome["h"].sum(),
                    "sync_energy_mw": outcome["pg"].sum(),
                    "price_energy": outcome["prices"]["energy"],
                    "price_reserve": outcome["prices"]["reserve"],
                    "price_inertia": outcome["prices"]["inertia"],
                })

    frame = pd.DataFrame(records)
    frame.to_csv(RESULTS / "procurement_prototype.csv", index=False)

    print("polygon conservatism at 24 faces: "
          f"{100 * cap.polygon_conservatism(24):.3f} %\n")
    print("clearing under the two capability models\n")
    shown = frame.copy()
    for column in ("cost", "conv_energy_mw", "conv_inertia_mw",
                   "conv_reserve_mw", "price_inertia", "price_reserve",
                   "price_energy"):
        if column not in shown:
            shown[column] = float("nan")
    print(
        shown[[
            "priority", "i_short_term", "form", "feasible", "cost",
            "conv_energy_mw", "conv_inertia_mw", "conv_reserve_mw",
            "price_energy", "price_reserve", "price_inertia",
        ]].to_string(index=False, float_format=lambda v: f"{v:9.2f}")
    )

    print("\nWhere the current-based model is infeasible and the box is\n"
          "not, the box is crediting converters with inertia the current\n"
          "limit cannot deliver. That is the failure the box hides.")

    both = frame[frame.feasible].pivot_table(
        index=["priority", "i_short_term"], columns="form",
        values="price_inertia",
    ).dropna()
    if not both.empty:
        both["overstatement"] = both["box"] / both["current"]
        print("\ninertia price under each capability model")
        print(both.to_string(float_format=lambda v: f"{v:8.2f}"))

    print("\nstate of charge of the four converters:",
          ", ".join(f"{s:.0%}" for s in system.conv_soc))
    print("the overload credit fades below 20 per cent, so the last two")
    print("converters cannot be credited with the full short-term capability.")
    print(f"\nwrote {RESULTS / 'procurement_prototype.csv'}")


if __name__ == "__main__":
    main()
