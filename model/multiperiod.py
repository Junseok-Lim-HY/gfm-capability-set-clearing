#!/usr/bin/env python3
"""Multi-period clearing on real test systems, with the store's state carried.

Three things change from the single-interval prototype.

The state of charge becomes a decision rather than a datum, so the overload
credit is coupled to the energy schedule: a converter that has been discharging
all evening cannot be credited in the last hours with a capability its store
cannot support. That coupling is the reason the gate is worth having, and it
cannot appear in a single interval.

The gate stays linear. The availability fraction is an ordinary variable
bounded by one and by the state of charge over the threshold, and it enters the
short-term radius multiplied by a constant, so the clearing remains a linear
programme and the prices remain duals.

And the systems are real. Operating points come from power flow on IEEE 39-bus
and RTS-24 rather than from assumption. Voltage is taken from a power flow at
each load level and held fixed while the clearing runs, which decouples the
capability from the dispatch that produces it; that is a stated limitation, not
an oversight.
"""
from __future__ import annotations

import copy
import dataclasses
import pathlib
import warnings

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
import scipy.sparse as sp
from scipy.optimize import linprog

import capability as cap
import network as grid
import prices

warnings.filterwarnings("ignore")

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parent / "study" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# A day compressed to eight three-hour blocks: night, morning ramp, midday
# surplus, evening peak.
PROFILE = np.array([0.72, 0.68, 0.80, 0.95, 1.00, 0.97, 1.05, 0.85])
HOURS = 3.0
INITIAL_SOC = 0.60
SOC_FLOOR = 0.05
ETA_C = ETA_D = 0.95
STORAGE_HOURS = 4.0
# How much cost the tie-break may give up to pick a single primal. Relative,
# and small enough that it selects among optima rather than among schedules.
TIE_BREAK_SLACK = 1e-9


def store_energy(conv_pmax_mw, hours: float = STORAGE_HOURS) -> np.ndarray:
    """Usable energy of a converter's store, in MWh.

    Duration is defined against the active-power limit, because that is what
    "a four-hour battery" means: four hours of discharge at rated power. It
    must go through this one function. Defining the base case against
    ``conv_pmax_mw`` and a duration sweep against ``conv_rating_mva`` --- which
    is what an earlier version did --- makes the four-hour row of the sweep a
    different machine from the four-hour base case, and the two disagree by
    the power factor.
    """
    return np.asarray(conv_pmax_mw, dtype=float) * float(hours)


class _SparseRows:
    """Inequality rows kept sparse as they are written.

    Each row touches a handful of columns out of a couple of thousand, and at
    192 polygon faces there are over a hundred thousand of them: held dense
    that is several gigabytes once the solver and the tie-break have each
    taken their copy. The rows are still *written* dense, one at a time, which
    keeps the constraint code readable; only what is stored is sparse.
    """

    def __init__(self):
        self.rows = []

    def append(self, row) -> None:
        self.rows.append(sp.csr_matrix(
            np.asarray(row, dtype=float).reshape(1, -1)))

    def __len__(self) -> int:
        return len(self.rows)

    def stack(self, width: int):
        if not self.rows:
            return sp.csr_matrix((0, width))
        return sp.vstack(self.rows, format="csr")


@dataclasses.dataclass
class Case:
    name: str
    demand_mw: np.ndarray          # per period
    reactive_mvar: np.ndarray
    conv_rating_mva: np.ndarray    # apparent power, the radius of the disc
    conv_pmax_mw: np.ndarray       # active limit, which is not the same thing
    conv_voltage_pu: np.ndarray    # period x converter
    conv_energy_mwh: np.ndarray
    sync_rating_mva: np.ndarray
    sync_h_s: np.ndarray
    sync_pmax_mw: np.ndarray
    sync_pmin_mw: np.ndarray
    sync_rmax_mw: np.ndarray
    reserve_frac: float = 0.06
    # Secure the largest outage at the unit's rating rather than
    # at its scheduled output. False recovers the scheduled form.
    outage_at_rating: bool = True
    # Whether the loss of a converter is a credible contingency. It is, once
    # the fleet is large enough for the paper's question to be interesting,
    # and leaving it out would let the study claim a security standard it did
    # not enforce.
    converter_outages: bool = True
    # Whether the reactive requirement is imposed per zone or on the system.
    # Separable from the branch constraints so that the two halves of "adding
    # the network" can be attributed: turning both on at once leaves it
    # unclear whether a cost movement came from congestion or from localising
    # reactive power.
    zonal_reactive: bool = True
    # How long each service must be sustained, in hours. The
    # containment reserve is a half hour; the inertial response is
    # two seconds, which is small but is not zero.
    reserve_hours: float = 0.5
    inertia_hours: float = 2.0 / 3600.0
    # How much of the containment reserve is assumed to have activated by the
    # time the inertial response peaks. One is the worst-case simultaneous
    # envelope and is the default; a real system would put it lower, and the
    # sensitivity study sweeps it because the assumption is not free.
    reserve_overlap: float = 1.0
    reactive_frac: float = 0.35
    rocof_hz_s: float = 1.0
    conv_cost_energy: float = 8.0
    conv_cost_reserve: float = 4.0
    conv_cost_inertia: float = 3.0
    # A tie-breaker, not a market price: without it the clearing is indifferent
    # to reactive output above the requirement and may schedule any amount.
    conv_cost_reactive: float = 0.01
    sync_cost_energy: float = 38.0
    sync_cost_reserve: float = 9.0
    # Throughput cost of the store, charged on energy in and energy out. Zero
    # by default, so the base case is the relaxation the literature uses, and
    # swept in the sensitivity study. Without it nothing discourages a
    # converter from cycling for no reason beyond the round-trip loss.
    conv_cost_cycling: float = 0.0
    # Pre-disturbance converter injection, per unit of rating, one entry
    # per period. Grid-forming converters are mostly inverter-coupled
    # resources whose operating point is set by the resource rather than by
    # arbitrage, and AEMO's account of headroom is written against exactly
    # this quantity. Zero recovers a storage-only fleet.
    conv_injection_pu: np.ndarray = None

    # Network. Absent, the clearing is the single-bus problem it started as;
    # present, branch flows are constrained by dc shift factors and the
    # reactive requirement is imposed per zone rather than on the system.
    network: object = None
    conv_bus: np.ndarray = None      # ppc bus column per converter
    sync_bus: np.ndarray = None
    # The same buses as pandapower numbers them. The columns above index
    # the shift-factor matrix; a power flow needs the original numbers,
    # and deriving one from the other at every call is how they drift.
    conv_bus_pp: np.ndarray = None
    sync_bus_pp: np.ndarray = None
    # Whether a converter may absorb reactive power as well as inject it.
    # The Proposition of Section 3 is written on |Qtilde| and so covers both
    # signs; the clearing has held Q >= 0, which is a narrower domain than the
    # analysis it is meant to illustrate. Absorption is carried as its own
    # non-negative column rather than by dropping the lower bound, because the
    # objective prices reactive power: a negative Q would earn the schedule
    # money and the clearing would absorb until something else stopped it.
    bidirectional_reactive: bool = False
    # Terminal voltage for the short-term face only, when it is not the
    # pre-contingency one. The continuous face is a pre-disturbance
    # constraint and keeps the pre-disturbance voltage; the short-term
    # face is about the second after the outage, and Section 7.3 shows
    # the voltage there is lower. None means the two are the same, which
    # is what the rest of the paper assumes.
    conv_voltage_short_pu: np.ndarray = None
    nodal_p_mw: np.ndarray = None    # period x bus
    nodal_q_mvar: np.ndarray = None
    # enough to rebuild the pandapower case for the ac feasibility pass
    conv_gen_index: list = None
    sync_gen_index: list = None
    load_scale: float = 1.0
    # Synchronous units whose loss must be survivable *through the network*:
    # for each, the clearing must hold a deployment of the reserve and
    # inertial power it bought that replaces the lost infeed without taking
    # any branch past its rating. None leaves the clearing as it was, with
    # the network constraining the pre-contingency dispatch only. Indices are
    # into the synchronous fleet.
    secure_outages: list = None
    # Fraction of each responder's holding deployed for each secured outage,
    # period x outage: min(1, lost / held). Data, because it is a ratio of
    # decisions; the caller iterates it. None means one.
    secure_fraction: np.ndarray = None
    # Branch rating that applies in the seconds after an outage, as a multiple
    # of the continuous rating. Post-contingency flows last as long as the
    # products being deployed -- seconds to a half hour -- and are held against
    # a short-term emergency rating in every security standard the authors
    # know of; holding them to the continuous rating makes the inertial pickup
    # of a large unit's neighbours infeasible on IEEE 39-bus before any
    # capability model is involved.
    secure_rating_factor: float = 1.25


def fleet(net) -> pd.DataFrame:
    """Every generating unit in the case, whichever table it is filed under.

    Three things this has to get right, each of which it got wrong before.

    pandapower's RTS-24 keeps ten units in `gen` and twenty-two in `sgen`, and
    both are generating units; taking only `gen` left out more than half the
    fleet. The slack machine sits in `ext_grid` and is a machine as well: on
    IEEE 39-bus it carries 646 MW, and leaving it out of the clearing while the
    ac pass leaned on it to close the balance was not a modelling choice but an
    omission.

    And `max_p_mw` is an active-power limit, not an apparent-power rating. The
    capability set is a disc of radius V I S with S in MVA, so using MW there
    understates the disc for every unit with reactive capability. Both cases
    carry reactive limits, so the rating is taken as the corner of the box the
    case itself defines, S = sqrt(Pmax^2 + Qmax^2), rather than from an assumed
    power factor.
    """
    rows = []
    for table in ("gen", "sgen", "ext_grid"):
        frame = getattr(net, table)
        if "max_p_mw" not in frame.columns:
            continue
        for index, unit in frame.iterrows():
            p_max = float(unit["max_p_mw"])
            if not np.isfinite(p_max) or p_max <= 0.0:
                continue
            q_max = max(abs(float(unit.get("max_q_mvar", 0.0) or 0.0)),
                        abs(float(unit.get("min_q_mvar", 0.0) or 0.0)))
            declared = float(unit.get("sn_mva", float("nan")) or float("nan"))
            rating = declared if np.isfinite(declared) and declared > 0.0 \
                else float(np.hypot(p_max, q_max))
            p_min = float(unit.get("min_p_mw", 0.0) or 0.0)
            rows.append(dict(table=table, index=int(index),
                             bus=int(unit["bus"]), rating=rating,
                             p_max=p_max, p_min=min(p_min, p_max)))
    return pd.DataFrame(rows).sort_values("rating").reset_index(drop=True)


def common_demand_peak(name: str, shares=None) -> float:
    """Peak demand as a property of the system, not of how its units are split.

    Sizing demand against the synchronous fleet that happens to survive the
    split makes demand a function of the converter share, so a comparison
    across shares moves demand, fleet and largest loss at once and cannot be
    read as a comparison of shares. The peak is therefore eighty per cent of
    the total installed active capacity of the system, which does not depend on
    the labelling at all.

    An earlier version took eighty per cent of the *surviving synchronous*
    capacity. That was share-independent only by construction, and once the
    synchronous minimum stable outputs were imposed it left the system
    structurally over-supplied: on RTS-24 at forty per cent converter share the
    minimum synchronous output plus the pre-disturbance converter injection
    came to \\SI{1938}{MW} against a peak demand of \\SI{1517}{MW}, so several
    hundred megawatts were curtailed in every period, the converters were never
    loaded, and the capability question the study asks did not arise. The
    `shares` argument is accepted and ignored so existing callers keep working.
    """
    net = load_network(name)
    return 0.80 * float(fleet(net).p_max.sum())


def uniform_derating(case, settings) -> float:
    """The largest active fraction of rating that is safe everywhere.

    A uniform derating is the honest version of the active-power bound: keep
    the single active limit, but set it low enough that a converter carrying
    its reactive obligation never asks its bridge for more current than it
    has, at any hour and at any terminal voltage the case produces. That makes
    it a fair comparator rather than a straw man -- it is what a careful
    planner does when the capability set is unavailable.

    The factor is computed from the case, not chosen. At the worst terminal
    voltage and with the reactive requirement met in full, it is what is left
    of the continuous current for active power.
    """
    voltage = float(np.min(case.conv_voltage_pu))
    reactive = derating_reactive_cap(case)
    reach = (voltage * settings.i_continuous) ** 2 - reactive ** 2
    return float(np.sqrt(reach)) if reach > 0.0 else 0.0


def derating_reactive_cap(case) -> float:
    """Per-unit reactive power each converter is limited to under derating.

    The derating is only safe against a stated reactive operating range: a
    unit held to ``gamma S`` of active power and ``kappa S`` of reactive power
    carries at most ``sqrt(gamma^2 + kappa^2) / V`` of current, so choosing
    ``gamma = sqrt((V_min I_c)^2 - kappa^2)`` keeps it inside the continuous
    rating at every voltage of the day. The earlier version put the zonal
    requirement fraction (0.35 of the zone's reactive *load*) in the place of
    ``kappa``. That is a share of demand, not a bound on any unit's ``Q/S``, and
    the clearing was free to schedule more: re-clearing RTS-24 gave continuous
    currents of 1.32 and 1.34 pu under a bound described as safe everywhere.

    ``kappa`` is therefore computed and then enforced. It is the smallest
    uniform cap under which every reactive requirement can still be met ---
    the requirement of the tightest zone at its tightest hour, spread over the
    ratings of the converters in that zone --- which is the choice that leaves
    the comparator the most active power, so the comparison is against the
    strongest derating that is actually safe rather than a weaker one.
    """
    rating = np.asarray(case.conv_rating_mva, dtype=float)
    if case.network is None or not case.zonal_reactive:
        need = float(case.reactive_frac) * float(np.max(case.reactive_mvar))
        worst = need / float(rating.sum())
    else:
        zone_of_conv = case.network.zone_of_bus[case.conv_bus]
        worst = 0.0
        for zone in np.unique(zone_of_conv):
            columns = np.flatnonzero(case.network.zone_of_bus == zone)
            load = np.asarray(case.nodal_q_mvar)[:, columns].sum(axis=1)
            need = float(case.reactive_frac) * float(np.max(load))
            worst = max(worst, need / float(rating[zone_of_conv == zone].sum()))
    # A hair of room, so the binding zone is feasible to solver tolerance.
    return worst * (1.0 + 1e-6)


# The voltage controls that put each shipped case inside its own declared
# limits, from study/base_case_repair.py, which re-derives them and fails if
# they no longer match. Keyed by dataframe index rather than by bus: RTS-24
# carries two transformers on each of buses 10 and 11, and keying by bus would
# collapse two different tap positions into one.
BASE_CASE_REPAIR = {
    "ieee39": {"gen": {5: 1.0586}, "tap": {}},
    "rts24": {"gen": {3: 0.9950, 4: 1.0240, 6: 1.0450,
                      7: 1.0400, 9: 1.0550},
              "tap": {0: -1.0, 1: -2.0, 2: 2.0, 3: -1.0, 4: 3.0}},
}


def load_network(name: str):
    """The test system, with its voltage profile repaired.

    Redispatch is deliberately not part of this. Moving active power would
    change which system is being studied; moving the voltage controls changes
    only how the same dispatch is supported, which is what an operator does
    and what the shipped cases did not. Every result in this study comes
    through here, so no result can rest on an unrepaired network unless all of
    them do.
    """
    net = {"ieee39": nw.case39, "rts24": nw.case24_ieee_rts,
           "ieee118": nw.case118}[name]()
    # The 118-bus case is used for a single scale check under the dc
    # approximation and is taken as shipped.
    repair = BASE_CASE_REPAIR.get(name, {"gen": {}, "tap": {}})
    for index, setpoint in repair["gen"].items():
        net.gen.at[index, "vm_pu"] = setpoint
    for index, position in repair["tap"].items():
        net.trafo.at[index, "tap_pos"] = position
    return net


def build_case(name: str, converter_share: float = 0.20,
               demand_peak_mw: float = None,
               converter_mask: np.ndarray = None) -> Case:
    """Convert a pandapower test system into a clearing case.

    Converters are taken from the smallest units upward until they reach a
    target share of installed capacity. Selecting by share rather than by count
    keeps the two systems comparable, and taking the smallest first avoids
    choosing the units that happen to flatter the result. Units with a zero
    rating are dropped: RTS-24 carries one, and it makes the store dynamics
    divide by zero.

    `converter_mask` overrides that rule with an explicit boolean over the
    fleet in `fleet()` order, so a study can ask whether the rule is doing the
    work. It is the only thing placement changes: everything downstream reads
    the two sides of this mask and nothing reads how it was drawn.
    """
    net = load_network(name)
    units = fleet(net)
    total = float(units.rating.sum())

    if converter_mask is not None:
        mask = np.asarray(converter_mask, dtype=bool)
        if mask.shape != (len(units),):
            raise ValueError(f"mask is {mask.shape}, fleet is {len(units)}")
        is_converter = list(mask)
    else:
        taken, is_converter = 0.0, []
        for rating in units.rating:
            is_converter.append(taken < converter_share * total)
            taken += float(rating)
    units["converter"] = is_converter
    conv = units[units.converter].reset_index(drop=True)
    sync = units[~units.converter].reset_index(drop=True)

    base_p = net.load.p_mw.copy()
    base_q = net.load.q_mvar.copy()
    demands = float(base_p.sum()) * PROFILE
    reactives = float(base_q.sum()) * PROFILE

    conv_rating = conv.rating.to_numpy(dtype=float)
    conv_pmax = conv.p_max.to_numpy(dtype=float)
    sync_rating = sync.rating.to_numpy(dtype=float)
    sync_pmax = sync.p_max.to_numpy(dtype=float)
    sync_pmin = sync.p_min.to_numpy(dtype=float)

    # Demand is sized against the synchronous fleet alone. A storage fleet
    # cannot carry bulk energy across a day, and requiring it to would make
    # every case infeasible for reasons that have nothing to do with
    # capability. Sized this way the converters are optional for energy and are
    # taken for reserve and inertia, which is the question at issue.
    # Demand is a property of the system, not of how its units are labelled.
    # Passing the peak in keeps it identical across converter shares; the
    # fallback reproduces the old share-dependent sizing for a single case.
    target = (float(demand_peak_mw) if demand_peak_mw is not None
              else 0.80 * float(sync_pmax.sum()))
    scale = float(target / demands.max())
    reactives = reactives * scale
    demands = demands * scale

    # Seed voltages, read at the load the case is actually cleared at. Running
    # this before the scale is known would read them off the unscaled profile
    # and hand the clearing terminal voltages belonging to a different system
    # state, which is the sort of error the fixed point then has to unwind.
    # Generation follows the load, or the slack machine is left to absorb the
    # whole difference and the flow does not converge at all.
    voltages = []
    base_gen = {table: getattr(net, table).p_mw.copy()
                for table in ("gen", "sgen") if len(getattr(net, table))}
    for level in PROFILE:
        net.load.p_mw = base_p * level * scale
        net.load.q_mvar = base_q * level * scale
        for table, values in base_gen.items():
            getattr(net, table).p_mw = values * level * scale
        pp.runpp(net, numba=False)
        voltages.append([float(net.res_bus.loc[b, "vm_pu"]) for b in conv.bus])
    net.load.p_mw = base_p
    net.load.q_mvar = base_q
    for table, values in base_gen.items():
        getattr(net, table).p_mw = values

    # Network data. The shift factors and ratings do not depend on the load
    # level, so they are read once; the nodal load does, so it is carried per
    # period at the same scale as the aggregate demand above.
    network = grid.extract(net)
    nodal_p, nodal_q = [], []
    for level in PROFILE:
        p_bus, q_bus = grid.nodal_load(net, level * scale)
        nodal_p.append(p_bus)
        nodal_q.append(q_bus)

    return Case(
        name=name,
        network=network,
        conv_bus=np.array([network.bus_of[b] for b in conv.bus]),
        sync_bus=np.array([network.bus_of[b] for b in sync.bus]),
        conv_bus_pp=np.array(conv.bus, dtype=int),
        sync_bus_pp=np.array(sync.bus, dtype=int),
        nodal_p_mw=np.array(nodal_p),
        nodal_q_mvar=np.array(nodal_q),
        conv_gen_index=list(zip(conv.table, conv["index"])),
        sync_gen_index=list(zip(sync.table, sync["index"])),
        load_scale=float(scale),
        demand_mw=np.asarray(demands, dtype=float),
        reactive_mvar=np.asarray(reactives, dtype=float),
        conv_rating_mva=conv_rating,
        conv_pmax_mw=conv_pmax,
        conv_voltage_pu=np.array(voltages),
        conv_energy_mwh=store_energy(conv_pmax),
        sync_rating_mva=sync_rating,
        sync_h_s=np.full(len(sync_rating), 4.0),
        sync_pmax_mw=sync_pmax,
        sync_pmin_mw=sync_pmin,
        sync_rmax_mw=sync_pmax * 0.15,
    )


class MultiPeriod:
    """Column layout, constraints and solve for one capability formulation."""

    def __init__(self, case: Case, settings, form: str, extra=None,
                 losses=None):
        # `extra` is (rows, rhs) built outside and appended after the
        # requirement rows. After, not before: the requirement count is
        # what tells solve() which multipliers are prices, so inserting
        # ahead of it would quietly reassign every price to another row.
        self.extra = extra
        # `losses` is one (row, constant) per period, a linearisation of the
        # active losses in the decisions. It is not an extra row: the dc
        # balance is simply wrong without it, carrying no losses at all and
        # leaving the slack machine to invent them after the fact. Folding it
        # into the balance keeps the programme linear and the equality count
        # unchanged, and gives the energy price its marginal-loss component,
        # which is where that component belongs.
        self.losses = losses
        self.case = case
        self.cap = settings
        self.form = form
        self.T = len(case.demand_mw)
        self.n = len(case.conv_rating_mva)
        self.g = len(case.sync_rating_mva)

        profile = case.conv_injection_pu
        if profile is None:
            self.injection = np.zeros((self.T, self.n))
        else:
            self.injection = np.outer(np.asarray(profile, dtype=float),
                                      case.conv_rating_mva)

        if case.network is not None:
            self.conv_zone = case.network.zone_of_bus[case.conv_bus]
            self.zones_with_converters = np.unique(self.conv_zone)
            zone_q = np.zeros((self.T, int(case.network.zones.max()) + 1))
            for t in range(self.T):
                for zone in case.network.zones:
                    columns = np.flatnonzero(case.network.zone_of_bus == zone)
                    zone_q[t, int(zone)] = case.nodal_q_mvar[t, columns].sum()
            self.zone_q_mvar = zone_q

        self.index, cursor = {}, 0
        names = ["pd", "pc", "q", "r", "h", "qe", "a", "soc", "curt"]
        if case.bidirectional_reactive:
            names.append("qm")
        for name in names:
            self.index[name] = np.arange(
                cursor, cursor + self.T * self.n
            ).reshape(self.T, self.n)
            cursor += self.T * self.n
        for name in ("pg", "rg"):
            self.index[name] = np.arange(
                cursor, cursor + self.T * self.g
            ).reshape(self.T, self.g)
            cursor += self.T * self.g
        self.outages = list(case.secure_outages or [])
        self.size = cursor

    def objective(self) -> np.ndarray:
        """Cost of the day, in currency units.

        Every offer is quoted per megawatt per hour --- energy in $/MWh,
        reserve and inertial power in $/MW/h of availability --- so each term
        carries the period length. Without it the objective is a sum of
        megawatts, the offers have no units, and the duals are not prices.
        The period length is uniform here, so it multiplies the objective
        through and leaves the schedule alone; that is a property of this
        parameterisation and not a licence to omit it, since a clearing with
        periods of different lengths would weight them differently.
        """
        c = np.zeros(self.size)
        c[self.index["pd"]] = (self.case.conv_cost_energy
                               + self.case.conv_cost_cycling)
        c[self.index["pc"]] = self.case.conv_cost_cycling
        c[self.index["r"]] = self.case.conv_cost_reserve
        c[self.index["h"]] = self.case.conv_cost_inertia
        c[self.index["q"]] = self.case.conv_cost_reactive
        if self.case.bidirectional_reactive:
            # Absorbing costs what injecting costs. Pricing it at the negative
            # of the export term would pay the converter to absorb.
            c[self.index["qm"]] = self.case.conv_cost_reactive
        c[self.index["pg"]] = self.case.sync_cost_energy
        c[self.index["rg"]] = self.case.sync_cost_reserve
        return c * HOURS

    def reactive(self, row, t, i, weight):
        """Put `weight` on converter i's net reactive power in `row`.

        One place, because the net is two columns when absorption is allowed
        and one when it is not, and a constraint that writes only the export
        column is a constraint on the wrong quantity.
        """
        row[self.index["q"][t, i]] += weight
        if self.case.bidirectional_reactive:
            row[self.index["qm"][t, i]] -= weight

    def build(self):
        case, T, n = self.case, self.T, self.n
        eq_rows, eq_rhs = [], []
        ub_rows, ub_rhs, req_names = _SparseRows(), [], []

        for t in range(T):
            row = np.zeros(self.size)
            row[self.index["pd"][t]] = 1.0
            row[self.index["pc"][t]] = -1.0
            row[self.index["pg"][t]] = 1.0
            row[self.index["curt"][t]] = -1.0
            served = case.demand_mw[t] - self.injection[t].sum()
            if self.losses is not None:
                loss_row, constant = self.losses[t]
                row = row - np.asarray(loss_row, dtype=float)
                served = served + float(constant)
            eq_rows.append(row)
            eq_rhs.append(served)

        # store dynamics, round trip split so charging is not free
        eta_c, eta_d = ETA_C, ETA_D
        for t in range(T):
            row = np.zeros(self.size)
            row[self.index["soc"][t]] = 1.0
            if t > 0:
                row[self.index["soc"][t - 1]] = -1.0
            row[self.index["pd"][t]] = HOURS / (eta_d * case.conv_energy_mwh)
            row[self.index["pc"][t]] = -HOURS * eta_c / case.conv_energy_mwh
            eq_rows.append(row)
            eq_rhs.append(INITIAL_SOC if t == 0 else 0.0)

        for i in range(n):
            row = np.zeros(self.size)
            row[self.index["soc"][T - 1, i]] = 1.0
            ub_rows_terminal = row
            eq_rows.append(ub_rows_terminal)
            eq_rhs.append(INITIAL_SOC)

        for t in range(T):
            row = np.zeros(self.size)
            row[self.index["r"][t]] = -1.0
            row[self.index["rg"][t]] = -1.0
            ub_rows.append(row)
            ub_rhs.append(-case.reserve_frac * case.demand_mw[t])
            req_names.append(f"reserve_t{t}")

            # One row per credible outage. The machine that trips is
            # disconnected, so its stored energy leaves the system with it and
            # cannot be counted in the post-fault inertia; summing the whole
            # fleet credited the tripped unit with arresting its own loss.
            #
            # The loss is taken at the unit's rating rather than at its
            # scheduled output. Writing it as the scheduled output is the more
            # faithful statement and stays linear, since that output is a
            # variable, but it lets the clearing escape the requirement by
            # de-loading the machine that would have set it, which is a
            # security standard no operator applies. Taking the rating is the
            # conservative full-output convention; Section 6 reports what the
            # scheduled-output form does instead.
            machine = cap.inertial_power_from_h(
                case.sync_h_s, case.sync_rating_mva, case.rocof_hz_s)
            for w in range(self.g):
                loss = (float(case.sync_pmax_mw[w]) if case.outage_at_rating
                        else None)
                row = np.zeros(self.size)
                row[self.index["h"][t]] = -1.0
                if loss is None:
                    row[self.index["pg"][t, w]] = 1.0
                    rhs = float(machine.sum() - machine[w])
                else:
                    rhs = float(machine.sum() - machine[w] - loss)
                ub_rows.append(row)
                ub_rhs.append(rhs)
                req_names.append(f"inertia_t{t}_w{w}")

            # A converter can be the largest infeed too, and a fleet of them
            # is exactly what this paper is about. The tripped converter's own
            # inertial response leaves with it, so its H is struck from the
            # left-hand side while the whole synchronous inertia stays: no
            # synchronous machine has been lost in this scenario.
            #
            # The loss is the unit's active-power maximum, the same convention
            # the synchronous rows use, and it is exogenous for the same
            # reason. Writing it as the scheduled net infeed instead puts pd,
            # pc and curtailment on the left, which lets the clearing shrink a
            # converter's own contingency by charging or curtailing -- exactly
            # the artefact the synchronous rows avoid by not using the
            # schedule. One security convention, applied to both fleets.
            if case.converter_outages:
                for w in range(n):
                    loss = float(case.conv_pmax_mw[w])
                    if loss <= 0.0:
                        continue
                    row = np.zeros(self.size)
                    row[self.index["h"][t]] = -1.0
                    row[self.index["h"][t, w]] = 0.0
                    ub_rows.append(row)
                    ub_rhs.append(float(machine.sum()) - loss)
                    req_names.append(f"inertia_t{t}_v{w}")

            if case.network is None or not case.zonal_reactive:
                row = np.zeros(self.size)
                for i in range(n):
                    self.reactive(row, t, i, -1.0)
                ub_rows.append(row)
                ub_rhs.append(-case.reactive_frac * case.reactive_mvar[t])
                req_names.append(f"reactive_t{t}")
            else:
                # Reactive deliverability is location-dependent, so a
                # system-wide requirement
                # means little. It is imposed per zone instead, on the zones
                # that hold converters; zones without one have their reactive
                # load served by plant this model does not schedule.
                for zone in self.zones_with_converters:
                    members = np.flatnonzero(self.conv_zone == zone)
                    row = np.zeros(self.size)
                    for i in members:
                        self.reactive(row, t, int(i), -1.0)
                    ub_rows.append(row)
                    ub_rhs.append(-case.reactive_frac
                                  * self.zone_q_mvar[t, int(zone)])
                    req_names.append(f"reactive_t{t}_z{zone:g}")

        n_req = len(ub_rows)

        # Energy and reserve come out of the same machine, so they share its
        # rating. Bounding them separately let a unit at full output sell
        # reserve it had no way to deliver.
        for t in range(T):
            for j in range(self.g):
                row = np.zeros(self.size)
                row[self.index["pg"][t, j]] = 1.0
                row[self.index["rg"][t, j]] = 1.0
                ub_rows.append(row)
                ub_rhs.append(float(case.sync_pmax_mw[j]))

        # The converter's net active injection, bounded by its active limit.
        #
        # Bounding the discharge and the charge separately does not do this.
        # The quantity that leaves the terminals is e + pd - pc - c, and with
        # the pre-disturbance injection e set as a fraction of the apparent
        # rating it can exceed the active limit on its own: measured at 1.151
        # times Pmax on RTS-24 before this constraint existed. Both
        # formulations then compared curtailment at different physical active
        # limits, which is not a comparison of capability models at all.
        #
        # Two-sided, because a converter with a store can absorb as well as
        # inject and the limit is symmetric.
        for t in range(T):
            for i in range(n):
                row = np.zeros(self.size)
                row[self.index["pd"][t, i]] = 1.0
                row[self.index["pc"][t, i]] = -1.0
                row[self.index["curt"][t, i]] = -1.0
                ub_rows.append(row)
                ub_rhs.append(float(case.conv_pmax_mw[i])
                              - self.injection[t, i])
                ub_rows.append(-row)
                ub_rhs.append(float(case.conv_pmax_mw[i])
                              + self.injection[t, i])

        # A reserve that is called must be delivered for the containment
        # period, and an inertial response for its own second or two. Both
        # come out of the store, so the energy above the floor has to cover
        # them. Without this a converter sitting at its floor could sell a
        # half-hour reserve it had no energy for.
        #
        # Both ends of the interval, for the same reason the gate above tests
        # both: a contingency can arrive in the last minute of a three-hour
        # block, by which time a converter that has been discharging all
        # interval no longer has the energy its opening state implied. Two
        # rows are the linear way to write min(sigma_open, sigma_close).
        for t in range(T):
            for i in range(n):
                for endpoint in ("start", "end"):
                    row = np.zeros(self.size)
                    row[self.index["r"][t, i]] = case.reserve_hours / ETA_D
                    row[self.index["h"][t, i]] = case.inertia_hours / ETA_D
                    rhs = -SOC_FLOOR * case.conv_energy_mwh[i]
                    if endpoint == "end" or t > 0:
                        index = t if endpoint == "end" else t - 1
                        row[self.index["soc"][index, i]] = \
                            -case.conv_energy_mwh[i]
                    else:
                        rhs += INITIAL_SOC * case.conv_energy_mwh[i]
                    ub_rows.append(row)
                    ub_rhs.append(rhs)

        # Branch flows. The clearing may not schedule a dispatch the network
        # cannot carry. These are ordinary inequalities and not requirements,
        # so they sit after the count that separates priced rows from the rest.
        if case.network is not None:
            ptdf, limit = case.network.ptdf, case.network.limit_mw
            for t in range(T):
                fixed = np.zeros(ptdf.shape[1])
                np.add.at(fixed, case.conv_bus, self.injection[t])
                fixed -= case.nodal_p_mw[t]
                base = ptdf @ fixed
                for b in range(ptdf.shape[0]):
                    row = np.zeros(self.size)
                    row[self.index["pd"][t]] = ptdf[b, case.conv_bus]
                    row[self.index["pc"][t]] = -ptdf[b, case.conv_bus]
                    row[self.index["curt"][t]] = -ptdf[b, case.conv_bus]
                    row[self.index["pg"][t]] = ptdf[b, case.sync_bus]
                    ub_rows.append(row)
                    ub_rhs.append(limit[b] - base[b])
                    ub_rows.append(-row)
                    ub_rhs.append(limit[b] + base[b])

        if self.form in ("box", "derated"):
            # Two readings of a single active limit. The box draws it at the
            # rating; the uniform derating draws it where a converter meeting
            # its reactive obligation would still be inside its current limit
            # at the worst voltage of the day, which is the same constraint
            # set honestly rather than optimistically.
            margin = (self.cap.box_margin if self.form == "box"
                      else uniform_derating(case, self.cap))
            for t in range(T):
                for i in range(n):
                    row = np.zeros(self.size)
                    row[self.index["pd"][t, i]] = 1.0
                    row[self.index["pc"][t, i]] = -1.0
                    row[self.index["curt"][t, i]] = -1.0
                    row[self.index["r"][t, i]] = 1.0
                    row[self.index["h"][t, i]] = 1.0
                    reference = (case.conv_pmax_mw[i]
                                 if self.cap.box_bound == "pmax"
                                 else case.conv_rating_mva[i])
                    ub_rows.append(row)
                    ub_rhs.append(margin * reference - self.injection[t, i])
                    if self.form == "derated":
                        # The derating is a current limit in disguise, so it
                        # binds a charging unit as much as a discharging one:
                        # the net absorption may not exceed the same derated
                        # level. Without this row the comparator called safe
                        # could charge at its full active limit while carrying
                        # its reactive obligation.
                        low = np.zeros(self.size)
                        low[self.index["pd"][t, i]] = -1.0
                        low[self.index["pc"][t, i]] = 1.0
                        low[self.index["curt"][t, i]] = 1.0
                        ub_rows.append(low)
                        ub_rhs.append(margin * reference
                                      + self.injection[t, i])
        elif self.form == "apparent":
            # Reactive power counted, terminal voltage and short-term rating
            # not. One disc at the rating, so no overload band and no state of
            # charge gating it; and the reactive coordinate is the scheduled
            # one, because the priority fraction is physics the proposed model
            # brings and this model does not have it.
            faces = cap.polygon_rows(self.cap.polygon_sides)
            for t in range(T):
                for i in range(n):
                    radius = cap.current_radius(1.0, 1.0,
                                                case.conv_rating_mva[i],
                                                self.cap.polygon_sides)
                    for nx, ny in faces:
                        row = np.zeros(self.size)
                        row[self.index["pd"][t, i]] = nx
                        row[self.index["pc"][t, i]] = -nx
                        row[self.index["curt"][t, i]] = -nx
                        row[self.index["r"][t, i]] = nx
                        row[self.index["h"][t, i]] = nx
                        self.reactive(row, t, i, ny)
                        ub_rows.append(row)
                        ub_rhs.append(radius - nx * self.injection[t, i])
                        # The pre-activation point, for the same reason as in
                        # the proposed form below: a charging unit carries its
                        # largest current before its reserve is called.
                        if nx < 0.0:
                            row = np.zeros(self.size)
                            row[self.index["pd"][t, i]] = nx
                            row[self.index["pc"][t, i]] = -nx
                            row[self.index["curt"][t, i]] = -nx
                            self.reactive(row, t, i, ny)
                            ub_rows.append(row)
                            ub_rhs.append(radius - nx * self.injection[t, i])
        else:
            # "cone" is "current" with the polygon faces withheld, so that a
            # caller can put exact second-order cones on the same variables
            # and the two problems differ in the capability representation and
            # in nothing else. Everything below the faces -- the state of
            # charge gate on the overload band and the reactive priority
            # equality -- belongs to both and is emitted for both.
            faces = ([] if self.form == "cone"
                     else cap.polygon_rows(self.cap.polygon_sides))
            for t in range(T):
                for i in range(n):
                    v = case.conv_voltage_pu[t, i]
                    s = case.conv_rating_mva[i]
                    r_cont = cap.current_radius(
                        v, self.cap.i_continuous, s, self.cap.polygon_sides)
                    v_short = v
                    if case.conv_voltage_short_pu is not None:
                        v_short = case.conv_voltage_short_pu[t, i]
                    r_full = cap.current_radius(
                        v_short, self.cap.i_short_term, s,
                        self.cap.polygon_sides)
                    # The short-term face is the capability in the second
                    # after the outage, so both of its ends are taken at the
                    # voltage of that second: the radius is
                    # v_short * (Ic + a (Is - Ic)) * S, which is Eq. (27) read
                    # literally. An earlier version anchored the face at the
                    # pre-contingency continuous radius and added
                    # a * (v_short Is - v Ic) S, which agrees at a = 1 only and
                    # overstates the face for every partly gated store; it also
                    # clipped the band at zero, hiding a voltage drop deep
                    # enough to take the short-term radius below the continuous
                    # one. With no separate short-term voltage the two readings
                    # coincide.
                    r_short_base = cap.current_radius(
                        v_short, self.cap.i_continuous, s,
                        self.cap.polygon_sides)
                    band = r_full - r_short_base
                    for nx, ny in faces:
                        row = np.zeros(self.size)
                        row[self.index["pd"][t, i]] = nx
                        row[self.index["pc"][t, i]] = -nx
                        row[self.index["curt"][t, i]] = -nx
                        row[self.index["r"][t, i]] = nx
                        self.reactive(row, t, i, ny)
                        ub_rows.append(row)
                        ub_rhs.append(r_cont - nx * self.injection[t, i])

                        # The same face at the operating point before any
                        # reserve is called. With P >= 0 the activated point
                        # (P + R, Q) is the larger current and the row above
                        # covers it, but a charging converter moves towards
                        # the origin when its reserve activates, so its
                        # largest current is the one it carries beforehand.
                        # Both ends of the activation segment are inside a
                        # convex set, hence so is everything between them.
                        # Only faces with a negative active normal can bind
                        # here, since r >= 0.
                        if nx < 0.0:
                            row = np.zeros(self.size)
                            row[self.index["pd"][t, i]] = nx
                            row[self.index["pc"][t, i]] = -nx
                            row[self.index["curt"][t, i]] = -nx
                            self.reactive(row, t, i, ny)
                            ub_rows.append(row)
                            ub_rhs.append(r_cont - nx * self.injection[t, i])

                        # The short-term face carries the inertial response
                        # in full and the reserve scaled by an overlap
                        # factor. Charging the whole reserve against a
                        # two-second envelope assumes containment reserve is
                        # fully activated within two seconds, which nothing
                        # here establishes; charging none of it assumes the
                        # two never coincide, which is worse. The default is
                        # one, the worst-case simultaneous-activation
                        # envelope, and it is swept.
                        row = np.zeros(self.size)
                        row[self.index["pd"][t, i]] = nx
                        row[self.index["pc"][t, i]] = -nx
                        row[self.index["curt"][t, i]] = -nx
                        row[self.index["r"][t, i]] = nx * case.reserve_overlap
                        row[self.index["h"][t, i]] = nx
                        row[self.index["qe"][t, i]] = ny
                        row[self.index["a"][t, i]] = -band
                        ub_rows.append(row)
                        ub_rhs.append(r_short_base
                                      - nx * self.injection[t, i])

                    # Availability of the overload band, bounded by the store
                    # at both ends of the interval. Gating on the closing
                    # state alone credits a converter that spends the interval
                    # emptying itself with the band it had at the start;
                    # gating on both is the conservative reading and stays
                    # linear, since a <= x and a <= y give a <= min.
                    #
                    # The fraction is measured from the energy floor, not from
                    # zero. Writing it as sigma / sigma_star credits a quarter
                    # of the band to a store sitting at a floor of 0.05 with
                    # no usable energy at all, which is the opposite of what
                    # the gate is for. Two thresholds, not one: SOC_FLOOR is
                    # where usable energy runs out and soc_threshold is where
                    # the band is fully available.
                    #
                    # With the two equal there is no gate, and the rows must be
                    # omitted rather than written with a zero denominator.
                    if self.cap.soc_threshold > SOC_FLOOR:
                        span = self.cap.soc_threshold - SOC_FLOOR
                        for endpoint in ("start", "end"):
                            row = np.zeros(self.size)
                            row[self.index["a"][t, i]] = 1.0
                            rhs = -SOC_FLOOR / span
                            if endpoint == "end" or t > 0:
                                index = t if endpoint == "end" else t - 1
                                row[self.index["soc"][index, i]] = -1.0 / span
                            else:
                                rhs += INITIAL_SOC / span
                            ub_rows.append(row)
                            ub_rhs.append(rhs)

                    # The current-priority policy is a converter setting, so
                    # it is a parameter and not something the clearing picks.
                    # Leaving it as 0 <= Qtilde <= Q let the optimiser choose
                    # the physics, and it always chose the convenient one.
                    row = np.zeros(self.size)
                    row[self.index["qe"][t, i]] = 1.0
                    self.reactive(row, t, i, -self.cap.reactive_alpha)
                    eq_rows.append(row)
                    eq_rhs.append(0.0)

        if self.outages and case.network is not None:
            # Post-contingency deliverability, for the secured outages.
            #
            # What stops flowing is the tripped unit's scheduled output. What
            # replaces it, in the seconds the products are bought for, is the
            # response the schedule holds -- reserve and inertial power at the
            # converters, reserve at the other synchronous units -- deployed
            # pro rata and no further than the loss, and whatever that leaves
            # uncovered is picked up inertially by the surviving synchronous
            # machines in proportion to their stored energy, which is what
            # they do whether or not anybody paid them to.
            #
            # The pro-rata fraction is a ratio of decisions, so it is carried
            # as data, ``secure_fraction``, taken from the previous solve and
            # iterated by the caller; with it fixed every row below is linear
            # and sits after the requirement rows like any other.
            ptdf, limit = case.network.ptdf, case.network.limit_mw
            stored = np.asarray(case.sync_h_s) * np.asarray(case.sync_rating_mva)
            fraction = (np.ones((T, len(self.outages)))
                        if case.secure_fraction is None
                        else np.asarray(case.secure_fraction, dtype=float))
            for t in range(T):
                fixed = np.zeros(ptdf.shape[1])
                np.add.at(fixed, case.conv_bus, self.injection[t])
                fixed -= case.nodal_p_mw[t]
                base = ptdf @ fixed
                for k, w in enumerate(self.outages):
                    phi = float(fraction[t, k])
                    share = stored.copy()
                    share[w] = 0.0
                    share = share / share.sum()
                    pickup = ptdf[:, case.sync_bus] @ share   # per MW uncovered
                    lost_col = ptdf[:, case.sync_bus[w]]
                    for b in range(ptdf.shape[0]):
                        row = np.zeros(self.size)
                        row[self.index["pd"][t]] = ptdf[b, case.conv_bus]
                        row[self.index["pc"][t]] = -ptdf[b, case.conv_bus]
                        row[self.index["curt"][t]] = -ptdf[b, case.conv_bus]
                        row[self.index["pg"][t]] = ptdf[b, case.sync_bus]
                        # the lost infeed leaves, and reappears as pickup
                        row[self.index["pg"][t, w]] += pickup[b] - lost_col[b]
                        # each MW deployed arrives at its own bus instead of
                        # through the inertial pickup
                        conv_gain = phi * (ptdf[b, case.conv_bus] - pickup[b])
                        row[self.index["r"][t]] += conv_gain
                        row[self.index["h"][t]] += conv_gain
                        sync_gain = phi * (ptdf[b, case.sync_bus] - pickup[b])
                        sync_gain[w] = 0.0
                        row[self.index["rg"][t]] += sync_gain
                        short = case.secure_rating_factor * limit[b]
                        ub_rows.append(row)
                        ub_rhs.append(short - base[b])
                        ub_rows.append(-row)
                        ub_rhs.append(short + base[b])

        if self.extra is not None:
            more_rows, more_rhs = self.extra
            for extra_row in more_rows:
                ub_rows.append(extra_row)
            ub_rhs = list(ub_rhs) + list(more_rhs)

        return (np.array(eq_rows), np.array(eq_rhs),
                ub_rows.stack(self.size), np.array(ub_rhs), req_names, n_req)

    def bounds(self):
        case = self.case
        limits = [None] * self.size
        q_cap = (derating_reactive_cap(case) if self.form == "derated"
                 else None)
        for t in range(self.T):
            for i in range(self.n):
                s = float(case.conv_rating_mva[i])
                p_max = float(case.conv_pmax_mw[i])
                limits[self.index["pd"][t, i]] = (0.0, p_max)
                limits[self.index["pc"][t, i]] = (0.0, p_max)
                # Under the uniform derating the reactive range is part of the
                # safety argument, so it is a bound and not an expectation.
                q_top = q_cap * s if q_cap is not None else s
                limits[self.index["q"][t, i]] = (0.0, q_top)
                if case.bidirectional_reactive:
                    limits[self.index["qm"][t, i]] = (0.0, q_top)
                limits[self.index["r"][t, i]] = (0.0, s)
                cap_h = {"box": self.cap.box_margin,
                         "derated": uniform_derating(case, self.cap),
                         "apparent": 1.0}.get(self.form,
                                              self.cap.i_short_term)
                limits[self.index["h"][t, i]] = (0.0, cap_h * s)
                limits[self.index["qe"][t, i]] = (0.0, s)
                limits[self.index["a"][t, i]] = (0.0, 1.0)
                limits[self.index["soc"][t, i]] = (SOC_FLOOR, 0.95)
                # Curtailment cannot exceed what was being injected. It costs
                # nothing directly: spilling a megawatt forces the balance to
                # replace it from synchronous plant at that plant's offer, so
                # the cost is already carried there.
                limits[self.index["curt"][t, i]] = (0.0, self.injection[t, i])
            for j in range(self.g):
                # Commitment is fixed, and a committed machine runs at or
                # above its minimum stable output. Letting it fall to zero
                # while its inertia is still counted in Eq. (inertia) is the
                # same error as counting the tripped machine: the stored
                # energy is only there if the machine is synchronised.
                limits[self.index["pg"][t, j]] = (float(case.sync_pmin_mw[j]),
                                                  float(case.sync_pmax_mw[j]))
                limits[self.index["rg"][t, j]] = (0.0, float(case.sync_rmax_mw[j]))
        return limits

    def _tie_objective(self) -> np.ndarray:
        """Secondary objective that picks one optimum out of the many.

        The offers are illustrative and identical across units of a kind, so
        the optimal face is large: any redistribution of energy among
        synchronous machines, or of reserve among converters, costs exactly
        the same. On IEEE 39-bus, synchronous output moves by more than
        900 MW between rounds of the voltage iteration at a cost identical to
        ten decimal places.

        The rule has to be physical, not convenient. It is storage
        throughput: among schedules at the optimal cost, take the one that
        cycles the stores least. Cycling degrades a battery and nothing in
        the objective charges for it, so of two schedules that cost the same
        an operator would run the one that uses the asset less; it is also
        the quantity the cycling-cost sensitivity varies, so the base case
        and that sweep agree on what they are trading off. A small
        index-ordered term breaks any remaining tie, which is arbitrary but
        only ever chooses between schedules already equal in both cost and
        throughput.

        Prices come from the first solve and are untouched. This selects
        among primals the cost minimisation was indifferent between; where
        that indifference is wide, the paper reports the interval rather than
        this point, because a rule cannot manufacture a result the data does
        not identify.
        """
        c = np.zeros(self.size)
        c[self.index["pd"]] = 1.0
        c[self.index["pc"]] = 1.0
        return c + 1e-6 * (1.0 + np.arange(self.size, dtype=float) / self.size)

    def solve(self, perturb: tuple = None, tie_break: bool = True) -> dict:
        """Clear the market. ``perturb`` raises one requirement for a dual test.

        ``perturb=(name, delta)`` raises the named requirement by ``delta``
        and returns the cost of the re-cleared problem, which is what a
        finite-difference check of the corresponding dual needs. Requirements
        are written ``-sum(...) <= -level``, so raising the level lowers the
        right-hand side; the energy balance is an equality in demand and moves
        the other way.

        ``tie_break`` selects a single primal among the alternative optima. It
        is on by default: without it the schedule a run reports depends on
        which vertex the solver happened to stop at, so the same code on a
        different machine can report a different curtailment at the same cost.
        Turning it off is only useful for measuring how wide that indifference
        is, which is what the optimal-face range in study/run_case_study.py
        does.
        """
        a_eq, b_eq, a_ub, b_ub, req_names, n_req = self.build()
        if perturb is not None:
            name, delta = perturb
            if name.startswith("energy_t"):
                b_eq = np.array(b_eq, dtype=float)
                b_eq[int(name.rsplit("_t", 1)[1])] += delta
            else:
                b_ub = np.array(b_ub, dtype=float)
                b_ub[req_names.index(name)] -= delta
        cost = self.objective()
        bounds = self.bounds()
        result = linprog(cost, A_ub=a_ub, b_ub=b_ub,
                         A_eq=a_eq, b_eq=b_eq, bounds=bounds,
                         method="highs-ds")
        if not result.success:
            return {"feasible": False, "message": result.message}
        x = result.x
        marginals = result.ineqlin.marginals[:n_req]
        prices = {name: -float(m) for name, m in zip(req_names, marginals)}

        if tie_break:
            # The clearing has alternative optima that differ in reactive
            # dispatch at equal cost, which leaves the map from assumed
            # voltage to resulting voltage set-valued and the fixed point
            # without anything to converge to. A second pass picks one of them
            # by a rule: among schedules costing no more than the optimum,
            # minimise a functional whose weights differ on every column, so
            # that the selection is unique. See _tie_objective.
            #
            # Prices stay those of the first solve. They are properties of the
            # cost minimisation; the second pass only selects among primals
            # the first pass was indifferent between, and its own duals would
            # be duals of a different problem.
            cap_row = np.zeros((1, self.size))
            cap_row[0] = cost
            slack = TIE_BREAK_SLACK * max(abs(float(result.fun)), 1.0)
            second = linprog(
                self._tie_objective(),
                A_ub=sp.vstack([a_ub, sp.csr_matrix(cap_row)], format="csr"),
                b_ub=np.concatenate([b_ub, [float(result.fun) + slack]]),
                A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs-ds")
            if second.success:
                x = second.x
        prices.update({
            f"energy_t{t}": float(result.eqlin.marginals[t])
            for t in range(self.T)
        })
        # The LP variable a is only pushed to its bound when the band it
        # unlocks is worth something, so reading it as a state of charge is
        # wrong wherever the band is slack: it reports zero availability for a
        # full store that simply was not asked for an excursion. The physical
        # quantity is the gate itself, evaluated on the cleared trajectory.
        soc = x[self.index["soc"]]
        opening = np.vstack([np.full((1, self.n), INITIAL_SOC), soc[:-1]])
        if self.cap.soc_threshold > SOC_FLOOR:
            span = self.cap.soc_threshold - SOC_FLOOR
            availability = np.clip(
                (np.minimum(opening, soc) - SOC_FLOOR) / span, 0.0, 1.0)
        else:
            availability = np.ones_like(soc)
        return {
            "feasible": True,
            "cost": float(result.fun),
            "prices": prices,
            "soc": soc,
            "availability": availability,
            "a": x[self.index["a"]],
            "h": x[self.index["h"]],
            "r": x[self.index["r"]],
            "pd": x[self.index["pd"]],
            "pc": x[self.index["pc"]],
            "q": (x[self.index["q"]] - x[self.index["qm"]]
                  if self.case.bidirectional_reactive
                  else x[self.index["q"]]),
            "q_export": x[self.index["q"]],
            "q_absorb": (x[self.index["qm"]]
                         if self.case.bidirectional_reactive
                         else np.zeros_like(x[self.index["q"]])),
            "pg": x[self.index["pg"]],
            "curt": x[self.index["curt"]],
            # The full primal, so a caller can measure how far the schedule
            # itself moved rather than only how far its cost did.
            "x": x,
        }


def optimal_face_range(case: Case, settings, form: str,
                       quantity: str = "curt") -> tuple:
    """How far a reported primal can move without changing the cost.

    A tie-break makes a run reproducible; it does not make the quantity it
    selects a result. If the cost-optimal face is wide in curtailment, then
    the curtailment a paper prints is a property of the rule that picked the
    vertex, not of the capability model, and reporting it as a finding would
    be reporting an arbitrary choice.

    So it is measured. The cost is fixed at its optimum to within a relative
    tolerance and the quantity is then minimised and maximised over what
    remains. Returns the interval; the caller decides whether it is narrow
    enough to quote a point.
    """
    model = MultiPeriod(case, settings, form)
    a_eq, b_eq, a_ub, b_ub, _names, _n = model.build()
    cost = model.objective()
    bounds = model.bounds()
    base = linprog(cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                   bounds=bounds, method="highs-ds")
    if not base.success:
        return (float("nan"), float("nan"))

    ceiling = np.zeros((1, model.size))
    ceiling[0] = cost
    rows = sp.vstack([a_ub, sp.csr_matrix(ceiling)], format="csr")
    rhs = np.concatenate(
        [b_ub, [base.fun + TIE_BREAK_SLACK * max(abs(base.fun), 1.0)]])

    target = np.zeros(model.size)
    target[model.index[quantity]] = HOURS if quantity == "curt" else 1.0
    low = linprog(target, A_ub=rows, b_ub=rhs, A_eq=a_eq, b_eq=b_eq,
                  bounds=bounds, method="highs-ds")
    high = linprog(-target, A_ub=rows, b_ub=rhs, A_eq=a_eq, b_eq=b_eq,
                   bounds=bounds, method="highs-ds")
    if not (low.success and high.success):
        return (float("nan"), float("nan"))
    return (float(low.fun), float(-high.fun))


def optimal_face_profile(case: Case, settings, form: str,
                         quantity: str = "r") -> list:
    """The same measurement, period by period rather than over the day.

    A day total can be narrow while the periods inside it are wide, because
    the face permits moving a quantity from one hour to another at no cost.
    Any statement about where in the day two formulations part company is a
    statement about single periods, so the interval has to be measured on
    single periods too. Returns one ``(lo, hi)`` per period.

    The model is built once and only the objective changes between solves,
    which is what makes sixteen intervals affordable.
    """
    model = MultiPeriod(case, settings, form)
    a_eq, b_eq, a_ub, b_ub, _names, _n = model.build()
    cost = model.objective()
    bounds = model.bounds()
    base = linprog(cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                   bounds=bounds, method="highs-ds")
    if not base.success:
        return [(float("nan"), float("nan"))] * len(case.demand_mw)

    rows = sp.vstack([a_ub, sp.csr_matrix(cost.reshape(1, -1))],
                     format="csr")
    rhs = np.concatenate(
        [b_ub, [base.fun + TIE_BREAK_SLACK * max(abs(base.fun), 1.0)]])

    span = []
    for t in range(len(case.demand_mw)):
        target = np.zeros(model.size)
        target[model.index[quantity][t]] = 1.0
        low = linprog(target, A_ub=rows, b_ub=rhs, A_eq=a_eq, b_eq=b_eq,
                      bounds=bounds, method="highs-ds")
        high = linprog(-target, A_ub=rows, b_ub=rhs, A_eq=a_eq, b_eq=b_eq,
                       bounds=bounds, method="highs-ds")
        span.append((float(low.fun), float(-high.fun))
                    if low.success and high.success
                    else (float("nan"), float("nan")))
    return span


def ac_voltage_pass(case: Case, outcome: dict, solved: list = None,
                    converter_model: str = "injection",
                    controls: list = None) -> np.ndarray:
    """Terminal voltages that actually result from a cleared schedule.

    The clearing takes converter terminal voltage as a datum, but voltage is an
    outcome of the reactive power the clearing schedules, so the two are only
    consistent at a fixed point. This runs one ac power flow per period at the
    cleared dispatch, with the converters entered as constant injections
    carrying the active and reactive power they were cleared for, and returns
    the voltages that result. Synchronous units keep their voltage setpoints,
    which is what they do.
    """
    net = load_network(case.name)
    base_p = net.load.p_mw.copy()
    base_q = net.load.q_mvar.copy()

    # Converters leave the generator tables and re-enter as injections, so
    # their terminal voltage is a result rather than a setpoint.
    # What each converter's bus was held at before the machine there became a
    # converter. Used only by the voltage model below, and taken from the case
    # rather than chosen, so that a converter inherits the setpoint of the
    # machine it replaced instead of one this study made up.
    setpoint = {}
    for table in ("gen", "ext_grid"):
        frame = getattr(net, table)
        if "vm_pu" in frame.columns:
            for _, unit in frame.iterrows():
                setpoint.setdefault(int(unit["bus"]), float(unit["vm_pu"]))

    conv_buses = []
    for table, index in case.conv_gen_index:
        frame = getattr(net, table)
        conv_buses.append(int(frame.at[index, "bus"]))
        frame.drop(index=index, inplace=True)

    # On IEEE 39-bus the slack machine is small enough to be taken as a
    # converter, and removing it leaves the ac case with no reference bus.
    # Losses have to close somewhere, and the honest place is the largest
    # remaining synchronous machine, which is the unit that would pick them up.
    if not len(net.ext_grid):
        biggest = int(np.argmax(case.sync_rating_mva))
        table, index = case.sync_gen_index[biggest]
        if table != "gen":
            raise RuntimeError(
                f"{case.name}: the largest synchronous unit is in {table} and "
                "cannot be made the reference bus")
        getattr(net, table).at[index, "slack"] = True

    net_p = outcome["pd"] - outcome["pc"] - outcome["curt"]
    injection = np.zeros_like(net_p)
    if case.conv_injection_pu is not None:
        injection = np.outer(np.asarray(case.conv_injection_pu, dtype=float),
                             case.conv_rating_mva)
    net_p = net_p + injection

    voltages = []
    for t, level in enumerate(PROFILE):
        working = copy.deepcopy(net)
        working.load.p_mw = base_p * level * case.load_scale
        working.load.q_mvar = base_q * level * case.load_scale
        # Voltage controls for this period, if a caller is correcting them.
        # One setting for the whole day is what the base-case repair gives,
        # and it cannot be right at both the peak and the trough: a profile
        # lifted to clear the floor at heavy load sits against the ceiling at
        # light load. An operator moves these between dispatches.
        if controls is not None and controls[t] is not None:
            for index, value in controls[t].get("gen", {}).items():
                working.gen.at[index, "vm_pu"] = float(value)
            for index, value in controls[t].get("tap", {}).items():
                working.trafo.at[index, "tap_pos"] = float(value)
            for index, value in controls[t].get("ext", {}).items():
                if index in working.ext_grid.index:
                    working.ext_grid.at[index, "vm_pu"] = float(value)
        for j, (table, index) in enumerate(case.sync_gen_index):
            getattr(working, table).at[index, "p_mw"] = float(outcome["pg"][t, j])
        added = []
        for i, bus in enumerate(conv_buses):
            if converter_model == "injection":
                added.append(pp.create_sgen(working, bus,
                                            p_mw=float(net_p[t, i]),
                                            q_mvar=float(outcome["q"][t, i])))
                continue
            # A grid-forming converter is a voltage source behind an
            # impedance, which is the premise of this whole paper, so under
            # this model it holds its bus and supplies whatever reactive power
            # that takes -- up to what its own capability set allows at the
            # active power it is delivering, and no further.
            hold = setpoint.get(bus, 1.0)
            rating = float(case.conv_rating_mva[i])
            reach = (hold * rating) ** 2 - float(net_p[t, i]) ** 2
            room = float(np.sqrt(reach)) if reach > 0.0 else 0.0
            added.append(pp.create_gen(working, bus,
                                       p_mw=float(net_p[t, i]), vm_pu=hold,
                                       min_q_mvar=-room, max_q_mvar=room))
        # A schedule the ac problem cannot solve is a result, and a loud one.
        # Returning the previous voltages instead makes the move zero, which
        # the fixed point then reports as convergence: the one outcome that
        # must never be confused with a diverged power flow.
        pp.runpp(working, numba=False, enforce_q_lims=True)
        voltages.append([float(working.res_bus.at[b, "vm_pu"])
                         for b in conv_buses])
        if solved is not None:
            solved.append(working)
    return np.array(voltages)


def ac_feasibility(case: Case, outcome: dict,
                   converter_model: str = "injection",
                   controls: list = None) -> dict:
    """Does the cleared schedule satisfy the ac limits the case declares?

    A fixed point is not a feasibility certificate. Convergence says the
    voltages the clearing assumed are the voltages its schedule produces; it
    says nothing about whether those voltages are inside the case's own
    limits, whether the machines can supply the reactive power the flow needs,
    whether any branch is overloaded, or how much the slack had to invent to
    close the balance. Those are separate questions and each is asked here.

    Reported per period and aggregated, so a configuration can be called
    ac-feasible only when every period passes every test.

    The slack figure is a correction and not an output. What matters is how
    much active power the reference bus had to add to what the clearing gave
    it, so it is reported as the difference between the two. Reporting the
    absolute output instead makes a well-behaved case look bad in proportion
    to the size of its reference machine, and reports zero for a case whose
    reference is a generator rather than an external grid --- which is the
    IEEE 39-bus configuration where the slack machine is itself a converter.
    """
    net = load_network(case.name)
    v_hi = float(net.bus.max_vm_pu.max())
    v_lo = float(net.bus.min_vm_pu.min())

    # Which unit closes the balance, and what the clearing had given it. The
    # selection repeats the rule in ac_voltage_pass, deliberately: if the two
    # ever disagree the difference below is measured against the wrong unit.
    reference = next((j for j, (table, _) in enumerate(case.sync_gen_index)
                      if table == "ext_grid"), None)
    if reference is None:
        reference = int(np.argmax(case.sync_rating_mva))

    worst = dict(v_max=-np.inf, v_min=np.inf, loading_max=0.0,
                 slack_mw=0.0, q_at_limit=0, periods_failed=0)
    solved = []
    ac_voltage_pass(case, outcome, solved=solved,
                    converter_model=converter_model, controls=controls)
    for t, working in enumerate(solved):
        bad = False
        worst["v_max"] = max(worst["v_max"], float(working.res_bus.vm_pu.max()))
        worst["v_min"] = min(worst["v_min"], float(working.res_bus.vm_pu.min()))
        bad |= working.res_bus.vm_pu.max() > v_hi + 1e-6
        bad |= working.res_bus.vm_pu.min() < v_lo - 1e-6
        for table in ("res_line", "res_trafo"):
            frame = getattr(working, table, None)
            if frame is not None and len(frame) and "loading_percent" in frame:
                worst["loading_max"] = max(
                    worst["loading_max"],
                    float(np.nanmax(frame.loading_percent.to_numpy())))
                bad |= bool(np.nanmax(frame.loading_percent.to_numpy())
                            > 100.0 + 1e-6)

        cleared = float(outcome["pg"][t, reference])
        table, index = case.sync_gen_index[reference]
        if table == "ext_grid" and len(working.res_ext_grid):
            actual = float(working.res_ext_grid.p_mw.sum())
        elif index in working.res_gen.index:
            actual = float(working.res_gen.at[index, "p_mw"])
        else:
            actual = cleared
        worst["slack_mw"] = max(worst["slack_mw"], abs(actual - cleared))

        # enforce_q_lims turns a machine that runs out of reactive power into
        # a PQ bus, so the flow solves and the voltage it was holding is gone.
        # That is a failure of the schedule, not a detail of the solver: the
        # reactive requirement the clearing imposed is not being delivered.
        at_limit = 0
        if len(working.res_gen):
            q = working.res_gen.q_mvar.to_numpy(dtype=float)
            for column, sign in (("max_q_mvar", 1.0), ("min_q_mvar", -1.0)):
                if column not in working.gen:
                    continue
                limit = working.gen[column].to_numpy(dtype=float)
                near = np.isfinite(limit) & (sign * (q - limit) > -1e-6)
                at_limit += int(near.sum())
        worst["q_at_limit"] = max(worst["q_at_limit"], at_limit)
        bad |= at_limit > 0

        worst["periods_failed"] += int(bad)
    worst["v_limit_hi"], worst["v_limit_lo"] = v_hi, v_lo
    worst["feasible"] = worst["periods_failed"] == 0
    return worst


def solve_to_voltage_fixed_point(case: Case, settings, form: str,
                                 rounds: int = 40, tol: float = 1e-3,
                                 cost_tol: float = 1e-4,
                                 primal_tol: float = 1e-4,
                                 damping: float = 0.5) -> dict:
    """Clear, re-solve the network, re-clear, until nothing moves.

    Three residuals have to settle. The voltage residual is the gap between
    the voltages the clearing assumed and the voltages its own schedule
    produces. The cost residual is the relative change in cleared cost. The
    primal residual is the relative change in the decision vector itself,
    and it is the one that means what the other two are often taken to mean:
    cost is constant across alternative optima by construction, so a cost
    residual can sit at machine precision while every dispatch in the problem
    is still moving. That is exactly what happened here before the clearing
    was made single-valued, and it is why the earlier version of this routine
    could report a settled schedule that was not settled at all.

    Uniqueness comes from the tie-break in ``solve``: among schedules at the
    optimal cost, the one using least reactive current. Without it there is no
    map to converge to, only a correspondence, and no amount of damping helps.

    Reports the path, not just the endpoint: a referee's question about holding
    voltage fixed is answered by how far the answer travels, and that is only
    visible if every round is kept. A run that does not converge is returned
    as not converged, with its residuals, rather than truncated and reported
    alongside the ones that did.
    """
    working, history, previous_x = case, [], None
    for round_index in range(rounds):
        outcome = MultiPeriod(working, settings, form).solve(tie_break=True)
        if not outcome["feasible"]:
            return {"feasible": False, "rounds": round_index,
                    "converged": False, "history": history}
        voltages = ac_voltage_pass(working, outcome)
        move = float(np.abs(voltages - working.conv_voltage_pu).max())
        previous = history[-1]["cost"] if history else None
        drift = (abs(outcome["cost"] - previous) / max(abs(previous), 1e-9)
                 if previous is not None else float("inf"))
        # The primal residual, which is the one that actually says the
        # schedule has stopped moving. Cost is flat across alternative optima
        # by definition, so a cost residual can read zero while every
        # dispatch in the problem is still changing.
        step = (float(np.abs(outcome["x"] - previous_x).max())
                / max(1.0, float(np.abs(previous_x).max()))
                if previous_x is not None else float("inf"))
        previous_x = outcome["x"]
        history.append({
            "round": round_index,
            "cost": outcome["cost"],
            "price_inertia": prices.price(outcome["prices"], "inertia"),
            "voltage_move": move,
            "cost_drift": drift,
            "primal_move": step,
            "min_voltage": float(voltages.min()),
            "max_voltage": float(voltages.max()),
        })
        if move < tol and drift < cost_tol and step < primal_tol:
            break
        blended = (working.conv_voltage_pu
                   + damping * (voltages - working.conv_voltage_pu))
        working = dataclasses.replace(working, conv_voltage_pu=blended)
    last = history[-1]
    outcome["rounds"] = len(history)
    outcome["history"] = history
    outcome["voltage_residual"] = last["voltage_move"]
    outcome["cost_residual"] = last["cost_drift"]
    outcome["primal_residual"] = last["primal_move"]
    outcome["converged"] = bool(last["voltage_move"] < tol
                                and last["cost_drift"] < cost_tol
                                and last["primal_move"] < primal_tol)
    return outcome
