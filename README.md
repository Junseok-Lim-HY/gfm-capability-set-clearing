# Current-limited capability of grid-forming converters in reserve and inertia scheduling

Code, data and manuscript sources for

> J. Lim, S. Ji, B. Lee and S. Bae, "Current-limited capability of grid-forming
> converters in reserve and inertia scheduling: formulation, sign condition and
> validation", submitted to *Sustainable Energy, Grids and Networks* (Elsevier).

**Status: under review.** The repository is private while the manuscript is
under review. On acceptance it will be made public and an archived release
with a DOI will be added here and in the paper's Data availability statement.

## What is here

The paper represents the continuous and short-term current ratings, reactive
current and terminal voltage of a grid-forming converter as a polyhedral
capability set, embeds it in a multi-period linear clearing of energy, reserve
and inertia whose prices are dual variables, derives a piecewise condition for
the sign of the error made by the usual active-power bound, and validates the
schedules by ac power flow, a post-contingency deliverability constraint and an
RMS converter model on the IEEE 39-bus and RTS-24 systems.

    model/                      capability set and clearing
      capability.py             current disc, inscribed polygon, inertia in MW at a RoCoF
      multiperiod.py            multi-period clearing, sparse constraint rows, derating,
                                post-contingency rows, ac voltage pass
      network.py                shift factors, branch ratings, zones; IEEE 39, RTS-24, IEEE 118
      ac_linear.py, ac_rows.py  finite-difference sensitivities and linearised ac rows
      ac_clearing.py            successive linear programming loop
      voltage_control.py        least-change search over setpoints, taps and the reference bus
      prices.py                 requirement duals to a reported price
      procurement_lp.py         Capability settings

    study/                      every script behind a number in the paper; see reproduce.py
      results/                  the csv files the tables and figures are built from
      check_r2_numbers.py       headline numbers of the paper against results/
      check_r2_numbers_v2.py    full numbers check: recomputes every printed value from the
                                cost/price columns and compares the PDF text (needs pymupdf)
      audit_reruns_260929.py    re-runs behind the 2026-09-29 corrections

    manuscript/                 LaTeX sources (elsarticle), refs.bib, highlights
      SEGAN_Manuscript_<date>.pdf, SEGAN_Supplementary_<date>.pdf
                                compiled current version (one pair; replaced on each update)
      figures/                  figure PDFs and the scripts that draw them from study/results

    reproduce.py                runs the study in dependency order and checks the numbers
    requirements.txt            direct dependencies; requirements-lock.txt pins the exact environment

## Reproducing

    python -m venv venv
    venv/Scripts/pip install -r requirements-lock.txt      # Windows; use venv/bin/pip elsewhere
    venv/Scripts/python reproduce.py --quick               # tests + number check, minutes
    venv/Scripts/python reproduce.py                       # full regeneration, hours

The results were produced with Python 3.12, pandapower 3.5.4, SciPy/HiGHS and
cvxpy 1.9.2 (CLARABEL) on a 16 GB desktop. The AC-feasible clearing builds
large constraint matrices; run the scripts one at a time.

## Test systems

The IEEE 39-bus and RTS-24 cases are loaded from pandapower and modified in
code (`model/network.py`, `study/base_case_repair.py`): the source cases
violate their own voltage limits and are corrected by moving generator
setpoints and transformer taps before use. The converter placement, rating
conventions and all parameters are stated in the paper and in the
Supplementary Material and are set in `model/procurement_lp.py` and the study
scripts.

## Building the manuscript

    cd manuscript
    pdflatex main && bibtex main && pdflatex main && pdflatex main
    pdflatex supplementary && pdflatex supplementary

`elsarticle` is part of TeX Live and MiKTeX.

## Licence

MIT, see `LICENSE`.

## Contact

Junseok Lim, Power and Energy System Laboratory, Department of Electrical
Engineering, Hanyang University (joonseok1003@hanyang.ac.kr).
