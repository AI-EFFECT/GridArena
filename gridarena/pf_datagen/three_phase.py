"""Balanced three-phase convention for the MV PF data-generation pipeline.

MV injections are three-phase totals (W / VAr) and the MV nominal voltage is
line-to-line (see docs/api/index.md#units). Branch quantities are reported on
the per-phase (phase-to-neutral) equivalent:

    S_phase = S_3ph / 3,    V_phase = V_LL / sqrt(3),    Z, Y = per-phase cable values

so branch currents are the real line currents -- directly comparable to a
cable's rating -- and branch powers are converted back to three-phase totals.

The pipeline solves and computes branch flows with pf_datagen/solver.py (per
unit, where this split is implicit) and converts to SI in
pf_datagen/records.py. solve_balanced_pf() and branch_flow() below express
the same convention on top of the shared single-phase full_pf; they are kept
as the reference the MV solver is regression-tested against.
"""

import numpy as np

from gridarena.powerflow.powerflow_algorithm import full_pf

SQRT3 = np.sqrt(3.0)


def solve_balanced_pf(s_three_phase, connections, admittances, v_ll_ref):
    """Solve a balanced MV network from three-phase injections with full_pf
    (no shunts; connections must be root-outward, parent -> child).

    s_three_phase: complex three-phase power per node (VA, load convention).
    v_ll_ref: line-to-line voltage at the reference (substation) node (V).

    Returns (v_phase, converged, n_iter), where v_phase holds the complex
    phase-to-neutral voltage per node; multiply its magnitude by SQRT3 for the
    line-to-line value.
    """
    return full_pf(
        np.asarray(s_three_phase) / 3.0, connections, admittances, v_ll_ref / SQRT3,
        return_diagnostics=True,
    )


def branch_flow(v_from_phase, v_to_phase, y):
    """Series current (A) and three-phase sending-end power (VA) through one
    branch without shunts, from its end-point phase-to-neutral voltages and
    series admittance."""
    i_line = (v_from_phase - v_to_phase) * y
    return i_line, 3.0 * v_from_phase * np.conj(i_line)
