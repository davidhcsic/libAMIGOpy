"""sim(): run sim_CVODES through the compiled dylib (port of mini_AMIGO_sim.m)."""

from __future__ import annotations

import ctypes
import warnings
from dataclasses import dataclass

import numpy as np

from mini_amigo.errors import SimError
from mini_amigo.prep import Session


@dataclass
class CostResults:
    """Mirrors the costMex 'cost_LSQ' output (mxEvalAMIGOlsq)."""

    f: float
    w_res: np.ndarray          # (n_data,), exp-major then obs-major then time
    success: list[bool]
    error_flag: list[int]


@dataclass
class SimResults:
    """Mirrors results.sim.* of mini_AMIGO_sim.m."""

    obs: list[np.ndarray]      # per exp: (n_s, n_obs)
    states: list[np.ndarray]   # per exp: (n_s, n_states)
    tsim: list[np.ndarray]     # per exp: (n_s,)
    success: list[bool]
    error_flag: list[int]


def sim(session: Session, theta: np.ndarray | None = None) -> SimResults:
    """Simulate all experiments. theta defaults to the current model.par."""
    inp = session.inputs
    pe = inp.PEsol

    if theta is not None and pe.n_global_theta > 0:
        session.set_theta(theta)
    elif theta is None and pe.n_global_theta > 0:
        # mini_AMIGO_sim.m:22 — theta from inputs.model.par
        session.set_theta(inp.model.par[pe.index_global_theta])

    rc = session.lib.amigo_py_sim_cvodes(session._session)
    if rc != 0:
        raise SimError(f"amigo_py_sim_cvodes failed (rc={rc})")

    n_st = inp.model.n_st
    obs, states, tsim, success, eflags = [], [], [], [], []
    for i, e in enumerate(inp.exps):
        nob, nts = e.n_obs, e.n_s
        pobs = session.lib.amigo_py_get_obs(session._session, i)
        psim = session.lib.amigo_py_get_sim(session._session, i)
        # C output is column-major: obs[j*nts + k] (obs j contiguous)
        obs.append(np.ctypeslib.as_array(pobs, shape=(nob, nts)).T.copy() if nob else np.zeros((nts, 0)))
        states.append(np.ctypeslib.as_array(psim, shape=(n_st, nts)).T.copy())
        tsim.append(np.array(e.t_s, dtype=float))
        success.append(bool(session.lib.amigo_py_get_success(session._session, i)))
        eflags.append(int(session.lib.amigo_py_get_error_flag(session._session, i)))

    n_fail = success.count(False)
    if n_fail:
        warnings.warn(
            f"mini_AMIGO_sim: {n_fail} of {inp.n_exp} experiments failed to integrate.",
            RuntimeWarning, stacklevel=2)

    return SimResults(obs=obs, states=states, tsim=tsim, success=success,
                      error_flag=eflags)


def cost(session: Session, theta: np.ndarray | None = None) -> CostResults:
    """LSQ cost and weighted residuals (costMex 'cost_LSQ' task)."""
    inp = session.inputs
    pe = inp.PEsol

    if theta is not None and pe.n_global_theta > 0:
        session.set_theta(theta)
    elif theta is None and pe.n_global_theta > 0:
        session.set_theta(inp.model.par[pe.index_global_theta])

    n_data = session.lib.amigo_py_n_data(session._session)
    w_res = np.zeros(n_data, dtype=np.float64)
    f = session.lib.amigo_py_cost_lsq(
        session._session, w_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    if f < 0:
        raise SimError("amigo_py_cost_lsq failed")

    success = [bool(session.lib.amigo_py_get_success(session._session, i))
               for i in range(inp.n_exp)]
    eflags = [int(session.lib.amigo_py_get_error_flag(session._session, i))
              for i in range(inp.n_exp)]
    return CostResults(f=float(f), w_res=w_res, success=success,
                       error_flag=eflags)
