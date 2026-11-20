"""prep(): inputs -> amigoRHS.c -> compiled dylib -> Session (ctypes).

Port of mini_AMIGO_prep.m. The Session owns the loaded dylib and the
ctypes problem description (CONSTITUTION Art. VI.2: no global state).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mini_amigo.build import build_dylib, LIBOMP_PREFIX
from mini_amigo.errors import InputError
from mini_amigo.gen_c import write_c
from mini_amigo.inputs import Inputs, check_inputs
from mini_amigo.build import RESULTS_DIR


class PyExperimentDesc(ctypes.Structure):
    _fields_ = [
        ("n_times", ctypes.c_int),
        ("t0", ctypes.c_double),
        ("tf", ctypes.c_double),
        ("t", ctypes.POINTER(ctypes.c_double)),
        ("y0", ctypes.POINTER(ctypes.c_double)),
        ("n_opt_ics", ctypes.c_int),
        ("index_opt_ics", ctypes.POINTER(ctypes.c_int)),
        ("y0_guess", ctypes.POINTER(ctypes.c_double)),
        ("y0_LB", ctypes.POINTER(ctypes.c_double)),
        ("y0_UB", ctypes.POINTER(ctypes.c_double)),
        ("n_controls_t", ctypes.c_int),
        ("t_con", ctypes.POINTER(ctypes.c_double)),
        ("u", ctypes.POINTER(ctypes.c_double)),
        ("pend", ctypes.POINTER(ctypes.c_double)),
        ("n_obs", ctypes.c_int),
        ("index_observables", ctypes.POINTER(ctypes.c_int)),
        ("exp_data", ctypes.POINTER(ctypes.c_double)),
        ("Q", ctypes.POINTER(ctypes.c_double)),
        ("w_obs", ctypes.POINTER(ctypes.c_double)),
    ]


class PyProblemDesc(ctypes.Structure):
    _fields_ = [
        ("n_exp", ctypes.c_int),
        ("n_states", ctypes.c_int),
        ("n_pars", ctypes.c_int),
        ("n_opt_pars", ctypes.c_int),
        ("n_controls", ctypes.c_int),
        ("pars", ctypes.POINTER(ctypes.c_double)),
        ("index_opt_pars", ctypes.POINTER(ctypes.c_int)),
        ("pars_guess", ctypes.POINTER(ctypes.c_double)),
        ("pars_LB", ctypes.POINTER(ctypes.c_double)),
        ("pars_UB", ctypes.POINTER(ctypes.c_double)),
        ("reltol", ctypes.c_double),
        ("atol", ctypes.c_double),
        ("max_step_size", ctypes.c_double),
        ("max_num_steps", ctypes.c_int),
        ("max_error_test_fails", ctypes.c_int),
        ("nthreads", ctypes.c_int),
        ("exps", ctypes.POINTER(PyExperimentDesc)),
    ]


def _dp(a: np.ndarray | None):
    """ctypes pointer to a C-contiguous float64 array (or NULL)."""
    if a is None or np.asarray(a).size == 0:
        return None
    arr = np.ascontiguousarray(a, dtype=np.float64)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _ip(a: np.ndarray | None):
    if a is None or np.asarray(a).size == 0:
        return None
    arr = np.ascontiguousarray(a, dtype=np.int32)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


@dataclass
class Session:
    """A prepared, compiled problem ready for simulation."""

    inputs: Inputs
    dylib_path: Path
    c_file: Path
    lib: ctypes.CDLL = field(repr=False, default=None)
    desc: PyProblemDesc = field(repr=False, default=None)
    _keepalive: list = field(repr=False, default_factory=list)
    _session: ctypes.c_void_p = field(repr=False, default=None)

    def __post_init__(self):
        self.lib = ctypes.CDLL(str(self.dylib_path))
        self.lib.amigo_py_create.argtypes = [ctypes.POINTER(PyProblemDesc)]
        self.lib.amigo_py_create.restype = ctypes.c_void_p
        self.lib.amigo_py_sim_cvodes.argtypes = [ctypes.c_void_p]
        self.lib.amigo_py_sim_cvodes.restype = ctypes.c_int
        self.lib.amigo_py_cost_lsq.argtypes = [ctypes.c_void_p,
                                               ctypes.POINTER(ctypes.c_double)]
        self.lib.amigo_py_cost_lsq.restype = ctypes.c_double
        self.lib.amigo_py_n_data.argtypes = [ctypes.c_void_p]
        self.lib.amigo_py_n_data.restype = ctypes.c_int
        self.lib.amigo_py_get_obs.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.amigo_py_get_obs.restype = ctypes.POINTER(ctypes.c_double)
        self.lib.amigo_py_get_sim.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.amigo_py_get_sim.restype = ctypes.POINTER(ctypes.c_double)
        self.lib.amigo_py_get_success.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.amigo_py_get_success.restype = ctypes.c_int
        self.lib.amigo_py_get_error_flag.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.amigo_py_get_error_flag.restype = ctypes.c_int
        self.lib.amigo_py_set_pars.argtypes = [ctypes.c_void_p,
                                               ctypes.POINTER(ctypes.c_double),
                                               ctypes.c_int]
        self.lib.amigo_py_set_pars.restype = None
        self.lib.amigo_py_set_y0.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                             ctypes.POINTER(ctypes.c_double),
                                             ctypes.c_int]
        self.lib.amigo_py_set_y0.restype = None
        self.lib.amigo_py_free.argtypes = [ctypes.c_void_p]
        self.lib.amigo_py_free.restype = None

        self._build_desc()
        self._session = ctypes.c_void_p(self.lib.amigo_py_create(ctypes.byref(self.desc)))
        if not self._session:
            raise InputError("amigo_py_create failed (see stderr)")

    def _build_desc(self) -> None:
        inp, m, iv = self.inputs, self.inputs.model, self.inputs.ivpsol
        n_exp = self.inputs.n_exp
        pe = inp.PEsol

        exps = (PyExperimentDesc * n_exp)()
        bufs: list[np.ndarray] = []

        def keep(a):
            if isinstance(a, np.ndarray):
                bufs.append(a)
            return a

        for i, e in enumerate(inp.exps):
            d = exps[i]
            d.n_times = e.n_s
            d.t0 = float(e.t_in)
            d.tf = float(e.t_f)
            d.t = _dp(keep(np.ascontiguousarray(e.t_s, dtype=np.float64)))
            d.y0 = _dp(keep(np.ascontiguousarray(e.exp_y0, dtype=np.float64)))
            d.n_opt_ics = 0
            d.index_opt_ics = None
            d.y0_guess = None
            d.y0_LB = None
            d.y0_UB = None
            d.n_controls_t = int(e.t_con.size)
            d.t_con = _dp(keep(np.ascontiguousarray(e.t_con, dtype=np.float64)))
            # C expects column-major [n_controls_t-1 x n_controls]
            # (AMIGO_mxInterface.c reads u linearly per control)
            d.u = _dp(keep(np.ascontiguousarray(e.u, dtype=np.float64)))
            d.pend = _dp(keep(np.ascontiguousarray(e.pend, dtype=np.float64)))
            d.n_obs = int(e.n_obs)
            d.index_observables = _ip(np.ascontiguousarray(e.index_observables, dtype=np.int32)) \
                if e.index_observables is not None and len(e.index_observables) else None
            ed = e.exp_data if e.exp_data is not None else np.zeros((e.n_s, e.n_obs))
            d.exp_data = _dp(keep(np.ascontiguousarray(np.asarray(ed, dtype=np.float64).T)))
            d.w_obs = _dp(keep(np.ascontiguousarray(e.w_obs, dtype=np.float64)))
            q = e.Q if e.Q is not None else np.ones((e.n_s, e.n_obs))
            d.Q = _dp(keep(np.ascontiguousarray(np.asarray(q, dtype=np.float64).T)))

        pars = np.ascontiguousarray(m.par, dtype=np.float64)
        idx = np.ascontiguousarray(pe.index_global_theta, dtype=np.int32)
        guess = np.ascontiguousarray(pe.global_theta_guess, dtype=np.float64)
        lb = np.ascontiguousarray(pe.global_theta_min, dtype=np.float64)
        ub = np.ascontiguousarray(pe.global_theta_max, dtype=np.float64)

        desc = PyProblemDesc()
        desc.n_exp = n_exp
        desc.n_states = m.n_st
        desc.n_pars = m.n_par
        desc.n_opt_pars = pe.n_global_theta
        desc.n_controls = m.n_stimulus
        desc.pars = _dp(pars)
        desc.index_opt_pars = _ip(idx)
        desc.pars_guess = _dp(guess)
        desc.pars_LB = _dp(lb)
        desc.pars_UB = _dp(ub)
        desc.reltol = float(iv.rtol)
        desc.atol = float(iv.atol)
        desc.max_step_size = float(iv.max_step_size)
        desc.max_num_steps = int(iv.ivp_maxnumsteps)
        desc.max_error_test_fails = int(iv.max_error_test_fails)
        desc.nthreads = int(iv.nthreads)
        desc.exps = exps

        # keep every buffer referenced by the desc alive for the session life
        self._keepalive = [exps, pars, idx, guess, lb, ub] + bufs
        self.desc = desc

    # -- theta handling (AMIGO_transform_theta, mini_AMIGO_sim.m:25-31) ----
    def set_theta(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=np.float64).ravel()
        pe = self.inputs.PEsol
        if theta.size != pe.n_global_theta:
            raise InputError(
                f"theta has {theta.size} entries, expected n_global_theta={pe.n_global_theta}")
        par = np.array(self.inputs.model.par, dtype=np.float64)
        par[pe.index_global_theta] = theta
        ptr = _dp(np.ascontiguousarray(par))
        self._keepalive.append(par)
        self.lib.amigo_py_set_pars(self._session, ptr, self.inputs.model.n_par)
        self.inputs.model.par = par
        # mini_AMIGO_sim.m:26 keep global_theta_guess in sync
        pe.global_theta_guess = theta.copy()

    def close(self) -> None:
        if self._session:
            self.lib.amigo_py_free(self._session)
            self._session = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def prep(inputs: Inputs, force_recompile: bool = False) -> Session:
    """Python equivalent of mini_AMIGO_prep.m (costMex + charmodelC)."""
    check_inputs(inputs)

    problem_folder = RESULTS_DIR / inputs.results_folder
    problem_folder.mkdir(parents=True, exist_ok=True)

    c_file = problem_folder / "amigoRHS.c"
    if inputs.model.overwrite_model and inputs.model.eqns:
        write_c(inputs, c_file)
    inputs.model.odes_file = str(c_file)

    dylib = build_dylib(c_file, inputs.short_name, force=force_recompile)
    inputs.model.lib_file = str(dylib)

    # privstruct.theta default == nominal parameters at theta indices
    pe = inputs.PEsol
    if pe.global_theta_guess is None:
        pe.global_theta_guess = inputs.model.par[pe.index_global_theta].copy()

    return Session(inputs=inputs, dylib_path=dylib, c_file=c_file)
