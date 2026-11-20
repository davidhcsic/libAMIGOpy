from pathlib import Path

import numpy as np
import pytest

from mini_amigo import Inputs, Experiment, PEsol, IvpSol, check_inputs
from mini_amigo.errors import InputError


def mini_inputs(**kw):
    """2-experiment step-stimulus problem (same as mini_run_example.m)."""
    inp = Inputs()
    m = inp.model
    m.n_st, m.st_names, m.eqns = 1, ["X"], ["dX=k1*S0*X-k2*X"]
    m.n_par, m.par_names, m.par = 2, ["k1", "k2"], np.array([0.1, 0.02])
    m.n_stimulus, m.stimulus_names = 1, ["S0"]
    inp.ivpsol = IvpSol(rtol=1e-5, atol=1e-6, max_step_size=100,
                        ivp_maxnumsteps=10000, nthreads=4)
    inp.PEsol = PEsol(id_global_theta=["k1", "k2"],
                      global_theta_min=np.array([0.001, 0.0001]),
                      global_theta_max=np.array([10.0, 10.0]))
    inp.results_folder = kw.get("results_folder", "py_test")
    inp.short_name = kw.get("short_name", "py_test")
    for iexp in (1, 2):
        t = np.linspace(0, 60, 40)
        e = Experiment(n_obs=2, obs=["Xobs=X", "X2obs=2*X"],
                       obs_names=["Xobs", "X2obs"], t_s=t, t_in=0.0, t_f=t[-1],
                       exp_y0=np.array([1.0]), u_interp="step",
                       t_con=np.concatenate([[0], t[1:-1] - 0.5 * np.mean(np.diff(t)), [t[-1]]]),
                       u=np.full((1, 39), 0.5 + 0.5 * iexp), n_steps=39,
                       exp_data=np.zeros((40, 2)))
        inp.exps.append(e)
    return inp


def test_valid_inputs_pass():
    inp = mini_inputs()
    check_inputs(inp)  # no raise
    assert inp.PEsol.n_global_theta == 2
    np.testing.assert_array_equal(inp.PEsol.index_global_theta, [0, 1])


def test_theta_index_selection():
    inp = mini_inputs()
    inp.PEsol = PEsol(id_global_theta=["k2"])
    check_inputs(inp)
    assert inp.PEsol.n_global_theta == 1
    np.testing.assert_array_equal(inp.PEsol.index_global_theta, [1])


def test_theta_all_and_none():
    inp = mini_inputs()
    inp.PEsol = PEsol(id_global_theta="all")
    check_inputs(inp)
    assert inp.PEsol.n_global_theta == 2

    inp2 = mini_inputs()
    inp2.PEsol = PEsol(id_global_theta="none")
    check_inputs(inp2)
    assert inp2.PEsol.n_global_theta == 0


def test_unknown_theta_name_rejected():
    inp = mini_inputs()
    inp.PEsol = PEsol(id_global_theta=["k9"])
    with pytest.raises(InputError, match="unknown parameter"):
        check_inputs(inp)


def test_wrong_exe_type_rejected():
    inp = mini_inputs()
    inp.model.exe_type = "standard"
    with pytest.raises(InputError, match="costMex"):
        check_inputs(inp)


def test_bounds_mismatch_rejected():
    inp = mini_inputs()
    inp.PEsol.global_theta_min = np.array([1.0])  # wrong size
    with pytest.raises(InputError, match="global_theta_min"):
        check_inputs(inp)

    inp2 = mini_inputs()
    inp2.PEsol.global_theta_min = np.array([5.0, 5.0])
    inp2.PEsol.global_theta_max = np.array([1.0, 1.0])
    with pytest.raises(InputError, match="error_bounds_013"):
        check_inputs(inp2)


def test_step_t_con_length_enforced():
    inp = mini_inputs()
    inp.exps[0].t_con = np.linspace(0, 60, 10)  # != n_steps+1
    with pytest.raises(InputError, match="error_exps_007"):
        check_inputs(inp)


def test_t_con_coinciding_with_sampling_rejected():
    # README gotcha 2: control times must not coincide with sampling times
    inp = mini_inputs()
    e = inp.exps[0]
    e.t_con = np.concatenate([[0], e.t_s[1:-1], [e.t_s[-1]]])
    with pytest.raises(InputError, match="must not coincide"):
        check_inputs(inp)


def test_step_pend_and_ncon():
    inp = mini_inputs()
    check_inputs(inp)
    e = inp.exps[0]
    assert e.n_con == 39
    assert e.pend.shape == (1, 39)
    assert np.all(e.pend == 0)


def test_linear_pend_slope():
    inp = mini_inputs()
    e = inp.exps[0]
    e.u_interp = "linear"
    e.n_linear = 2
    e.t_con = np.array([0.0, 60.0])
    e.u = np.array([[1.0, 3.0]])
    check_inputs(inp)
    assert e.n_con == 1
    assert e.pend[0, 0] == pytest.approx((3.0 - 1.0) / 60.0)
    np.testing.assert_allclose(e.u, [[1.0]])


def test_pulse_up_profile():
    inp = mini_inputs()
    e = inp.exps[0]
    e.u_interp = "pulse-up"
    e.n_pulses = 2
    e.u_min = np.array([0.0])
    e.u_max = np.array([1.0])
    e.t_con = np.array([])
    check_inputs(inp)
    assert e.n_steps == 5
    np.testing.assert_allclose(e.u, [[0, 1, 0, 1, 0]])


def test_no_stimulus_defaults():
    inp = mini_inputs()
    inp.model.n_stimulus = 0
    inp.model.stimulus_names = []
    for e in inp.exps:
        e.u_interp = "linear"
        e.u = np.zeros((0, 0))
        e.t_con = np.zeros(0)
    check_inputs(inp)
    e = inp.exps[0]
    assert e.n_con == 1
    assert e.u[0, 0] == 0.0
    np.testing.assert_allclose(e.t_con, [0.0, 60.0])


def test_index_observables_forced_empty():
    # README gotcha 1: equation observables -> index_observables None (OBS func)
    inp = mini_inputs()
    inp.exps[0].index_observables = np.array([0, 1], dtype=np.int32)
    check_inputs(inp)
    assert inp.exps[0].index_observables is None


def test_q_and_w_obs_defaults():
    inp = mini_inputs()
    check_inputs(inp)
    e = inp.exps[0]
    assert e.Q.shape == (40, 2)
    assert np.all(e.Q == 1)
    np.testing.assert_allclose(e.w_obs, [1.0, 1.0])


def test_exp_y0_wrong_size_rejected():
    inp = mini_inputs()
    inp.exps[0].exp_y0 = np.array([1.0, 2.0])
    with pytest.raises(InputError, match="exp_y0"):
        check_inputs(inp)


def test_t_f_before_t_in_rejected():
    inp = mini_inputs()
    inp.exps[0].t_f = -1.0
    with pytest.raises(InputError, match="error_exps_003"):
        check_inputs(inp)
