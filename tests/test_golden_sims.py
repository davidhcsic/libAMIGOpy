"""Golden integration tests (CONSTITUTION Art. III).

Expected values come from the fully compiled MATLAB costMex MEX
(python/tools/generate_golden_matlab.m). Never hand-edit the goldens.
"""

import json

import numpy as np
import pytest

from mini_amigo import prep, sim, cost
from mini_amigo.errors import BuildError
from tests.test_inputs import mini_inputs

def _load(golden_name):
    from pathlib import Path
    path = Path(__file__).parent / "golden" / f"{golden_name}.json"
    if not path.is_file():
        pytest.skip(f"golden file missing: {path} (regenerate with "
                    "python/tools/generate_golden_matlab.m)")
    return json.loads(path.read_text())


def _inputs_for(case):
    if case in ("mini_nominal", "mini_theta_alt"):
        return mini_inputs(results_folder="py_golden_mini",
                           short_name="py_golden_mini")
    if case == "mini_ramp_theta":
        inp = mini_inputs(results_folder="py_golden_ramp",
                          short_name="py_golden_ramp")
        for iexp, e in enumerate(inp.exps, start=1):
            t = np.linspace(0, 30, 25)
            e.t_s = t
            e.t_f = t[-1]
            e.u_interp = "linear"
            e.n_linear = 2
            e.t_con = np.array([0.0, 30.0])
            e.u = np.array([[0.2, 1.0 + 0.5 * iexp]])
        return inp
    raise ValueError(case)


@pytest.mark.parametrize("case", ["mini_nominal", "mini_theta_alt", "mini_ramp_theta"])
def test_golden_against_matlab_costmex(case):
    gold = _load(case)
    tol = float(gold["tolerance_rel"])
    theta = np.array(gold["theta"])

    inp = _inputs_for(case)
    inp.ivpsol.nthreads = 1  # determinism (CONSTITUTION Art. III.3)
    try:
        session = prep(inp)
    except BuildError as exc:
        pytest.skip(f"toolchain unavailable: {exc}")
    try:
        res = sim(session, theta=theta)

        assert [bool(s) for s in res.success] == [e["success"] for e in gold["exps"]]
        assert len(res.obs) == gold["n_exp"]

        for iexp, (exp_gold, obs, states, tsim) in enumerate(
                zip(gold["exps"], res.obs, res.states, res.tsim), start=1):
            t_gold = np.array(exp_gold["t_s"])
            # golden JSON stores 10 significant digits (MATLAB string())
            np.testing.assert_allclose(tsim, t_gold, rtol=1e-9, atol=1e-12)

            obs_gold = np.array(exp_gold["obs"]).T          # (n_obs, n_t) -> (n_t, n_obs)
            st_gold = np.array(exp_gold["states"]).T
            assert obs.shape == obs_gold.shape, f"exp {iexp} obs shape"
            assert states.shape == st_gold.shape, f"exp {iexp} states shape"

            # same sampling grid -> elementwise comparison
            denom = np.maximum(1.0, np.abs(obs_gold))
            err = np.max(np.abs(obs - obs_gold) / denom)
            assert err < tol, f"exp {iexp}: observables differ by {err:.3e} (tol {tol})"

            denom = np.maximum(1.0, np.abs(st_gold))
            err = np.max(np.abs(states - st_gold) / denom)
            assert err < tol, f"exp {iexp}: states differ by {err:.3e} (tol {tol})"
    finally:
        session.close()


@pytest.mark.parametrize("case", ["mini_nominal", "mini_theta_alt", "mini_ramp_theta"])
def test_golden_cost_lsq(case):
    """cost() vs MATLAB costMex 'cost_LSQ' (regenerate goldens to enable)."""
    gold = _load(case)
    if "cost_f" not in gold:
        pytest.skip(f"golden {case} lacks cost_f (regenerate with "
                    "python/tools/generate_golden_matlab.m)")
    tol = float(gold["tolerance_rel"])
    inp = _inputs_for(case)
    inp.ivpsol.nthreads = 1
    try:
        session = prep(inp)
    except BuildError as exc:
        pytest.skip(f"toolchain unavailable: {exc}")
    try:
        c = cost(session, theta=np.array(gold["theta"]))
        assert abs(c.f - gold["cost_f"]) <= tol * max(1.0, abs(gold["cost_f"]))
        w_gold = np.array(gold["cost_w_res"])
        assert c.w_res.shape == w_gold.shape
        denom = np.maximum(1.0, np.abs(w_gold))
        err = np.max(np.abs(c.w_res - w_gold) / denom)
        assert err < tol, f"w_res differs by {err:.3e} (tol {tol})"
    finally:
        session.close()


def test_golden_theta_sync():
    """mini_AMIGO_sim.m:26 — global_theta_guess tracks the active theta."""
    gold = _load("mini_theta_alt")
    theta = np.array(gold["theta"])
    inp = _inputs_for("mini_theta_alt")
    try:
        session = prep(inp)
    except BuildError as exc:
        pytest.skip(f"toolchain unavailable: {exc}")
    try:
        sim(session, theta=theta)
        np.testing.assert_allclose(session.inputs.PEsol.global_theta_guess, theta)
        np.testing.assert_allclose(
            session.inputs.model.par[session.inputs.PEsol.index_global_theta], theta)
    finally:
        session.close()


def test_cost_matches_sim_and_analytic():
    """f == sum(w_res^2) and f == analytic sum for exp-growth model."""
    inp = _inputs_for("mini_nominal")
    inp.ivpsol.nthreads = 1
    try:
        session = prep(inp)
    except BuildError as exc:
        pytest.skip(f"toolchain unavailable: {exc}")
    try:
        theta = np.array([0.1, 0.02])
        res = sim(session, theta=theta)
        c = cost(session, theta=theta)
        assert all(c.success)
        assert c.f == pytest.approx(float(np.sum(c.w_res ** 2)), rel=1e-12)

        k1, k2 = theta
        f_ref = 0.0
        for i, e in enumerate(session.inputs.exps):
            s0 = e.u[0, 0]
            for j in range(e.n_obs):
                ref = (2.0 ** j) * np.exp((k1 * s0 - k2) * res.tsim[i])
                # exp_data is zeros, Q is ones -> residual is the observable
                f_ref += float(np.sum(ref ** 2))
        # f squares a solution with rtol=1e-5 -> squared relative error ~2e-3
        rel = abs(c.f - f_ref) / max(1.0, f_ref)
        assert rel < 1e-2, f"cost vs analytic: {rel:.3e}"
    finally:
        session.close()


def test_analytic_solution():
    """Complements goldens: exponential growth model has a closed form."""
    inp = _inputs_for("mini_nominal")
    try:
        session = prep(inp)
    except BuildError as exc:
        pytest.skip(f"toolchain unavailable: {exc}")
    try:
        res = sim(session)
        k1, k2 = session.inputs.model.par
        assert all(res.success)
        for i, e in enumerate(session.inputs.exps):
            s0 = e.u[0, 0]
            ref = np.exp((k1 * s0 - k2) * res.tsim[i])
            err = np.max(np.abs(res.obs[i][:, 0] - ref) / np.maximum(1, np.abs(ref)))
            assert err < 1e-2, f"exp {i}: {err:.3e}"
            # second observable is exactly 2*X
            np.testing.assert_allclose(res.obs[i][:, 1], 2 * res.obs[i][:, 0], rtol=1e-12)
    finally:
        session.close()
