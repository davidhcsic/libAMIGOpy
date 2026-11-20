import numpy as np
import pytest

from mini_amigo import convert_power_operator, gen_c
from mini_amigo.errors import InputError
from tests.test_inputs import mini_inputs


# ---- convert_power_operator (port of convertPowerOperator.m) ---------------

@pytest.mark.parametrize("src,expected", [
    ("dX=k1*X", "dX=k1*X"),
    ("dX=k1*X^2", "dX=k1*pow(X,2)"),
    ("dX=k1*X^k2", "dX=k1*pow(X,k2)"),
    ("dX=(X+1)^2", "dX=pow(X+1,2)"),
    ("dX = k1 * X ^ 2 - k2", "dX=k1*pow(X,2)-k2"),
    ("dX=power(X,2)", "dX=pow(X,2)"),
])
def test_power_operator(src, expected):
    assert convert_power_operator(src) == expected


def test_power_operator_elseif():
    # bug-compatible with MATLAB: elseif->else if happens after the simple
    # power regex, so "if" is consumed as the base (convertPowerOperator.m)
    assert convert_power_operator("elseif X^2>1") == "else pow(ifX,2)>1"


def test_power_operator_invalid():
    with pytest.raises(InputError):
        convert_power_operator("dX=(X^2")
    with pytest.raises(InputError):
        convert_power_operator("dX=k1*(X^2)^k2")  # nested parens unsupported (same as MATLAB)


# ---- gen_c (port of AMIGO_gen_C.m) ------------------------------------------

@pytest.fixture
def rhs_c():
    return gen_c(mini_inputs())


def test_gen_c_includes(rhs_c):
    for inc in ("amigoRHS.h", "math.h", "amigoJAC.h", "amigoSensRHS.h"):
        assert f"#include <{inc}>" in rhs_c


def test_gen_c_defines(rhs_c):
    assert "#define\tX Ith(y,0)" in rhs_c
    assert "#define\tdX Ith(ydot,0)" in rhs_c
    assert "#define\tk1 (*amigo_model).pars[0]" in rhs_c
    assert "#define\tk2 (*amigo_model).pars[1]" in rhs_c
    assert "#define iexp amigo_model->exp_num" in rhs_c


def test_gen_c_stimulus_macro(rhs_c):
    # S0 -> controls_v[0][index_t_stim] + slope term (AMIGO_gen_C.m:76-80)
    assert "#define S0\t((*amigo_model).controls_v[0][(*amigo_model).index_t_stim]" in rhs_c
    assert "slope[0][(*amigo_model).index_t_stim])" in rhs_c


def test_gen_c_rhs_equation(rhs_c):
    assert "\tdX=k1*S0*X-k2*X;" in rhs_c


def test_gen_c_functions_present(rhs_c):
    for fn in ("int amigoRHS(", "int amigoJAC(", "int amigoSensRHS(",
               "void amigoRHS_get_OBS(", "void amigoRHS_get_sens_OBS(",
               "void amigo_Y_at_tcon("):
        assert fn in rhs_c, fn


def test_gen_c_obs_cases(rhs_c):
    # one switch case per experiment, obs assigned into obs_results
    assert "case 0:" in rhs_c and "case 1:" in rhs_c
    assert "#define\t Xobs amigo_model->obs_results[0][j]" in rhs_c
    assert "#define\t X2obs amigo_model->obs_results[1][j]" in rhs_c
    assert "Xobs=X;" in rhs_c
    assert "X2obs=2*X;" in rhs_c


def test_gen_c_positive_states():
    inp = mini_inputs()
    inp.model.positive_states = 1
    code = gen_c(inp)
    assert "if(X <0.0) X = 0;" in code


def test_gen_c_algebraic_variable_threadprivate():
    inp = mini_inputs()
    inp.model.n_st = 2
    inp.model.st_names = ["X", "Y"]
    inp.model.eqns = ["v=k1*X", "dX=-v", "dY=v"]
    code = gen_c(inp)
    assert "double\tv;" in code
    assert "#pragma omp threadprivate(v)" in code


def test_gen_c_no_decl_for_state_assignment():
    # "X=..." (state name on LHS) must NOT declare a variable (flag logic
    # in AMIGO_gen_C.m:94-129)
    inp = mini_inputs()
    inp.model.eqns = ["dX=k1*S0*X-k2*X", "X=1"]
    code = gen_c(inp)
    assert "double\tX;" not in code
