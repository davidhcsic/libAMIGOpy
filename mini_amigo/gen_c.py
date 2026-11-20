"""amigoRHS.c generator — port of core/AMIGO_gen_C*.m + convertPowerOperator.m.

Produces a C file semantically identical to the MATLAB generator for the
costMex path (CONSTITUTION Art. II.1).
"""

from __future__ import annotations

import re

from mini_amigo.errors import InputError
from mini_amigo.inputs import Inputs


def convert_power_operator(formula: str) -> str:
    """Port of core/convertPowerOperator.m: a^b -> pow(a,b).

    MATLAB order: strip spaces -> simple word^word -> power->pow ->
    elseif->else if -> iterative paren-aware conversion (single level).
    """
    f = re.sub(r"\s", "", formula)
    f = f.replace("elseif", "else if")
    # first fix the simple problem (convertPowerOperator.m:7): the MATLAB
    # regex consumes the whole base, so it cannot start right after '('
    f = re.sub(r"(?<!\()(?<!\w)[\w]+\.?[\w]*\^[\w]+\.?[\w]*",
               lambda m: "pow(%s)" % m.group(0).replace("^", ","), f)
    f = f.replace("power", "pow")
    # then the more complicated stuff: (expr)^simple — one level of parens,
    # iterated innermost-first like the MATLAB while loop
    prev = None
    while "^" in f and f != prev:
        prev = f
        f = re.sub(r"\(([^()]*)\)\^([\w]+\.?[\w]*)", r"pow(\1,\2)", f)
    if "^" in f:
        raise InputError(f"Error in formula: {f}")
    return f


def _rhs(fid, inputs: Inputs) -> None:
    m = inputs.model
    fid.write("/* Right hand side of the system (f(t,x,p))*/\n")
    fid.write("int amigoRHS(realtype t, N_Vector y, N_Vector ydot, void *data){\n")
    fid.write("\tAMIGO_model* amigo_model=(AMIGO_model*)data;\n")

    if m.positive_states == 1:
        fid.write("\n\t/* *** Test for negative states *** */\n\n")
        for nm in m.st_names:
            fid.write(f"\tif({nm} <0.0) {nm} = 0;\n")

    fid.write("\n\t/* *** Equations *** */\n\n")
    for eq in m.eqns:
        eqn = convert_power_operator(eq)
        fid.write(f"\t{eqn};\n" if re.search(r"[\w)(]$", eqn) else f"\t{eqn}\n")

    if m.positive_states == 2:
        fid.write("\n\t/* *** Test for negative states *** */\n\n")
        for nm in m.st_names:
            fid.write(f"\tif({nm} <0.0) return(1);\n")

    fid.write("\n\treturn(0);\n\n}\n\n")


def _jac(fid, inputs: Inputs) -> None:
    # AMIGO_gen_C_jacobian with empty J (AMIGOjac=0): empty stub.
    fid.write("\n/* Jacobian of the system (dfdx)*/\n")
    fid.write("int amigoJAC(long int N, realtype t, N_Vector y, N_Vector fy, "
              "DlsMat J, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3){\n")
    fid.write("\tAMIGO_model* amigo_model=(AMIGO_model*)user_data;\n")
    fid.write("\treturn(0);\n}\n")


def _sensrhs(fid, inputs: Inputs) -> None:
    # AMIGO_gen_C_sensrhs without AMIGOsensrhs: empty stub.
    fid.write("\n/* R.H.S of the sensitivity dsi/dt = (df/dx)*si + df/dp_i */\n")
    fid.write("int amigoSensRHS(int Ns, realtype t, N_Vector y, N_Vector ydot, int iS, "
              "N_Vector yS, N_Vector ySdot, void *data, N_Vector tmp1, N_Vector tmp2){\n")
    fid.write("\tAMIGO_model* amigo_model=(AMIGO_model*)data;\n")
    fid.write("\n\treturn(0);\n\n}\n")


def _obs(fid, inputs: Inputs) -> None:
    m = inputs.model
    fid.write("\n\n")
    for i, nm in enumerate(m.st_names):
        fid.write(f"#define\t {nm} (amigo_model->sim_results[{i}][j]) \n")

    fid.write("\n\n\nvoid amigoRHS_get_OBS(void* data){\n")
    fid.write("\n\tint j;\n")
    fid.write("\tdouble t;\n")
    fid.write("\tAMIGO_model* amigo_model=(AMIGO_model*)data;\n\n")
    fid.write("\n\t switch (amigo_model->exp_num){\n\n")

    for iexp, e in enumerate(inputs.exps):
        for i, onm in enumerate(e.obs_names):
            fid.write(f"\t\t#define\t {onm} amigo_model->obs_results[{i}][j] \n")
        fid.write(f"\n\t\t case {iexp}:\n\n")
        if e.n_obs > 0:
            fid.write("\n\t\t\t for (j = 0; j < amigo_model->n_times; ++j){\n")
            for expr in e.obs:
                fid.write(f"\t\t\t\t{_obs_expr(expr, e.obs_names)};\n")
            fid.write("\n\t\t\t}\n")
        fid.write("\n\t\t break;\n")

    fid.write("\n\t}\n")
    fid.write("\n\t\n\n}\n\n")


def _sens_obs(fid, inputs: Inputs) -> None:
    m = inputs.model
    for i, nm in enumerate(m.st_names):
        fid.write(f"#define\t {nm} (amigo_model->sens_results[{i}][j][k]) \n")

    fid.write("\n\n\nvoid amigoRHS_get_sens_OBS(void* data){\n")
    fid.write("\tint j,k;\n\n")
    fid.write("\tAMIGO_model* amigo_model=(AMIGO_model*)data;\n\n")
    fid.write("\n\t switch (amigo_model->exp_num){\n\n")

    for iexp, e in enumerate(inputs.exps):
        fid.write(f"\n\t\t case {iexp}:\n\n")
        if e.n_obs > 0:
            for i, onm in enumerate(e.obs_names):
                fid.write(f"\t\t#define\t {onm} amigo_model->sens_obs[{i}][j][k] \n")
            fid.write("\n\t\t\t for (j = 0; j < amigo_model->n_total_x; ++j){")
            fid.write("\n\t\t\t\t for (k = 0; k < amigo_model->n_times; ++k){\n")
            for expr in e.obs:
                fid.write(f"\t\t\t\t\t{_obs_expr(expr, e.obs_names)};\n")
            fid.write("\t\t\t\t}\n\t\t\t}")
        fid.write("\n\t\t break;\n")

    fid.write("\t}\n}\n")


def _obs_expr(expr: str, obs_names: list[str]) -> str:
    """AMIGO_gen_costMex_obs: strip whitespace/dots, obs names are already C ids."""
    temp = re.sub(r"[\s.]", "", expr)
    for nm in obs_names:
        nm_c = re.sub(r"\s", "", nm)
        temp = temp.replace(nm, nm_c)
    return temp


def gen_c(inputs: Inputs) -> str:
    """Return the full amigoRHS.c content (port of core/AMIGO_gen_C.m)."""
    m = inputs.model
    import io
    fid = io.StringIO()

    fid.write("#include <amigoRHS.h>\n\n")
    fid.write("#include <math.h>\n\n")
    fid.write("#include <amigoJAC.h>\n\n")
    fid.write("#include <amigoSensRHS.h>\n\n")

    fid.write("\n\t/* *** Definition of the states *** */\n\n")
    for i, nm in enumerate(m.st_names):
        fid.write(f"#define\t{nm} Ith(y,{i})\n")
    fid.write("#define iexp amigo_model->exp_num\n")

    fid.write("\n\t/* *** Definition of the sates derivative *** */\n\n")
    for i, nm in enumerate(m.st_names):
        fid.write(f"#define\td{nm} Ith(ydot,{i})\n")

    fid.write("\n\t/* *** Definition of the parameters *** */\n\n")
    for i, nm in enumerate(m.par_names):
        fid.write(f"#define\t{nm} (*amigo_model).pars[{i}]\n")

    for i in range(m.n_stimulus):
        fid.write(
            f"#define {m.stimulus_names[i]}\t((*amigo_model).controls_v[{i}]"
            f"[(*amigo_model).index_t_stim]+(t-(*amigo_model).tlast)"
            f"*(*amigo_model).slope[{i}][(*amigo_model).index_t_stim])\n")

    fid.write("\n\t/* *** Definition of the algebraic variables *** */\n\n")

    # algebraic intermediate variables (AMIGO_gen_C.m:94-129): equations that
    # are neither dX=... nor a plain state assignment declare a threadprivate
    # double. Only names not matching a state name are declared.
    declared: set[str] = set()
    c_species = [re.split(r"\s", nm)[0] for nm in m.st_names]
    for eq in m.eqns:
        eqn = convert_power_operator(eq)
        mm = re.search(r"\w*=", eqn)
        if mm is None or re.match(r"^d\w+=", eqn):
            continue
        if mm.group(0)[:-1] in c_species:
            continue
        var_name = re.sub(r"=[\x20-\x7E]+$", "", eqn)
        if not var_name or var_name in m.st_names or var_name in declared:
            continue
        # AMIGO_gen_C.m:114-123: names longer than 3 chars are skipped when
        # they look like C control keywords; short names are always declared
        if len(var_name) > 3 and (var_name[:4] in ("for(", "else")
                                  or var_name[:3] == "if("):
            continue
        fid.write(f"\tdouble\t{var_name};\n")
        fid.write(f"#pragma omp threadprivate({var_name})\n")
        declared.add(var_name)

    _rhs(fid, inputs)
    _jac(fid, inputs)
    _sensrhs(fid, inputs)
    _obs(fid, inputs)
    _sens_obs(fid, inputs)

    fid.write("\n\n\nvoid amigo_Y_at_tcon(void* data,realtype t, N_Vector y){\n")
    fid.write("\tAMIGO_model* amigo_model=(AMIGO_model*)data;\n\n")
    fid.write("\n}\n")

    return fid.getvalue()


def write_c(inputs: Inputs, path: str) -> str:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(gen_c(inputs))
    return path
