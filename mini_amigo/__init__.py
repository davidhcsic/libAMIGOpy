"""mini_amigo — Python port of the minimal AMIGO costMex runner (basic_amigo).

See CONSTITUTION.md at the repository root for the binding rules.
"""

from mini_amigo.errors import BuildError, InputError, SimError
from mini_amigo.inputs import Inputs, Experiment, Model, PEsol, IvpSol, check_inputs
from mini_amigo.gen_c import gen_c, convert_power_operator
from mini_amigo.build import build_libs, build_dylib
from mini_amigo.prep import prep, Session
from mini_amigo.sim import sim, cost, SimResults, CostResults

__all__ = [
    "BuildError", "InputError", "SimError",
    "Inputs", "Experiment", "Model", "PEsol", "IvpSol", "check_inputs",
    "gen_c", "convert_power_operator",
    "build_libs", "build_dylib",
    "prep", "Session",
    "sim", "cost", "SimResults", "CostResults",
]
