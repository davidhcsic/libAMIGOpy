"""Toolchain orchestration (CONSTITUTION Art. IV).

build_libs(): make -C libAMIGO all   (static archives)
build_dylib(): make -C libAMIGO python PY_MODEL=<amigoRHS.c>
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from mini_amigo.errors import BuildError

PKG_ROOT = Path(__file__).resolve().parent          # python/mini_amigo
REPO_ROOT = PKG_ROOT.parent.parent                 # port_AMIGO2mini
BASIC_DIR = REPO_ROOT / "basic_amigo"
LIBAMIGO_DIR = BASIC_DIR / "libAMIGO"
LIB_DIR = LIBAMIGO_DIR / "lib_mac64"
RESULTS_DIR = BASIC_DIR / "Results"
LIBOMP_PREFIX = Path("/opt/homebrew/opt/libomp")

STATIC_LIBS = ("libAMIGO.a", "libmxInterface.a", "libf2c.a")


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    # The libAMIGO Makefile relies on $(PWD), so it must run from LIBAMIGO_DIR.
    proc = subprocess.run(
        cmd, cwd=str(cwd or LIBAMIGO_DIR), capture_output=True, text=True,
        env={**os.environ, "PWD": str(cwd or LIBAMIGO_DIR)})
    if proc.returncode != 0:
        raise BuildError(f"command failed ({' '.join(cmd)}):\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout + proc.stderr


def static_libs_present() -> bool:
    return all((LIB_DIR / f).is_file() for f in STATIC_LIBS)


def build_libs(force: bool = False) -> Path:
    """Build libAMIGO.a, libmxInterface.a, libf2c.a via the existing Makefile.

    Idempotent unless force=True (make clean first). Returns the lib dir.
    """
    if not static_libs_present() or force:
        if not LIBOMP_PREFIX.joinpath("include", "omp.h").is_file():
            raise BuildError(
                f"OpenMP not found under {LIBOMP_PREFIX}. Run: brew install libomp")
        if force:
            _run(["make", "clean"])
        _run(["make", "all", "-j4"])
    return LIB_DIR


def build_dylib(odes_file: str | os.PathLike, short_name: str,
                force: bool = False) -> Path:
    """Compile the per-problem shared library (amigoRHS.c + C entry point).

    Returns the dylib path: Results/<short_name>/libamigo_<short_name>.dylib
    """
    odes_file = Path(odes_file).resolve()
    if not odes_file.is_file():
        raise BuildError(f"model C file not found: {odes_file}")

    build_libs(force=False)

    out_dir = RESULTS_DIR / short_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dylib = out_dir / f"libamigo_{short_name}.dylib"

    if dylib.exists() and not force and dylib.stat().st_mtime > odes_file.stat().st_mtime:
        return dylib

    rebuild = force or not dylib.exists() or dylib.stat().st_mtime < odes_file.stat().st_mtime
    if rebuild:
        _run(["make", "python",
              f"PY_MODEL={odes_file}",
              f"PY_DYLIB={dylib}",
              "-B"])
    return dylib
