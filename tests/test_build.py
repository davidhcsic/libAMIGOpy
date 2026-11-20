import shutil
import subprocess

import pytest

from mini_amigo import build_libs, build_dylib
from mini_amigo.build import LIB_DIR, LIBAMIGO_DIR, RESULTS_DIR, static_libs_present
from mini_amigo.errors import BuildError


requires_make = pytest.mark.skipif(
    shutil.which("make") is None, reason="make not available")


@requires_make
def test_static_libs_present_after_build():
    out = build_libs()
    assert out == LIB_DIR
    assert static_libs_present()


@requires_make
def test_build_libs_idempotent(capsys):
    build_libs()
    mtime = (LIB_DIR / "libAMIGO.a").stat().st_mtime
    build_libs()  # no force -> no-op
    assert (LIB_DIR / "libAMIGO.a").stat().st_mtime == mtime


@requires_make
def test_build_dylib_for_mini_model(tmp_path):
    from mini_amigo import prep
    from tests.test_inputs import mini_inputs
    inp = mini_inputs(results_folder="py_test_build", short_name="py_test_build")
    s = prep(inp)
    try:
        assert s.dylib_path.is_file()
        assert s.dylib_path.name == "libamigo_py_test_build.dylib"
        assert s.dylib_path.parent == RESULTS_DIR / "py_test_build"
        # rebuild is skipped when up to date
        first = s.dylib_path.stat().st_mtime
        again = build_dylib(s.c_file, "py_test_build")
        assert again.stat().st_mtime == first
    finally:
        s.close()


@requires_make
def test_build_dylib_missing_source_raises():
    with pytest.raises(BuildError, match="not found"):
        build_dylib("/nonexistent/amigoRHS.c", "nope")


@requires_make
def test_build_error_contains_output(tmp_path):
    bad = tmp_path / "amigoRHS.c"
    bad.write_text("this is not valid C;")
    with pytest.raises(BuildError) as exc_info:
        build_dylib(bad, "py_test_bad")
    assert "error" in str(exc_info.value).lower()
