"""The test suite never runs against the real home folder (conftest.py)."""

import os
from pathlib import Path

import pytest

REAL = os.environ.get("PHI_STUDIO_TEST_REAL_HOME") == "1"


@pytest.mark.skipif(REAL, reason="opted into the real home")
def test_tests_run_in_a_throwaway_home_with_no_hf_cache_variables() -> None:
    assert Path.home().name.startswith("phi-studio-test-home-")
    hf = ("HF_HOME", "HF_LEROBOT_HOME", "HF_LEROBOT_CALIBRATION")
    assert not any(os.environ.get(v) for v in hf)
