"""Every test runs with HOME in a throwaway folder and no Hugging Face cache variables.

WHY: code that falls back to the home folder (setup_api.data_dir -> ~/.cache/phi/studio, LeRobot's
calibration and dataset caches under ~/.cache/huggingface) must never write into the real Studio
data or HF cache of the machine running the tests. 2026-10-05 a test left align-result.json in the
real ~/.cache/phi/studio. This runs when pytest loads conftest, before any test module imports
LeRobot, whose cache paths are fixed at import.

Tests that need a cache set it themselves (monkeypatch.setenv). The two real-machine tests skip
here; to run them against this machine's own files, set PHI_STUDIO_TEST_REAL_HOME=1.
"""

from __future__ import annotations

import os
import tempfile

CACHE_VARS = ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "HF_LEROBOT_HOME",
              "HF_LEROBOT_CALIBRATION", "XDG_CACHE_HOME")  # fmt: skip

if os.environ.get("PHI_STUDIO_TEST_REAL_HOME") != "1":
    os.environ["HOME"] = tempfile.mkdtemp(prefix="phi-studio-test-home-")
    for var in CACHE_VARS:
        os.environ.pop(var, None)
