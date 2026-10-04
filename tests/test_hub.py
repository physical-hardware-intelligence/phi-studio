"""Hub models: login, inspect, fit check, verified download. No network: every HfApi call and
hf_hub_download is replaced, and HF_HUB_OFFLINE is forced on so a call that slips through raises
instead of reaching the Hub."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
from collections import namedtuple
from pathlib import Path
from typing import Any

import pytest

from phi_studio import hub, rigspec

# WHY a mark, not a module-level importorskip: the compatibility and local_models tests need no
# huggingface_hub and must still run where it is missing.
needs_hub = pytest.mark.skipif(
    importlib.util.find_spec("huggingface_hub") is None, reason="huggingface_hub is not installed"
)

FAKE = "hf_FAKEtokenDoNotLeak0123456789abcdef"
SHA = "a" * 40
RID = "someone/act_so101"

SINGLE = """\
robot:
  type: so101_follower
  id: f
  port: /dev/tty.F
  cameras:
    front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}
teleop:
  type: so101_leader
  id: l
  port: /dev/tty.L
"""

BIMANUAL = """\
robot:
  type: bi_so_follower
  id: bf
  left_arm_config:
    port: /dev/tty.FL
    cameras:
      wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}
  right_arm_config:
    port: /dev/tty.FR
  cameras:
    top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}
teleop:
  type: bi_so_leader
  id: bl
  left_arm_config:
    port: /dev/tty.LL
  right_arm_config:
    port: /dev/tty.LR
"""


def _feat(kind: str, *shape: int) -> dict[str, Any]:
    return {"type": kind, "shape": list(shape)}


def _model(cams: dict[str, tuple[int, ...]], state: int = 6, action: int = 6) -> dict[str, Any]:
    inputs = {f"observation.images.{k}": _feat("VISUAL", *s) for k, s in cams.items()}
    inputs["observation.state"] = _feat("STATE", state)
    return {"input_features": inputs, "output_features": {"action": _feat("ACTION", action)}}


ACT_CONFIG = {"type": "act", **_model({"front": (3, 480, 640)})}


def _problems(lines: list[str]) -> list[str]:
    return [x for x in lines if not x.startswith(hub.WARNING)]


# -- compatibility (no huggingface_hub needed) ---------------------------------------------------


def test_fits_single_arm():
    assert hub.compatibility(ACT_CONFIG, rigspec.parse(SINGLE)) == []


def test_missing_camera_is_a_problem_and_extra_rig_camera_a_warning():
    out = hub.compatibility(_model({"wrist": (3, 480, 640)}), rigspec.parse(SINGLE))
    assert len(_problems(out)) == 1 and "wrist" in _problems(out)[0]
    warns = [x for x in out if x.startswith(hub.WARNING)]
    assert len(warns) == 1 and "observation.images.front" in warns[0]


def test_image_shape_mismatch_names_both_sizes():
    out = hub.compatibility(_model({"front": (3, 224, 224)}), rigspec.parse(SINGLE))
    assert len(out) == 1 and "224x224" in out[0] and "640x480" in out[0]


def test_width_and_height_are_not_swapped():
    # 480 wide, 640 high: the same numbers as the rig, the other way round.
    out = hub.compatibility(_model({"front": (3, 640, 480)}), rigspec.parse(SINGLE))
    assert len(out) == 1 and "480x640" in out[0]


def test_single_arm_model_on_bimanual_rig():
    spec = rigspec.parse(BIMANUAL)
    out = hub.compatibility(_model({"top": (3, 480, 640), "left_wrist": (3, 480, 640)}), spec)
    assert len(out) == 2
    assert all("12 joints (2 arms x 6)" in x for x in out)
    assert any("state has 6" in x for x in out) and any("action has 6" in x for x in out)


def test_bimanual_model_fits_bimanual_rig():
    spec = rigspec.parse(BIMANUAL)
    model = _model({"top": (3, 480, 640), "left_wrist": (3, 480, 640)}, state=12, action=12)
    assert hub.compatibility(model, spec) == []


def test_bimanual_model_on_single_arm_rig():
    out = hub.compatibility(_model({"front": (3, 480, 640)}, 12, 12), rigspec.parse(SINGLE))
    assert len(out) == 2 and all("6 joints (1 arm x 6)" in x for x in out)


def test_camera_without_device_is_a_problem():
    spec = rigspec.parse(SINGLE.replace("index_or_path: 0", "index_or_path: TBD"))
    out = hub.compatibility(ACT_CONFIG, spec)
    assert len(out) == 1 and "no device" in out[0]


def test_camera_without_size_is_a_warning():
    spec = rigspec.parse(SINGLE.replace(", width: 640, height: 480", ""))
    out = hub.compatibility(ACT_CONFIG, spec)
    assert len(out) == 1 and out[0].startswith(hub.WARNING)


def test_depth_model_and_sim_state_are_problems():
    model = _model({"front": (1, 480, 640)})
    model["input_features"]["observation.environment_state"] = _feat("ENV", 16)
    out = hub.compatibility(model, rigspec.parse(SINGLE))
    assert any("1-channel" in x for x in out) and any("environment_state" in x for x in out)


def test_no_input_features_is_a_warning():
    out = hub.compatibility({"input_features": None, "output_features": {}}, rigspec.parse(SINGLE))
    assert out[0].startswith(hub.WARNING) and not _problems(out)


# -- local_models (no huggingface_hub needed) ----------------------------------------------------


def _snapshot(root: Path, rid: str, sha: str, files: dict[str, bytes]) -> Path:
    snap = root / f"models--{rid.replace('/', '--')}" / "snapshots" / sha
    for name, data in files.items():
        (snap / name).parent.mkdir(parents=True, exist_ok=True)
        (snap / name).write_bytes(data)
    return snap


def test_local_models_finds_policies_newest_first(tmp_path):
    import os

    old = _snapshot(tmp_path, "lerobot/act_x", "1" * 40,
                    {"config.json": b'{"type": "act"}', "model.safetensors": b"w"})  # fmt: skip
    new = _snapshot(tmp_path, "me/smol-v2", "2" * 40, {"config.json": b'{"type": "smolvla"}'})
    _snapshot(tmp_path, "bert/base", "3" * 40, {"config.json": b'{"model_type": "bert"}'})
    _snapshot(tmp_path, "broken/json", "4" * 40, {"config.json": b"{nope"})
    _snapshot(tmp_path, "no/config", "5" * 40, {"model.safetensors": b"w"})
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    got = hub.local_models(tmp_path)
    assert [m["repo_id"] for m in got] == ["me/smol-v2", "lerobot/act_x"]
    assert got[0] | {"modified": 0} == {
        "repo_id": "me/smol-v2", "revision": "2" * 40, "path": str(new),
        "policy_type": "smolvla", "has_weights": False, "modified": 0,
    }  # fmt: skip
    assert got[1]["has_weights"] is True and got[1]["policy_type"] == "act"


def test_local_models_empty_or_missing_cache(tmp_path):
    assert hub.local_models(tmp_path / "nowhere") == []


# -- fakes for the Hub ---------------------------------------------------------------------------


@pytest.fixture
def hh(monkeypatch, tmp_path):
    """huggingface_hub with every way out to the network shut, and a fake token in the env."""
    import huggingface_hub
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setenv("HF_TOKEN", FAKE)
    monkeypatch.delenv("HF_OIDC_RESOURCE", raising=False)

    def no_network(*a: Any, **k: Any) -> Any:
        raise AssertionError("a test reached for the network")

    for name in ("model_info", "whoami"):
        monkeypatch.setattr(huggingface_hub.HfApi, name, no_network)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_network)
    return huggingface_hub


def _http_error(hh: Any, cls_name: str, status: int, message: str) -> Exception:
    import httpx

    resp = httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/api/x"))
    return getattr(hh.errors, cls_name)(message, response=resp)


class FakeRepo:
    """One repo at one commit: files with their bytes, served by fake model_info and
    hf_hub_download into a real-shaped cache folder."""

    def __init__(self, hh: Any, monkeypatch: Any, files: dict[str, bytes], lfs: set[str]):
        self.hh, self.files, self.lfs = hh, files, lfs
        self.calls: list[tuple[str, str | None]] = []  # (filename, cache_dir)
        self.corrupt: dict[str, bytes] = {}  # what to write instead of the real bytes
        self.on_chunk: Any = None  # called after each chunk with (filename, bytes so far)
        self.chunk = 4
        repo = self

        def model_info(self_api: Any, repo_id: str, **kw: Any) -> Any:
            return repo.info(repo_id)

        def hf_hub_download(repo_id: str, filename: str, **kw: Any) -> str:
            return repo.download(repo_id, filename, **kw)

        monkeypatch.setattr(hh.HfApi, "model_info", model_info)
        monkeypatch.setattr(hh, "hf_hub_download", hf_hub_download)

    def info(self, repo_id: str) -> Any:
        sibs = []
        for name, data in self.files.items():
            s: dict[str, Any] = {"rfilename": name, "size": len(data), "blobId": "b" * 40}
            if name in self.lfs:
                sha = hashlib.sha256(data).hexdigest()
                s["lfs"] = {"size": len(data), "sha256": sha, "pointerSize": 130}
            sibs.append(s)
        return self.hh.hf_api.ModelInfo(
            id=repo_id, sha=SHA, private=True, gated=False, siblings=sibs,
            lastModified="2026-09-17T09:37:40.000Z",
            cardData={"license": "apache-2.0", "datasets": "me/pick_cube", "base_model": "x/y"},
        )  # fmt: skip

    def download(self, repo_id: str, filename: str, **kw: Any) -> str:
        cache = kw.get("cache_dir")
        self.calls.append((filename, cache))
        data = self.corrupt.get(filename, self.files[filename])
        snap = Path(cache) / f"models--{repo_id.replace('/', '--')}" / "snapshots" / kw["revision"]
        dest = snap / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".incomplete")
        bar = kw["tqdm_class"](total=len(data), initial=0, desc=filename)
        try:  # like _download_to_tmp_and_move: a failure leaves no partial file
            with bar as b, tmp.open("wb") as f:
                for i in range(0, len(data), self.chunk):
                    f.write(data[i : i + self.chunk])
                    b.update(len(data[i : i + self.chunk]))
                    b.update_transfer(len(data[i : i + self.chunk]))
                    if self.on_chunk:
                        self.on_chunk(filename, i + self.chunk)
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
        return str(dest)


FILES = {
    "config.json": json.dumps(ACT_CONFIG).encode(),
    "model.safetensors": bytes(range(256)) * 4,
    "README.md": b"# act",
}


@pytest.fixture
def repo(hh, monkeypatch):
    return FakeRepo(hh, monkeypatch, dict(FILES), lfs={"model.safetensors"})


# -- whoami --------------------------------------------------------------------------------------


@needs_hub
def test_whoami_ok_returns_only_name_and_orgs(hh, monkeypatch):
    answer = {"name": "parv", "orgs": [{"name": "phi"}, {"name": "lerobot"}],
              "auth": {"accessToken": {"displayName": FAKE, "role": "read"}}}  # fmt: skip
    monkeypatch.setattr(hh.HfApi, "whoami", lambda self, *a, **k: answer)
    assert hub.whoami() == {"user": "parv", "orgs": ["phi", "lerobot"], "error": None}


@needs_hub
def test_whoami_not_logged_in(hh, monkeypatch):
    def raise_(self, *a, **k):
        raise hh.errors.LocalTokenNotFoundError("no token")

    monkeypatch.setattr(hh.HfApi, "whoami", raise_)
    got = hub.whoami()
    assert got["user"] is None and got["orgs"] == []
    assert "Not logged in" in got["error"] and "`hf auth login`" in got["error"]


@needs_hub
def test_whoami_offline_names_the_fix(hh, monkeypatch):
    import httpx

    def raise_(self, *a, **k):
        raise httpx.ConnectError(f"dns failed while sending Bearer {FAKE}")

    monkeypatch.setattr(hh.HfApi, "whoami", raise_)
    got = hub.whoami()
    assert "Cannot reach" in got["error"] and "`hf auth login`" in got["error"]
    assert FAKE not in json.dumps(got)


@needs_hub
def test_whoami_bad_token(hh, monkeypatch):
    def raise_(self, *a, **k):
        raise _http_error(hh, "HfHubHTTPError", 401, f"Invalid user token {FAKE}")

    monkeypatch.setattr(hh.HfApi, "whoami", raise_)
    got = hub.whoami()
    assert "--force" in got["error"] and FAKE not in json.dumps(got)


def test_cli_name_matches_the_installed_entry_point():
    import importlib.metadata as md

    if importlib.util.find_spec("huggingface_hub") is None:
        pytest.skip("huggingface_hub is not installed")
    eps = {e.name: e.value for e in md.entry_points(group="console_scripts")}
    assert eps.get(hub.LOGIN_CMD.split()[0]) == "huggingface_hub.cli.hf:main"


# -- the token never leaks -----------------------------------------------------------------------


def _raises_with_token(hh: Any, kind: str) -> Exception:
    import httpx

    msg = f"boom Authorization: Bearer {FAKE} token={FAKE}"
    return {
        "plain": RuntimeError(msg),
        "http": _http_error(hh, "HfHubHTTPError", 500, msg),
        "auth": _http_error(hh, "HfHubHTTPError", 401, msg),
        "missing": _http_error(hh, "RepositoryNotFoundError", 404, msg),
        "gated": _http_error(hh, "GatedRepoError", 403, msg),
        "offline": httpx.ConnectError(msg),
    }[kind]


@needs_hub
@pytest.mark.parametrize("kind", ["plain", "http", "auth", "missing", "gated", "offline"])
def test_token_never_leaks(hh, monkeypatch, tmp_path, kind):
    exc = _raises_with_token(hh, kind)

    def raise_(*a, **k):
        raise exc

    monkeypatch.setattr(hh.HfApi, "whoami", raise_)
    monkeypatch.setattr(hh.HfApi, "model_info", raise_)
    assert FAKE not in json.dumps(hub.whoami())
    calls = (
        lambda: hub.inspect_model(RID),
        lambda: hub.download(RID, None, tmp_path, lambda d, t: None, threading.Event()),
    )
    for call in calls:
        with pytest.raises(hub.HubError) as e:
            call()
        assert FAKE not in str(e.value) and "Bearer" not in str(e.value)
        assert e.value.__cause__ is None and e.value.__suppress_context__


@needs_hub
def test_token_never_leaks_from_a_pasted_token(hh):
    with pytest.raises(hub.HubError) as e:
        hub.inspect_model(f"oops {FAKE}")
    assert FAKE not in str(e.value)


# -- inspect_model -------------------------------------------------------------------------------


@needs_hub
def test_inspect_reads_metadata_and_config_only(repo):
    info = hub.inspect_model(RID)
    assert [c[0] for c in repo.calls] == ["config.json"]  # no weights
    assert not Path(repo.calls[0][1]).exists()  # the throwaway cache is gone
    assert info["policy_type"] == "act" and info["revision"] == SHA
    assert info["cameras"] == {"observation.images.front": [3, 480, 640]}
    assert info["state_shape"] == [6] and info["action_shape"] == [6]
    assert info["total_size"] == sum(len(b) for b in FILES.values())
    assert {f["path"]: f["lfs"] for f in info["files"]}["model.safetensors"] is True
    assert info["has_weights"] is True and info["private"] is True and info["gated"] is False
    assert info["last_modified"] == "2026-09-17T09:37:40+00:00"
    assert info["card"] == {"datasets": ["me/pick_cube"], "license": "apache-2.0",
                            "base_model": ["x/y"]}  # fmt: skip
    json.dumps(info)  # the server sends it as JSON


@needs_hub
def test_inspect_takes_a_pasted_url(repo, monkeypatch, hh):
    seen = []
    real = hh.HfApi.model_info

    def spy(self, repo_id, **kw):
        seen.append(repo_id)
        return real(self, repo_id, **kw)

    monkeypatch.setattr(hh.HfApi, "model_info", spy)
    hub.inspect_model(f"https://huggingface.co/{RID}/tree/main")
    assert seen == [RID]


@needs_hub
@pytest.mark.parametrize("bad", ["", "act", "a/b/c", "a--b/c", "no spaces/x"])
def test_inspect_refuses_bad_ids(hh, bad):
    with pytest.raises(hub.HubError, match="not a model id"):
        hub.inspect_model(bad)


@needs_hub
@pytest.mark.parametrize(
    "kind, cls, status, words",
    [
        ("missing", "RepositoryNotFoundError", 404, "No model"),
        ("gated", "GatedRepoError", 403, "gated"),
        ("revision", "RevisionNotFoundError", 404, "no revision"),
    ],
)
def test_inspect_readable_hub_errors(hh, monkeypatch, kind, cls, status, words):
    def raise_(self, *a, **k):
        raise _http_error(hh, cls, status, "server text")

    monkeypatch.setattr(hh.HfApi, "model_info", raise_)
    with pytest.raises(hub.HubError, match=words) as e:
        hub.inspect_model(RID, revision="v9")
    if kind != "revision":
        assert "hf auth login" in str(e.value)


@needs_hub
def test_inspect_not_a_policy_without_config(hh, monkeypatch):
    FakeRepo(hh, monkeypatch, {"pytorch_model.bin": b"w"}, lfs=set())
    with pytest.raises(hub.HubError, match="no config.json"):
        hub.inspect_model(RID)


@needs_hub
def test_inspect_not_a_policy_without_type(hh, monkeypatch):
    FakeRepo(hh, monkeypatch, {"config.json": b'{"model_type": "bert"}'}, lfs=set())
    with pytest.raises(hub.HubError, match="no policy `type`"):
        hub.inspect_model(RID)


# -- download ------------------------------------------------------------------------------------


@needs_hub
def test_download_verifies_and_reports_bytes(repo, tmp_path):
    seen: list[tuple[int, int]] = []
    snap = hub.download(RID, None, tmp_path, lambda d, t: seen.append((d, t)), threading.Event())
    total = sum(len(b) for b in FILES.values())
    assert snap == tmp_path / "models--someone--act_so101" / "snapshots" / SHA
    assert all((snap / n).read_bytes() == b for n, b in FILES.items())
    assert seen[0] == (0, total) and seen[-1] == (total, total)
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)
    assert repo.calls[-1][0] == "config.json"  # last, so a cut-short download is not listed
    assert hub.local_models(tmp_path)[0]["has_weights"] is True


@needs_hub
def test_download_skips_files_already_there(repo, tmp_path):
    _snapshot(tmp_path, RID, SHA, {"model.safetensors": FILES["model.safetensors"]})
    seen: list[tuple[int, int]] = []
    hub.download(RID, None, tmp_path, lambda d, t: seen.append((d, t)), threading.Event())
    assert "model.safetensors" not in [c[0] for c in repo.calls]
    assert seen[0][0] == len(FILES["model.safetensors"])


@needs_hub
def test_download_cancel_mid_file_stops_and_leaves_no_config(repo, tmp_path):
    cancel = threading.Event()
    repo.on_chunk = lambda name, n: cancel.set() if name == "model.safetensors" and n > 64 else None
    with pytest.raises(hub.Cancelled):
        hub.download(RID, None, tmp_path, lambda d, t: None, cancel)
    names = [c[0] for c in repo.calls]
    assert names[-1] == "model.safetensors" and "config.json" not in names
    snap = tmp_path / "models--someone--act_so101" / "snapshots" / SHA
    assert not (snap / "model.safetensors").exists() and not list(snap.glob("*.incomplete"))
    assert hub.local_models(tmp_path) == []


@needs_hub
def test_download_cancel_before_start_fetches_nothing(repo, tmp_path):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(hub.Cancelled):
        hub.download(RID, None, tmp_path, lambda d, t: None, cancel)
    assert repo.calls == []


@needs_hub
def test_download_cancel_wrapped_by_the_xet_path(repo, tmp_path, monkeypatch, hh):
    """The Xet path may hand our exception back as another type; a set cancel still wins."""
    cancel = threading.Event()

    def xet_like(*a, **k):
        cancel.set()
        raise RuntimeError("xet: Python exception updating progress")

    monkeypatch.setattr(hh, "hf_hub_download", xet_like)
    with pytest.raises(hub.Cancelled):
        hub.download(RID, None, tmp_path, lambda d, t: None, cancel)


@needs_hub
@pytest.mark.parametrize("bad, words", [(b"\x00" * 1024, "sha256"), (b"short", "bytes")])
def test_download_verification_failure_names_the_file_and_keeps_it(repo, tmp_path, bad, words):
    repo.corrupt["model.safetensors"] = bad
    with pytest.raises(hub.VerificationError, match=words) as e:
        hub.download(RID, None, tmp_path, lambda d, t: None, threading.Event())
    assert "model.safetensors" in str(e.value)
    snap = tmp_path / "models--someone--act_so101" / "snapshots" / SHA
    assert (snap / "model.safetensors").read_bytes() == bad  # nothing deleted


@needs_hub
def test_download_refuses_without_disk_space(repo, tmp_path, monkeypatch):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(hub.shutil, "disk_usage", lambda p: usage(100, 90, 10))
    with pytest.raises(hub.HubError, match="Not enough disk space"):
        hub.download(RID, None, tmp_path, lambda d, t: None, threading.Event())
    assert repo.calls == []


@needs_hub
def test_download_hub_error_is_readable(repo, tmp_path, monkeypatch, hh):
    def raise_(*a, **k):
        raise hh.errors.LocalEntryNotFoundError("connection reset")

    monkeypatch.setattr(hh, "hf_hub_download", raise_)
    with pytest.raises(hub.HubError, match="Cannot reach"):
        hub.download(RID, None, tmp_path, lambda d, t: None, threading.Event())
