"""LeRobot policy models on the Hugging Face Hub: who is logged in, what a model expects, whether it
fits this rig, and a verified download into the Hugging Face cache.

Every rule cites the installed huggingface_hub 1.25.1 or LeRobot 0.6.0 source. huggingface_hub is
imported inside each function because CI and a bare checkout may not have it, and the rest of
Studio must still import.

The token: this module never puts it in a return value, a log line or an exception message. It
never reads the token itself except to scrub it, and every message that can carry Hub text goes
through _scrub. Errors are raised `from None` so a traceback does not print the Hub's own.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from phi_studio.rigspec import RigSpec

# The `hf` entry point is huggingface_hub.cli.hf:main (entry_points.txt), its auth group is added at
# cli/hf.py:100 and `login` is cli/auth.py:38-39. `huggingface-cli` only prints a deprecation
# notice and exits 1 (cli/deprecated_cli.py:9-35).
LOGIN_CMD = "hf auth login"
CONFIG = "config.json"  # LeRobot reads the policy from this file (configs/policies.py:192-202)
WEIGHTS = "model.safetensors"  # what PreTrainedPolicy.from_pretrained loads (pretrained.py:196-203)
WARNING = "Warning: "  # compatibility() lines with this prefix do not stop a run
# LeRobot's feature keys (utils/constants.py:23, 25, 33) and feature types (configs/types.py:20-26).
IMAGES = "observation.images."
STATE = "observation.state"
ACTION = "action"
# [JUDGEMENT] Long enough for a slow link, short enough that the Models page does not hang.
TIMEOUT_S = 15.0
_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9_]{8,}|(?i:bearer)\s+\S+")
_URL_RE = re.compile(r"^https?://(?:www\.)?(?:huggingface\.co|hf\.co)/", re.IGNORECASE)
# The policy types LeRobot 0.6.0 registers (policies/*/configuration_*.py register_subclass;
# tests/test_hub.py checks this against PreTrainedConfig.get_known_choices() where LeRobot is
# installed). LeRobot's model card adds the type as a tag and as model_name
# (policies/pretrained.py:363-367), so a tag from this set names the policy type.
POLICY_TYPES = frozenset({
    "act", "diffusion", "eo1", "evo1", "fastwam", "gaussian_actor", "groot", "lingbot_va",
    "molmoact2", "multi_task_dit", "pi0", "pi05", "pi0_fast", "smolvla", "tdmpc", "vla_jepa",
    "vqbet", "wall_x", "xvla",
})  # fmt: skip
# The tag every LeRobot policy carries: the card sets library_name="lerobot" and adds the tag
# (policies/pretrained.py:363-366). [RUN 2026-10-04] list_models(filter="lerobot") returned 30 of
# 30 rows with library_name lerobot; filter="robotics" returned GGUF language models.
LIBRARY = "lerobot"
SORTS = ("downloads", "likes", "last_modified", "trending_score", "created_at")  # hf_api.py:2462
# What a search row needs, and no more: without expand the Hub leaves last_modified empty.
_EXPAND = ["author", "downloads", "likes", "lastModified", "tags", "gated", "private", "cardData"]


class HubError(Exception):
    """A message the UI can show as is. Never carries the token."""


class Cancelled(HubError):
    """The download stopped because the cancel event was set."""


class VerificationError(HubError):
    """A downloaded file does not match the Hub's size or sha256. Nothing was deleted."""


@dataclass(frozen=True)
class _File:
    path: str  # relative to the snapshot folder
    size: int
    sha256: str | None  # LFS files only: the Hub lists no sha256 for files kept in git


def _hub() -> Any:
    try:
        # WHY each submodule by name: the package loads attributes lazily and raises
        # AttributeError for `utils`, `errors` and `constants` until they are imported.
        import huggingface_hub
        import huggingface_hub.constants
        import huggingface_hub.errors
        import huggingface_hub.utils
    except ImportError:
        raise HubError(
            "huggingface_hub is not installed. Run: pip install -e '.[studio]'"
        ) from None
    return huggingface_hub


def _scrub(text: str) -> str:
    """Text with the stored token, and anything shaped like a token, replaced."""
    try:
        token = _hub().get_token()
    except Exception:  # WHY: get_token raises when OIDC is on and fails (utils/_auth.py:73-74)
        token = None
    if token:
        text = text.replace(token, "<token>")
    return _TOKEN_RE.sub("<token>", text)


def _offline(hh: Any, e: BaseException) -> bool:
    """A connection failure rather than an answer from the Hub. HF_HUB_OFFLINE raises
    OfflineModeIsEnabled (utils/_http.py:259-262); requests go through httpx (utils/_http.py:31);
    hf_hub_download turns a failed metadata call into LocalEntryNotFoundError
    (file_download.py:1872-1877)."""
    try:
        import httpx
    except ImportError:
        transport: tuple[type[BaseException], ...] = ()
    else:
        transport = (httpx.TransportError,)
    err = hh.errors
    return isinstance(e, (err.OfflineModeIsEnabled, err.LocalEntryNotFoundError, *transport))


def _readable(e: BaseException, repo_id: str | None = None, revision: str | None = None) -> str:
    """One plain sentence for a Hub failure, with the fix."""
    if isinstance(e, HubError):
        return str(e)
    hh = _hub()
    err = hh.errors
    status = getattr(getattr(e, "response", None), "status_code", None)
    if isinstance(e, err.LocalTokenNotFoundError):
        return f"Not logged in to Hugging Face on this Mac. Run `{LOGIN_CMD}` in a terminal."
    if _offline(hh, e):
        why = ("HF_HUB_OFFLINE is set" if isinstance(e, err.OfflineModeIsEnabled)
               else "no connection")  # fmt: skip
        return (f"Cannot reach the Hugging Face Hub ({why}). Check this Mac's internet "
                f"connection; if the login still fails once online, run "
                f"`{LOGIN_CMD}`.")  # fmt: skip
    # WHY this order: GatedRepoError subclasses RepositoryNotFoundError (errors.py:331).
    if repo_id and isinstance(e, err.GatedRepoError):
        return (f"{repo_id} is gated. Open https://huggingface.co/{repo_id} while logged in, "
                f"accept its terms, then try again. Not logged in: run `{LOGIN_CMD}` "
                "first.")  # fmt: skip
    if repo_id and isinstance(e, err.RepositoryNotFoundError):
        return (f"No model {repo_id} that this Mac can see on the Hub. Check the id; if it is "
                f"private, run `{LOGIN_CMD}` with an account that has access.")  # fmt: skip
    if isinstance(e, err.RevisionNotFoundError):
        return f"{repo_id} has no revision {revision!r}."
    if isinstance(e, err.DisabledRepoError):
        return f"{repo_id} has been disabled on the Hub."
    if isinstance(e, err.HfHubHTTPError) and status == 401:
        return (f"The Hub refused the stored Hugging Face token (invalid or expired). Run "
                f"`{LOGIN_CMD} --force` in a terminal.")  # fmt: skip
    return _scrub(f"Hugging Face Hub error ({type(e).__name__}): {e}")


# -- login ---------------------------------------------------------------------------------------


def whoami() -> dict[str, Any]:
    """Who the token stored on this Mac belongs to: {"user", "orgs", "error"}. HfApi.whoami reads
    the stored token itself (hf_api.py:2326-2335); its answer has `name` and `orgs[].name`
    (cli/auth.py:171-174)."""
    try:
        info = _hub().HfApi().whoami()
    except Exception as e:
        return {"user": None, "orgs": [], "error": _readable(e)}
    name = info.get("name") if isinstance(info, dict) else None
    if not name:
        return {"user": None, "orgs": [], "error": "The Hub did not say who this token belongs to."}
    orgs = [str(o["name"]) for o in info.get("orgs") or [] if isinstance(o, dict) and o.get("name")]
    return {"user": str(name), "orgs": orgs, "error": None}


# -- search --------------------------------------------------------------------------------------


def policy_type_of(tags: list[str], card: Any = None) -> str | None:
    """The policy type a Hub listing states, or None when it does not say or says two things.
    LeRobot's model card puts the type in the tags and in model_name (policies/pretrained.py:
    363-367). [RUN 2026-10-04] lerobot/smolvla_base and lerobot/pi0_base carry neither, so this is
    often None; inspect_model reads config.json for the real answer."""
    votes = {t for t in tags if t in POLICY_TYPES}
    name = card.get("model_name") if card is not None else None
    if isinstance(name, str) and name in POLICY_TYPES:
        votes.add(name)
    return votes.pop() if len(votes) == 1 else None


def _search_error(hh: Any, e: BaseException) -> str:
    if _offline(hh, e):
        why = ("HF_HUB_OFFLINE is set" if isinstance(e, hh.errors.OfflineModeIsEnabled)
               else "no connection")  # fmt: skip
        return (f"Cannot search the Hugging Face Hub ({why}). Models already on this Mac are "
                "still listed under On this Mac.")  # fmt: skip
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status == 429:
        return "The Hub is limiting requests from this Mac. Wait a minute, then search again."
    if isinstance(status, int) and status >= 500:
        return f"The Hub had a server error (HTTP {status}). Try again in a minute."
    return _readable(e)


def search_models(
    query: str = "", limit: int = 30, sort: str = "downloads"
) -> list[dict[str, Any]]:
    """LeRobot policies on the Hub whose id matches `query`, most downloaded first; an empty query
    lists the most popular. Raises HubError with a readable message.
    WHY filter="lerobot": the Hub matches it against tags, and every LeRobot policy card carries
    it (see LIBRARY). The search words go to the Hub's own `search` (hf_api.py:2559-2565)."""
    hh = _hub()
    q = " ".join(str(query or "").split())
    if len(q) > 200:
        raise HubError("Search for 200 characters or fewer.")
    if sort not in SORTS:
        raise HubError(f"Sort by one of {', '.join(SORTS)}.")
    n = max(1, min(int(limit), 100))
    try:
        rows = list(hh.HfApi().list_models(filter=LIBRARY, search=q or None, sort=sort, limit=n,
                                           expand=_EXPAND))  # fmt: skip
    except Exception as e:
        raise HubError(_search_error(hh, e)) from None
    out = []
    for m in rows:
        tags = [str(t) for t in m.tags or []]
        out.append({
            "repo_id": m.id,
            "author": m.author or m.id.split("/")[0],
            "downloads": m.downloads,
            "likes": m.likes,
            "last_modified": m.last_modified.isoformat() if m.last_modified else None,
            "tags": tags,
            "policy_type": policy_type_of(tags, m.card_data),
            "gated": m.gated,  # False, "auto" or "manual"
            "private": m.private,
        })  # fmt: skip
    return out


# -- inspect -------------------------------------------------------------------------------------


def _repo_id(text: str) -> str:
    """owner/name from what a person pastes: an id or a model page URL."""
    raw = str(text).strip()
    rid = _URL_RE.sub("", raw)
    if rid != raw:  # a URL: keep owner/name, drop /tree/main and the like
        rid = "/".join(rid.strip("/").split("/")[:2])
    try:
        _hub().utils.validate_repo_id(rid)  # utils/_validators.py:93-148
        ok = rid.count("/") == 1
    except HubError:
        raise
    except Exception:
        ok = False
    if not ok:
        raise HubError(_scrub(f"{raw!r} is not a model id. Use owner/name, for example "
                              "lerobot/smolvla_base."))  # fmt: skip
    return rid


def _files(mi: Any) -> list[_File]:
    """The repo's files with real sizes. With files_metadata=True the Hub sends each file's size
    and, for LFS files, the LFS size and sha256 (hf_api.py:1066-1083)."""
    out = []
    for s in mi.siblings or []:
        lfs = s.lfs
        size = lfs.size if lfs is not None else s.size
        if size is None:
            raise HubError(f"The Hub sent no size for {s.rfilename}.")
        out.append(_File(s.rfilename, int(size), lfs.sha256 if lfs is not None else None))
    return out


def _features(raw: Any, where: str) -> dict[str, dict[str, Any]]:
    """input_features or output_features as LeRobot writes PolicyFeature: a type and a shape
    (configs/types.py:42-45). null means "infer from the dataset" (configs/policies.py:58-60)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise HubError(f"config.json {where} is not a mapping.")
    out = {}
    for key, ft in raw.items():
        shape = ft.get("shape") if isinstance(ft, dict) else None
        if not isinstance(shape, list) or not all(
            isinstance(n, int) and not isinstance(n, bool) for n in shape
        ):
            raise HubError(f"config.json {where}.{key} has no usable shape.")
        out[str(key)] = {"type": str(ft.get("type")), "shape": shape}
    return out


def _silent_bar(on_bytes: Callable[[int], None]) -> Any:
    """A stand-in for tqdm that hands each byte count to on_bytes and prints nothing.
    huggingface_hub builds a class that is not a tqdm subclass with only its keyword arguments
    (utils/tqdm.py:337-338). WHY update_transfer exists: without it the Xet path opens a second,
    real tqdm bar on stderr (utils/_xet_progress_reporting.py:87-89, 109-111). Only update() counts
    bytes: the HTTP path calls update and update_transfer for the same chunk
    (file_download.py:437-441)."""

    class _Bar:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.total = kwargs.get("total")
            self.n = 0

        def __enter__(self) -> _Bar:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def update(self, n: float | None = 1) -> None:
            self.n += int(n or 0)
            on_bytes(int(n or 0))

        def update_transfer(self, n: float | None = 1) -> None:
            return None

        def set_postfix_str(self, *args: Any, **kwargs: Any) -> None:
            return None

        def set_transfer_postfix_str(self, *args: Any, **kwargs: Any) -> None:
            return None

        def refresh(self) -> None:
            return None

        def close(self) -> None:
            return None

    return _Bar


def _read_config(hh: Any, repo_id: str, sha: str) -> dict[str, Any]:
    """config.json at this commit. WHY a throwaway cache: a config.json left in the real cache
    would make local_models() list a model whose weights were never downloaded."""
    with tempfile.TemporaryDirectory(prefix="phi-hub-") as tmp:
        try:
            path = hh.hf_hub_download(repo_id, CONFIG, revision=sha, cache_dir=tmp,
                                      etag_timeout=TIMEOUT_S,
                                      tqdm_class=_silent_bar(lambda n: None))  # fmt: skip
            text = Path(path).read_text(encoding="utf-8")
        except Exception as e:
            raise HubError(_readable(e, repo_id)) from None
    try:
        cfg = json.loads(text)
    except ValueError:
        raise HubError(f"{repo_id}'s config.json is not valid JSON.") from None
    if not isinstance(cfg, dict):
        raise HubError(f"{repo_id}'s config.json is not a JSON object.")
    return cfg


def _as_list(v: Any) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)] if v else []


def _policy(cfg: dict[str, Any], where: str) -> dict[str, Any]:
    """The policy type and the features a LeRobot config.json declares."""
    ptype = cfg.get("type")  # draccus writes the registered choice name here (policies.py:218-220)
    if not isinstance(ptype, str) or not ptype:
        raise HubError(f"{where} has no policy `type`, so it is not a LeRobot policy.")
    inputs = _features(cfg.get("input_features"), "input_features")
    outputs = _features(cfg.get("output_features"), "output_features")
    return {
        "policy_type": ptype,
        "input_features": inputs,
        "output_features": outputs,
        # Image shapes are channels, height, width (utils/feature_utils.py:159-167).
        "cameras": {k: f["shape"] for k, f in inputs.items() if f["type"] == "VISUAL"},
        "state_shape": inputs[STATE]["shape"] if STATE in inputs else None,
        "action_shape": outputs[ACTION]["shape"] if ACTION in outputs else None,
        "resizes_images": _resizes_images(ptype, cfg),
    }


def _resizes_images(ptype: str, cfg: dict[str, Any]) -> bool:
    """True when the policy scales and pads every camera frame to its own size before the model
    sees it, so a camera at another resolution still runs. LeRobot itself never checks frame
    sizes (rollout/context.py:308-326 compares camera names only)."""
    if ptype in ("pi0", "pi05", "pi0_fast"):
        # unconditional: modeling_pi0.py:1212-1213, modeling_pi05.py:1190-1191,
        # modeling_pi0_fast.py:1078-1079
        return True
    # modeling_smolvla.py:431-432; default (512, 512) at configuration_smolvla.py:45
    if ptype == "smolvla":
        return cfg.get("resize_imgs_with_padding", [512, 512]) is not None
    if ptype == "xvla":  # modeling_xvla.py:327-328; default None, configuration_xvla.py:89
        return cfg.get("resize_imgs_with_padding") is not None
    return False


def inspect_model(repo_id: str, revision: str | None = None) -> dict[str, Any]:
    """What a LeRobot policy on the Hub expects, read from its metadata and config.json only; no
    weights are downloaded. Raises HubError with a readable message."""
    hh = _hub()
    rid = _repo_id(repo_id)
    try:
        mi = hh.HfApi().model_info(rid, revision=revision, files_metadata=True, timeout=TIMEOUT_S)
    except Exception as e:
        raise HubError(_readable(e, rid, revision)) from None
    files = _files(mi)
    if CONFIG not in {f.path for f in files}:
        raise HubError(f"{rid} has no config.json, so it is not a LeRobot policy.")
    cfg = _read_config(hh, rid, mi.sha)
    card = mi.card_data  # README metadata; CardData.get at repocard_data.py:228
    lic = card.get("license") if card is not None else None
    return {
        "repo_id": rid,
        "revision": mi.sha,
        **_policy(cfg, f"{rid}'s config.json"),
        "files": [{"path": f.path, "size": f.size, "lfs": f.sha256 is not None} for f in files],
        "total_size": sum(f.size for f in files),
        "has_weights": WEIGHTS in {f.path for f in files},
        "last_modified": mi.last_modified.isoformat() if mi.last_modified else None,
        "private": mi.private,
        "gated": mi.gated,
        "card": {
            "datasets": _as_list(card.get("datasets") if card is not None else None),
            "license": str(lic) if lic else None,
            "base_model": _as_list(card.get("base_model") if card is not None else None),
        },
    }


# -- fit -----------------------------------------------------------------------------------------


def _size(shape: list[int]) -> str:
    return f"{shape[2]}x{shape[1]}" if len(shape) == 3 else "x".join(map(str, shape))


def compatibility(info: dict[str, Any], spec: RigSpec) -> list[str]:
    """What stops this model running on this rig, in plain English. Lines that start with WARNING
    do not stop a run; no other line means it fits."""
    out: list[str] = []
    inputs: dict[str, Any] = info.get("input_features") or {}
    outputs: dict[str, Any] = info.get("output_features") or {}
    ptype = info.get("policy_type")
    if isinstance(ptype, str) and ptype not in POLICY_TYPES:
        # A plugin package can add a type: lerobot-rollout imports every installed lerobot_policy_*
        # package first (lerobot_rollout.py:246; utils/import_utils.py:214-228).
        out.append(WARNING + f"LeRobot 0.6.0 has no built-in policy type `{ptype}`, so "
                   "lerobot-rollout loads this model only if a LeRobot plugin that adds it is "
                   "installed.")  # fmt: skip
    if not inputs:
        out.append(WARNING + "The model's config.json lists no input features, so Studio cannot "
                   "check its cameras or state.")  # fmt: skip
    model_cams = {k: f["shape"] for k, f in inputs.items() if f.get("type") == "VISUAL"}
    rig_cams = {c.feature: c for c in spec.cameras}
    # WHY backticks: the Models page shows a `name` as code, so a camera key reads as a key.
    names = [f"`{k.removeprefix(IMAGES)}`" for k in rig_cams]
    has = (f"your rig has {join_and(names)}" if len(names) > 1
           else f"your rig has only {names[0]}" if names
           else "your rig has no cameras")  # fmt: skip
    # WHY one line for all of them: three near-identical lines for camera1..3 bury the fix.
    missing = [f"`{k.removeprefix(IMAGES)}`" for k in model_cams if k not in rig_cams]
    if missing:
        one = len(missing) == 1
        what = f"a camera named {missing[0]}" if one else f"cameras named {join_and(missing)}"
        # lerobot-rollout maps camera names with --rename_map (rollout/configs.py:242), and the
        # Models page offers only cameras that have a device.
        fix = ("Add " + ("a camera" if one else "cameras") + " to robot-config.yaml first."
               if not rig_cams
               else "robot-config.yaml gives none of them a device yet, so set their "
               "`index_or_path` first, then map " + ("one to it." if one else "them.")
               if all(c.source is None for c in rig_cams.values())
               else "Map one of them to it when you run the model, or rename a camera in "
               "robot-config.yaml." if one
               else "Map one of your cameras to each when you run the model, or rename cameras "
               "in robot-config.yaml.")  # fmt: skip
        out.append(f"This model expects {what}; {has}. {fix}")
    fed_by: dict[str, str] = info.get("fed_by") or {}  # rig key -> model key, as_seen_by_rig()
    for key, shape in model_cams.items():
        name = f"`{key.removeprefix(IMAGES)}`"
        cam = rig_cams.get(key)
        if cam is None:
            continue
        if cam.source is None:
            out.append(f"The model takes the {name} camera, but robot-config.yaml gives it no "
                       f"device, so LeRobot gets no frames from it.")  # fmt: skip
            continue
        # A SO follower's frame is (height, width, 3) from the camera config, after rotation
        # (so_follower.py:70-75; camera_opencv.py:124-126 swaps only the capture size).
        w, h = cam.fields.get("width"), cam.fields.get("height")
        if len(shape) != 3:
            out.append(f"The model's {name} input has shape {shape}, not channels x height x "
                       f"width, so Studio cannot compare it with the camera.")  # fmt: skip
            continue
        if shape[0] != 3:
            out.append(f"The model takes {shape[0]}-channel images from {name}; the rig's "
                       f"cameras give 3 (RGB).")  # fmt: skip
        if not isinstance(w, int) or not isinstance(h, int):
            out.append(WARNING + f"The rig's {name} camera sets no width and height, so Studio "
                       f"cannot check it against the model's {_size(shape)}.")  # fmt: skip
        elif [h, w] != shape[1:]:
            own = fed_by.get(key)  # the model's own name for a camera the mapping renamed
            size = (f"The model was trained on {_size(shape)} frames from {name}; the rig's "
                    f"{name} camera is set to {w}x{h}." if own is None
                    else f"The model's `{own.removeprefix(IMAGES)}` camera was trained on "
                    f"{_size(shape)} frames; the rig's {name} camera, which feeds it, is set to "
                    f"{w}x{h}.")  # fmt: skip
            if info.get("resizes_images"):
                out.append(WARNING + size + f" `{ptype}` scales and pads each frame to its own "
                           "size, so it runs, but a view unlike its training data can still "
                           "lower its success.")  # fmt: skip
            else:
                out.append(size)
    for key in rig_cams:
        if key not in model_cams:
            out.append(WARNING + f"The rig's `{key.removeprefix(IMAGES)}` camera (`{key}`) is not "
                       "an input of this model, so the model ignores it.")  # fmt: skip
    # Simulator state; an SO follower observes only joints and cameras (so_follower.py:80-81).
    for key, f in inputs.items():
        if f.get("type") == "ENV":
            out.append(f"The model takes `{key}`, simulator state that an SO-101 rig does not "
                       f"produce.")  # fmt: skip
    # State and action are the same motors (so_follower.py:80-85).
    joints = len(spec.action_features())
    arms = joints // 6
    if not joints:
        out.append("robot-config.yaml has no follower arm, so the model has nothing to drive.")
        return out
    rig = f"this rig has {joints} joints ({arms} arm{'s' if arms > 1 else ''} x 6)"
    for what, key, feats in (("state", STATE, inputs), ("action", ACTION, outputs)):
        if key in feats and feats[key]["shape"] != [joints]:
            out.append(f"The model's {what} has {_size(feats[key]['shape'])} numbers; {rig}.")
    return out


def join_and(items: list[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _plain(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.removeprefix(IMAGES).lower())


def propose_rename_map(model_cams: list[str], rig_cams: list[str]) -> dict[str, str]:
    """A first guess at lerobot-rollout's --rename_map: {rig camera key: model camera key} for each
    model camera the rig lacks by name. LeRobot renames the robot's observation keys to the
    model's (processor/rename_processor.py:43-49; rollout/configs.py:241).
    Names first: equal once case and punctuation go, or one inside the other (`wrist` and
    `wrist_cam`). Then order: what is left pairs up in the order each side lists it."""
    missing = [k for k in model_cams if k not in rig_cams]
    free = [k for k in rig_cams if k not in model_cams]
    out: dict[str, str] = {}
    for want in list(missing):
        w = _plain(want)
        match = next((r for r in free if _plain(r) == w), None) or next(
            (r for r in free if w and _plain(r) and (w in _plain(r) or _plain(r) in w)), None
        )
        if match is not None:
            out[match] = want
            free.remove(match)
            missing.remove(want)
    for want, rig in zip(missing, free, strict=False):
        out[rig] = want
    return out


def as_seen_by_rig(info: dict[str, Any], rename_map: dict[str, str]) -> dict[str, Any]:
    """The model's features with each renamed camera under the rig's own key, so compatibility()
    checks the camera that will really feed it."""
    back = {model: rig for rig, model in rename_map.items()}
    inputs = info.get("input_features") or {}
    return {**info, "input_features": {back.get(k, k): f for k, f in inputs.items()},
            "fed_by": {rig: model for model, rig in back.items() if model in inputs}}  # fmt: skip


# -- download ------------------------------------------------------------------------------------


def _free_bytes(path: Path) -> int:
    for p in (path, *path.parents):
        if p.exists():
            return shutil.disk_usage(p).free
    return 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot(root: Path, repo_id: str, sha: str) -> Path:
    """Where the HF cache keeps this commit: models--owner--name/snapshots/<sha>
    (file_download.py:718-726, 2016-2019)."""
    return root / f"models--{repo_id.replace('/', '--')}" / "snapshots" / sha


def download(
    repo_id: str,
    revision: str | None,
    cache_dir: Path | None,
    progress: Callable[[int, int], None],
    cancel: threading.Event,
) -> Path:
    """Download every file of the model into the HF cache and verify it; return the snapshot
    folder. progress(done_bytes, total_bytes) is called as bytes land. Raises Cancelled,
    VerificationError or HubError. Blocks: run it off the event loop."""
    hh = _hub()
    rid = _repo_id(repo_id)
    try:
        mi = hh.HfApi().model_info(rid, revision=revision, files_metadata=True, timeout=TIMEOUT_S)
    except Exception as e:
        raise HubError(_readable(e, rid, revision)) from None
    files = _files(mi)
    root = Path(cache_dir).expanduser() if cache_dir else Path(hh.constants.HF_HUB_CACHE)
    snap = _snapshot(root, rid, mi.sha)
    have = {f.path for f in files if (snap / f.path).is_file()}
    total = sum(f.size for f in files)
    need = sum(f.size for f in files if f.path not in have)
    # WHY here: huggingface_hub only warns when a file will not fit (file_download.py:729-753).
    free = _free_bytes(root)
    if need > free:
        raise HubError(f"Not enough disk space for {rid}: it needs {need / 1e9:.2f} GB and "
                       f"{root} has {free / 1e9:.2f} GB free.")  # fmt: skip
    done = total - need
    progress(done, total)

    def on_bytes(n: int) -> None:
        nonlocal done
        done += n
        progress(done, total)
        if cancel.is_set():
            # WHY raising works on the HTTP path: nothing in http_get's chunk loop catches it
            # (file_download.py:437-444) and the partial file is deleted
            # (file_download.py:1951-1954).
            raise Cancelled(f"Download of {rid} cancelled.")

    # WHY config.json last: local_models() lists a snapshot once it has config.json, so a
    # cancelled or failed download must not leave one next to missing weights.
    for f in sorted((f for f in files if f.path not in have), key=lambda f: f.path == CONFIG):
        if cancel.is_set():
            raise Cancelled(f"Download of {rid} cancelled.")
        start = done
        try:
            hh.hf_hub_download(rid, f.path, revision=mi.sha, cache_dir=str(root),
                               etag_timeout=TIMEOUT_S,
                               tqdm_class=cast(Any, _silent_bar(on_bytes)))  # fmt: skip
        except Exception as e:
            # WHY check cancel first: the Xet path calls on_bytes from compiled code, so its
            # exception may come back wrapped in another type, or only once the file is done.
            if cancel.is_set():
                raise Cancelled(f"Download of {rid} cancelled.") from None
            raise HubError(_readable(e, rid, revision)) from None
        done = start + f.size  # WHY: retries and Range resets can skew the summed chunk counts
        progress(done, total)
    if cancel.is_set():
        raise Cancelled(f"Download of {rid} cancelled.")
    _verify(files, snap, cancel)
    return snap


def _verify(files: list[_File], snap: Path, cancel: threading.Event) -> None:
    """Each file's size against the Hub, and the sha256 of each LFS file.
    WHY both: huggingface_hub checks only the byte count, and only on the HTTP path
    (file_download.py:417-422, 469-474). xet_get checks neither in Python
    (file_download.py:477-574; whatever hf_xet checks is compiled and not read here). A file
    already in the cache is returned unchecked (file_download.py:1070-1084, 1188-1193). No sha256
    is computed after any download: the one sha_fileobj call is for local_dir mode
    (file_download.py:1399-1401)."""
    for f in files:
        if cancel.is_set():
            raise Cancelled("Download cancelled during verification.")
        p = snap / f.path
        if not p.is_file():
            raise VerificationError(f"Verification failed: {f.path} is missing from {snap}.")
        got = p.stat().st_size
        if got != f.size:
            raise VerificationError(f"Verification failed: {f.path} is {got} bytes, the Hub lists "
                                    f"{f.size}. Left in place at {p.resolve()}; delete it and "
                                    "download again.")  # fmt: skip
        if f.sha256 is not None:
            digest = _sha256(p)
            if digest != f.sha256:
                raise VerificationError(f"Verification failed: {f.path} has sha256 {digest}, the "
                                        f"Hub lists {f.sha256}. Left in place at {p.resolve()}; "
                                        "delete it and download again.")  # fmt: skip


# -- on disk -------------------------------------------------------------------------------------


def local_models(cache_dir: Path | None = None) -> list[dict[str, Any]]:
    """LeRobot policies in the HF cache, newest first: snapshots whose config.json has a policy
    type. has_weights says whether model.safetensors is there too."""
    if cache_dir is None:
        try:
            root = Path(_hub().constants.HF_HUB_CACHE)  # constants.py:185-191
        except HubError:
            return []
    else:
        root = Path(cache_dir).expanduser()
    out: list[dict[str, Any]] = []
    for repo in root.glob("models--*"):
        # WHY a plain replace: the folder joins type, owner and name with "--" (file_download.py:
        # 718-726) and a repo id may not contain "--" (utils/_validators.py:144-145).
        rid = repo.name.removeprefix("models--").replace("--", "/")
        for snap in (repo / "snapshots").glob("*"):
            try:
                cfg = json.loads((snap / CONFIG).read_text(encoding="utf-8"))
                modified = snap.stat().st_mtime
            except (OSError, ValueError):
                continue
            ptype = cfg.get("type") if isinstance(cfg, dict) else None
            if isinstance(ptype, str) and ptype:
                out.append({"repo_id": rid, "revision": snap.name, "path": str(snap),
                            "policy_type": ptype, "has_weights": (snap / WEIGHTS).is_file(),
                            "modified": modified})  # fmt: skip
    out.sort(key=lambda m: m["modified"], reverse=True)
    return out


def inspect_local(path: str | Path) -> dict[str, Any]:
    """What a downloaded policy expects, from its config.json on disk: the fields compatibility()
    reads, with no network. Raises HubError."""
    p = Path(path) / CONFIG
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise HubError(f"Cannot read {p}: {e}") from None
    if not isinstance(cfg, dict):
        raise HubError(f"{p} is not a JSON object.")
    return _policy(cfg, str(p))


def snapshot_size(path: str | Path) -> int:
    """Bytes of the files in one cache snapshot. Each is a link into the cache's blobs folder
    (file_download.py:1235, 1254), so stat follows it to the real file."""
    total = 0
    for f in Path(path).rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total
