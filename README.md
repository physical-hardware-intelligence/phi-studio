# Phi Studio

A local web app for SO-101 robot arms, single or bimanual. It walks a rig from first plug-in to running and
scoring a policy: find ports, set motor ids, calibrate, teleoperate, set up and align cameras, record, train,
import models from Hugging Face, run a policy and evaluate it.

Studio runs on your Mac and opens in your browser. It listens on 127.0.0.1 only, and every launch prints a link
with a fresh token.

## Status

| Part | State |
|---|---|
| Mock rig (1 or 2 pairs, cameras, faults) | Works. Every page runs on it. |
| Real arms | Through LeRobot's own commands, run from Studio's terminal panel with one click. Studio's own real-arm backend is not built yet, so `--hardware` stops with a message. |
| Terminal panel | Your shell, in the phi env, in the phi checkout. Run buttons type a command for you. |
| robot-config.yaml edits | Studio changes single values in place and keeps your comments. It saves a backup first. |
| Hugging Face | Login status, model inspect, fit check against your rig, verified download. |
| Camera align | Compares live cameras with a dataset's resting frame and says which way to move each one. |
| Datasets | Every LeRobot dataset on this Mac (v3.0 and v2.1), read with pyarrow, single arm or bimanual, with a health bar each. |
| Episode inspector | Cameras, a 3D SO-101 twin (measured solid, commanded ghost, tool path), a timeline and per-joint signals on one playhead. Keyboard: Space, ←/→, [ ], N, B, G. |
| Analysis | Follower lag, tracking error with the lag removed, idle time, grasps, gripper squeeze, one-frame jumps, frozen joints, smoothness, outliers. Thresholds measured on our recordings (`analysis.py`). |
| Notes and issues | Shared live across windows, kept in `studio.db` in the data folder. Episodes marked bad drop out of `--dataset.episodes`. |

## Install

Studio needs Python 3.12. Install it into the env that has LeRobot 0.6.0, so the Run buttons and camera
previews use the same LeRobot as your commands.

```bash
conda activate phi
pip install -e .
npm --prefix web ci
npm --prefix web run build
```

## Run

```bash
phi-studio
```

Options: `--pairs 2` for a bimanual mock rig, `--port 8766`, `--no-browser`, `--rig-dir PATH` for the folder
with `robot-config.yaml` (default: the current folder, then `~/phi`), `--data-dir PATH` for evals and config
backups (default `~/.cache/phi/studio`).

## Develop

```bash
pip install -e ".[dev]"
ruff check src tests
mypy src
pytest -q
npm --prefix web run typecheck
npm --prefix web run build
```

- `src/phi_studio/`: the server (aiohttp), the robot worker process, LeRobot command builder (`rigspec.py`),
  config writer (`configedit.py`), Hub client (`hub.py`), cameras (`cameras.py`), camera align (`align.py`),
  terminal (`terminal.py`).
- Data: `datasets.py` (LeRobot datasets, no torch), `analysis.py` (episode and dataset analysis), `kinematics.py`
  (SO-101 forward kinematics, checked against MuJoCo), `notes.py` (SQLite notes), `data_api.py` (HTTP routes and
  note commands).
- `assets/so101/`: the SO-101 model for the 3D view and the kinematics, compiled from TheRobotStudio's MJCF by
  `scripts/build_so101_model.py <phi>/simulation/model` (see its `NOTICE.md`).
- `web/`: the React app. `npm run build` writes it into `src/phi_studio/static/`, which the server serves.
- `tests/`: unit and end-to-end tests on the mock rig. Tests that need LeRobot, OpenCV or the Hub client skip
  when those are not installed.

## License

Apache 2.0. See `LICENSE`.
