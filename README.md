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
| Set up | Twelve steps from a new rig to a recorded dataset. Finds each arm's port by unplugging it, finds the cameras by picture, and saves both to `robot-config.yaml`. |
| Camera align | Compares live cameras with a dataset's resting frame and says which way to move each one. |
| Models | Hub login status, search or repo id, a fit check against your rig from the model's config, verified download, and the `lerobot-rollout` command. Recordings upload as private. |
| Train | On Explorer over ssh and SLURM (read-only checks, chained parts that resume from checkpoints, live loss chart, checkpoint fetch, cancel), or on this Mac in the terminal panel. A real submit has not been run yet. |
| 3D view | The SO-101 CAD model (Apache-2.0, see `NOTICE`) driven by live joint readings, on its own page and on Teleoperate and Run policy. The joint mapping is not yet checked on a physical arm. |
| Workspace points | A metric point cloud from one camera picture: Depth Anything V2 Small gives inverse depth up to scale and shift, and Studio fits both against the table plane. Needs a one-time 99 MB model download. Accuracy on a real rig depends on the camera pose you set, which is not measured. |

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
npm --prefix web test
npm --prefix web run build
```

- `src/phi_studio/`: the server (aiohttp), the robot worker process, LeRobot command builder (`rigspec.py`),
  config writer (`configedit.py`), Hub client (`hub.py`, `hub_api.py`), cameras (`cameras.py`), Set up and
  camera align (`setup_api.py`, `align.py`), training (`train.py`, `hpc.py`, `train_api.py`), the 3D model and
  forward kinematics (`robot_model.py`, `scene_api.py`), workspace points (`recon.py`, `depth_model.py`,
  `recon_api.py`), terminal (`terminal.py`).
- `web/`: the React app. `npm run build` writes it into `src/phi_studio/static/`, which the server serves.
- `web/tests/`: Node tests for the page logic that runs without a browser (`npm test`).
- `tests/`: unit and end-to-end tests on the mock rig. Tests that need LeRobot, OpenCV or the Hub client skip
  when those are not installed.
- `web/scripts/check-context-restore.mjs`: a headless browser check that the 3D view survives a lost GPU context
  without WebGL warnings. It needs Playwright, which Studio does not install; the file says how to run it.

## License

Apache 2.0. See `LICENSE`.
