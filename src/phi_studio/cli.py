"""`phi-studio`: start Studio on this Mac and open it in the browser."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

STUDIO_DATA = Path.home() / ".cache" / "phi" / "studio"


def find_rig_dir(given: str | None) -> Path | None:
    """The folder that holds robot-config.yaml (the phi checkout). WHY a search: Studio is its own
    repo now, and the rig's files live in the club repo, which git keeps them out of."""
    if given:
        return Path(given).expanduser().resolve()
    env = os.environ.get("PHI_STUDIO_RIG_DIR")
    if env:
        return Path(env).expanduser().resolve()
    for d in (Path.cwd(), Path.home() / "phi"):
        if (d / "robot-config.yaml").is_file():
            return d.resolve()
    return None


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phi-studio",
        description="Phi Studio: set up, calibrate, teleoperate, record, train, run and evaluate "
        "SO-101 arms in a local web app.",
    )
    rig = p.add_mutually_exclusive_group()
    rig.add_argument("--mock", dest="rig", action="store_const", const="mock",
                     help="A simulated rig, even with arms plugged in.")  # fmt: skip
    rig.add_argument("--hardware", dest="rig", action="store_const", const="hardware",
                     help="Real arms, from the robot-config.yaml onboarding wrote. Default: real "
                     "arms when an SO-101 driver board is plugged in and robot-config.yaml "
                     "exists, else the mock.")  # fmt: skip
    p.add_argument("--pairs", type=int, choices=(1, 2), default=1,
                   help="Leader and follower pairs: 1, or 2 for bimanual.")  # fmt: skip
    p.add_argument("--port", type=int, default=8765, help="Local port. Studio binds 127.0.0.1.")
    p.add_argument("--no-browser", dest="browser", action="store_false",
                   help="Do not open the browser.")  # fmt: skip
    p.add_argument("--data-dir", default=str(STUDIO_DATA),
                   help="Where Studio keeps evals, config backups, mock calibrations.")  # fmt: skip
    p.add_argument("--rig-dir", default=None,
                   help="Folder with robot-config.yaml. Default: here, then ~/phi.")  # fmt: skip
    p.add_argument("--assistant-model", default="",
                   help="Model for the Claude assistant, such as sonnet or opus.")  # fmt: skip
    return p


# WHY by USB vendor: SO-101 driver boards use WCH's CH343 or CH340 USB serial chips (vendor 0x1A86).
# Listing ports opens none of them.
BOARD_VENDORS = {0x1A86}


def boards_plugged_in() -> list[str]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return sorted(p.device for p in list_ports.comports() if p.vid in BOARD_VENDORS)


def rig_spec(rig: str | None, pairs: int, configs: list[Path],
             boards: Callable[[], list[str]] = boards_plugged_in) -> dict[str, object] | str:
    """The worker's rig, or why Studio cannot start with it. `configs`: where robot-config.yaml
    may be, in the order the server reads them (files.root_order)."""
    config = next((c for c in configs if c.is_file()), None)
    if rig is None:
        found = boards()
        rig = "hardware" if found and config is not None else "mock"
        why = (f"{len(found)} SO-101 driver board{'s' if len(found) != 1 else ''} plugged in"
               if found else "no SO-101 driver board plugged in")  # fmt: skip
        if found and config is None:
            why += ", and no robot-config.yaml yet: set the rig up on Home"
        print(f"Rig: {'real arms' if rig == 'hardware' else 'mock'} ({why}). "
              "--mock or --hardware chooses.", flush=True)  # fmt: skip
    if rig == "mock":
        return {"kind": "mock", "pairs": pairs}
    if config is None:
        return ("No robot-config.yaml yet. Start phi-studio without --hardware, set up the rig "
                "on Home, then start it again with --hardware.")  # fmt: skip
    return {"kind": "lerobot", "config": str(config)}


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    rig_dir = find_rig_dir(args.rig_dir)
    data_dir = Path(args.data_dir).expanduser()
    configs = [d / "robot-config.yaml" for d in (rig_dir, data_dir) if d is not None]
    spec = rig_spec(args.rig, args.pairs, configs)
    if isinstance(spec, str):
        print(spec, file=sys.stderr)
        return 1
    from phi_studio.server import PortInUse, serve

    try:
        serve(spec, port=args.port, open_browser=args.browser,
              data_dir=data_dir, rig_dir=rig_dir,
              assistant_model=args.assistant_model or None)  # fmt: skip
    except PortInUse as e:
        print(f"{e} Close it, or run with --port {e.port + 1}.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
