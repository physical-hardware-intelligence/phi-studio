"""`phi-studio`: start Studio on this Mac and open it in the browser."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
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
    rig.add_argument("--mock", dest="mock", action="store_true", default=True,
                     help="A simulated rig (the default).")  # fmt: skip
    rig.add_argument("--hardware", dest="mock", action="store_false", help="Real arms.")
    p.add_argument("--pairs", type=int, choices=(1, 2), default=1,
                   help="Leader and follower pairs: 1, or 2 for bimanual.")  # fmt: skip
    p.add_argument("--port", type=int, default=8765, help="Local port. Studio binds 127.0.0.1.")
    p.add_argument("--no-browser", dest="browser", action="store_false",
                   help="Do not open the browser.")  # fmt: skip
    p.add_argument("--data-dir", default=str(STUDIO_DATA),
                   help="Where Studio keeps evals, config backups and mock calibrations.")  # fmt: skip
    p.add_argument("--rig-dir", default=None,
                   help="The folder with robot-config.yaml. Default: this folder, then ~/phi.")  # fmt: skip
    p.add_argument("--assistant-model", default="",
                   help="Model for the Claude assistant, such as sonnet or opus.")  # fmt: skip
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.mock:
        print("The hardware backend is not built yet. Run with --mock for now.", file=sys.stderr)
        return 1
    from phi_studio.server import PortInUse, serve

    try:
        serve({"kind": "mock", "pairs": args.pairs}, port=args.port, open_browser=args.browser,
              data_dir=Path(args.data_dir).expanduser(), rig_dir=find_rig_dir(args.rig_dir),
              assistant_model=args.assistant_model or None)  # fmt: skip
    except PortInUse as e:
        print(f"{e} Close it, or run with --port {e.port + 1}.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
