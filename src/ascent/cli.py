from __future__ import annotations

import argparse
import sys
from importlib.metadata import version


def main() -> int:
    parser = argparse.ArgumentParser(prog="ascent", description="ASCENT command-line")
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"ascent {version('ascent')}",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run", help="Feature extraction + tracking")
    sub.add_parser("train", help="Train a model")

    # Parse only the subcommand; leave all other flags as 'unknown'
    ns, rest = parser.parse_known_args()

    if ns.cmd == "run":
        # Forward everything after 'run' to the real CLI in run_ascent.py
        from ascent.tools.run_ascent import main as run_ascent_main

        sys.argv = ["ascent-run"] + rest
        run_ascent_main()
    elif ns.cmd == "train":
        # Forward everything after 'train' to the real CLI in train.py
        from ascent.tools.train import main as train_main

        sys.argv = ["ascent-train"] + rest
        train_main()

    elif ns.cmd == "track":
        from ascent.tools.run_track import main as run_track_main

        sys.argv = ["ascent-track"] + rest
        run_track_main()

    return 0
