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

    # To display the help of subparsers
    # We have to setup argument in the parent cli interface.
    train_parser = sub.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--config", type=str, required=True, help="path/to/config.py"
    )
    train_parser.add_argument(
        "--disable-ddp",
        action="store_true",
        default=False,
        help="Disable multiple GPUs",
    )

    base = sub.add_parser("run", help="Feature extraction + tracking")
    base.add_argument("--config", required=True, help="path/to/config.py")
    base = sub.add_parser("track", help="Tracking only")
    base.add_argument("--config", required=True, help="path/to/config.py")
    # Parse only the subcommand; leave all other flags as 'unknown'
    ns, rest = parser.parse_known_args()
    # Omit the first argument and passing the remaining arguments via sys.argv
    if ns.cmd == "run":
        # Forward everything after 'run' to the real CLI in run_ascent.py
        from ascent.tools.run_ascent import main as run_ascent_main

        sys.argv = ["ascent-run"] + sys.argv[2:]
        run_ascent_main()
    elif ns.cmd == "train":
        # Forward everything after 'train' to the real CLI in train.py
        from ascent.tools.train import main as train_main

        sys.argv = ["ascent-train"] + sys.argv[2:]
        train_main()

    elif ns.cmd == "track":
        from ascent.tools.run_track import main as run_track_main

        sys.argv = ["ascent-track"] + sys.argv[2:]
        run_track_main()

    return 0
