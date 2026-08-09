"""Thin wrapper for the deterministic task inventory command."""

import sys

from trace_to_micro.cli import main


if __name__ == "__main__":
    main(["task-audit", *sys.argv[1:]])
