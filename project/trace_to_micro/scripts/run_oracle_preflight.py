"""Thin wrapper for the reference-transition preflight command."""

import sys

from trace_to_micro.cli import main


if __name__ == "__main__":
    main(["oracle-preflight", *sys.argv[1:]])
