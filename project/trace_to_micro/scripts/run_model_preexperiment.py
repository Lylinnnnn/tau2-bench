#!/usr/bin/env python3
"""Thin wrapper for the complete model-based pre-experiment."""

import sys

from trace_to_micro.cli import main

if __name__ == "__main__":
    main(["model-preexperiment", *sys.argv[1:]])
