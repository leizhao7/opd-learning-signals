#!/usr/bin/env python3
"""Compatibility entry point; prefer python -m opd train."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opd.train import main

if __name__ == '__main__':
    main()
