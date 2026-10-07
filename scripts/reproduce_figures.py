#!/usr/bin/env python3
"""Compatibility entry point; prefer python -m opd figures."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opd.figures import main

if __name__ == '__main__':
    main()
