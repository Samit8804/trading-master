#!/usr/bin/env python3
"""CrowdWisdomQuant — CLI entry point.

Usage::

    # From the project root directory:
    python main.py scrape

    # Or with PYTHONPATH set:
    cd ..
    python -m crowdwisdom_quant.main scrape
"""

from __future__ import annotations

import sys
from pathlib import Path

# When running ``python main.py`` from within the package root,
# we need the *parent* directory on sys.path so that
# ``import crowdwisdom_quant`` resolves correctly.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent  # e.g. E:\TradeStrategy Ai

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from crowdwisdom_quant.cli.entry import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
