#!/usr/bin/env python3
"""Prepare the US / small-version ESCI ranking tables."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prepare import prepare_dataset


def main() -> None:
    prepare_dataset()


if __name__ == "__main__":
    main()
