#!/usr/bin/env python3
"""Run the public, dependency-free reference-codec verification check."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from test_reference_roundtrip import ReferenceCodecRoundTripTests  # noqa: E402


suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceCodecRoundTripTests)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
