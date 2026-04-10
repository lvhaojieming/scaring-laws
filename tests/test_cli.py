"""Smoke tests for the starter CLI."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from scaring_laws.cli import build_status, main


class CliTests(unittest.TestCase):
    def test_build_status_mentions_readiness(self) -> None:
        self.assertIn("scaring-laws is ready.", build_status())

    def test_main_prints_status_and_returns_zero(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream):
            exit_code = main([])

        self.assertEqual(exit_code, 0)
        self.assertIn("Version: 0.1.0", stream.getvalue())
