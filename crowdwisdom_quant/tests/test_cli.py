"""Tests for the CLI entry point and argument parsing."""

import pytest

from crowdwisdom_quant.cli.entry import build_parser, parse_args, main


class TestCliParser:
    """Verify argument parsing behaviour."""

    def test_parse_valid_command(self) -> None:
        args = parse_args(["scrape"])
        assert args.command == "scrape"

    def test_parse_all_commands(self) -> None:
        for cmd in [
            "scrape",
            "preprocess",
            "feature",
            "train",
            "validate",
            "visualize",
            "report",
            "run_all",
        ]:
            args = parse_args([cmd])
            assert args.command == cmd

    def test_parse_no_args_shows_help(self) -> None:
        with pytest.raises(SystemExit):
            parse_args([])

    def test_parse_invalid_command(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["invalid_command"])

    def test_parser_has_epilog(self) -> None:
        parser = build_parser()
        assert parser.epilog is not None
        assert "docs/architecture.md" in parser.epilog

    def test_parser_has_description(self) -> None:
        parser = build_parser()
        assert "Quantitative trading research platform" in parser.description

    def test_parser_accepts_help(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--help"])


class TestCliMain:
    """Verify the main dispatch function."""

    def test_main_scrape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called = False

        def mock_cmd() -> None:
            nonlocal called
            called = True

        monkeypatch.setattr("crowdwisdom_quant.cli.entry.cmd_scrape", mock_cmd)
        monkeypatch.setattr("sys.argv", ["main.py", "scrape"])
        main()
        assert called

    def test_main_report(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called = False

        def mock_cmd() -> None:
            nonlocal called
            called = True

        monkeypatch.setattr("crowdwisdom_quant.cli.entry.cmd_report", mock_cmd)
        monkeypatch.setattr("sys.argv", ["main.py", "report"])
        main()
        assert called

    def test_main_run_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called = False

        def mock_cmd() -> None:
            nonlocal called
            called = True

        monkeypatch.setattr("crowdwisdom_quant.cli.entry.cmd_run_all", mock_cmd)
        monkeypatch.setattr("sys.argv", ["main.py", "run_all"])
        main()
        assert called
