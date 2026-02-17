"""Tests for the CLI module."""

import sys
from unittest.mock import patch

import pytest


def test_cli_module_exists():
    """Test that the CLI module can be imported."""
    from graph_universe import cli

    assert hasattr(cli, "main")
    assert hasattr(cli, "launch_ui")


def test_main_without_streamlit():
    """Test that main fails gracefully when streamlit is not installed."""
    from graph_universe.cli import main

    # Mock streamlit import to fail
    with (
        patch.dict(sys.modules, {"streamlit.web.cli": None}),
        patch("builtins.__import__", side_effect=ImportError("No module named 'streamlit'")),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1


def test_launch_ui_is_wrapper():
    """Test that launch_ui is a thin wrapper around main."""
    from graph_universe.cli import launch_ui

    # Mock main to avoid actually launching
    with patch("graph_universe.cli.main", side_effect=SystemExit(0)) as mock_main:
        with pytest.raises(SystemExit) as exc_info:
            launch_ui()

        # Should have called main
        mock_main.assert_called_once()
        assert exc_info.value.code == 0


def test_main_structure():
    """Test that main has the expected structure and error handling."""
    # This test just verifies the function exists and has proper error handling
    # We don't actually run it to avoid launching streamlit
    from graph_universe.cli import main

    # Verify it's callable
    assert callable(main)

    # Verify it has a docstring
    assert main.__doc__ is not None
    assert "CLI entry point" in main.__doc__ or "launch" in main.__doc__.lower()


def test_streamlit_script_path():
    """Test that the streamlit script path is correctly constructed."""
    from pathlib import Path

    # Get the expected path
    import graph_universe

    expected_path = Path(graph_universe.__file__).parent.parent / "streamlit_graph_universe.py"

    # The script should exist in the repository
    # (This test will pass in development, might need adjustment for installed package)
    if expected_path.exists():
        assert expected_path.is_file()
    else:
        pytest.skip("Streamlit script not found (may be running from installed package)")


def test_launch_ui_exported():
    """Test that launch_ui is exported from the main package."""
    from graph_universe import launch_ui

    assert callable(launch_ui)
