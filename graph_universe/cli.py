"""Command-line interface for GraphUniverse."""

import sys


def main():
    """
    Main CLI entry point for launching the GraphUniverse Streamlit UI.

    This is the primary interface for launching the interactive dashboard.
    It will open a browser window at http://localhost:8501 (or next available port).

    Press Ctrl+C in the terminal to stop the server.
    """
    try:
        import streamlit.web.cli as stcli
    except ImportError:
        print("Error: Streamlit is not installed.")
        print("Install it with: pip install graph-universe[viz]")
        sys.exit(1)

    from pathlib import Path

    # Get the path to the streamlit app
    streamlit_script = Path(__file__).parent / "streamlit_app.py"

    if not streamlit_script.exists():
        print(f"Error: Streamlit app not found at {streamlit_script}")
        sys.exit(1)

    print("=" * 60)
    print("Launching GraphUniverse Dashboard...")
    print("=" * 60)
    print("The dashboard will open in your browser.")
    print("Press Ctrl+C in this terminal to stop the server.")
    print("=" * 60)

    # Launch streamlit
    sys.argv = ["streamlit", "run", str(streamlit_script)]
    sys.exit(stcli.main())


def launch_ui():
    """
    Convenience wrapper to launch the GraphUniverse UI from Python.

    This is a thin wrapper around the CLI command. It will:
    - Open a browser window with the interactive dashboard
    - Block until you stop it (Ctrl+C)

    For most use cases, it's better to use the CLI directly:
        $ graph-universe-ui

    This function is provided for convenience when exploring the API
    in notebooks or interactive sessions.

    Raises:
        SystemExit: When the server is stopped or if streamlit is not installed.

    Example:
        >>> from graph_universe import launch_ui
        >>> launch_ui()  # Opens browser, blocks until Ctrl+C
    """
    main()


if __name__ == "__main__":
    main()
