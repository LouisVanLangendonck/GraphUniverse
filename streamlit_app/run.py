"""
Run script for the MMSB Streamlit application.
"""

import os
import sys
import subprocess

def run_streamlit_app():
    """Run the Streamlit application."""
    # Get the path to app.py
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_app", "app.py")
    
    # Run streamlit
    cmd = ["streamlit", "run", app_path, "--server.port=8501"]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("Streamlit app stopped.")

if __name__ == "__main__":
    run_streamlit_app()