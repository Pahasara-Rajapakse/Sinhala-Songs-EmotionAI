import subprocess
import sys
import os

# Path to the main Streamlit app
app_path = os.path.join(os.path.dirname(__file__), "Home.py")

# Launch Streamlit
subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
