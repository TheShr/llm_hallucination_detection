from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    app_path = project_root / "app" / "frontend" / "streamlit_app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Streamlit app not found at {app_path}")

    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    subprocess.run(command, check=True)
