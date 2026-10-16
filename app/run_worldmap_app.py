# run_worldmap_app.py
# ============================================================
# One-click launcher for the DeepCNT 3.0 Streamlit application
# ============================================================

from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
APP_PATH = SCRIPT_DIR / "app_oneclick_tabs.py"


def main() -> None:
    if not APP_PATH.is_file():
        raise FileNotFoundError(
            f"Streamlit app not found:\n  {APP_PATH}"
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP_PATH),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()