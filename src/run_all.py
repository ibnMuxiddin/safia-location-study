"""Run the whole pipeline in order. Uses the data committed in data/raw/, so no network is needed.

Usage:  .venv/Scripts/python src/run_all.py            (Windows)
        .venv/bin/python src/run_all.py                (Linux / macOS)
Pass --from NN to start at a step, e.g. --from 07.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = ["01_check_env", "02_scrape_safia", "03_tashkent_filter", "04_check_coords", "05_competitors",
         "06_osm_pois", "06b_population", "07_features", "08_eda", "09_model", "10_top10", "11_map", "12_report"]


def main() -> int:
    start = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else "01"
    for step in STEPS:
        if step[:2] < start[:2]:
            continue
        t0 = time.time()
        print(f"=== {step} ===", flush=True)
        result = subprocess.run([sys.executable, str(SRC / f"{step}.py")], cwd=SRC.parent,
                                capture_output=True, text=True, encoding="utf-8", errors="replace",
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})  # Cyrillic output on Windows
        if result.returncode != 0:
            print(result.stdout[-3000:], result.stderr[-3000:], sep="\n")
            print(f"FAILED at {step}")
            return result.returncode
        print(f"    ok ({time.time() - t0:.0f}s)")
    print("All steps finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
