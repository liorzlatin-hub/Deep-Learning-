from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    base = Path(__file__).resolve().parent

    scripts = [
        "01_tensor_advanced.py",
        "02_tensor_serialization.py",
        "03_tensor_images.py",
        "04_tensor_time_series.py",
        "05_complete_enumeration_vs_gd.py",
        "05_complete_enumeration_vs_gd_plot.py",
        "06_manual_gradient_descent_normalization.py",
        "06_manual_gradient_descent_normalization_plot.py",
        "07_quadratic_gradient_descent.py",
        "07_quadratic_gradient_descent_plot.py",
        "08_autograd_gradient_descent.py",
        "08_autograd_gradient_descent_plot.py",
    ]

    print("Running Session 2 demos...")
    print(f"Python: {sys.executable}")
    print()

    failed: list[tuple[str, int]] = []

    for script in scripts:
        script_path = base / script
        print(f"=== {script} ===")
        result = subprocess.run([sys.executable, str(script_path)], cwd=base)
        if result.returncode != 0:
            failed.append((script, result.returncode))
        print()

    if failed:
        print("Completed with failures:")
        for script, code in failed:
            print(f"- {script}: exit code {code}")
        return 1

    print("All scripts completed successfully.")
    print("Check artifacts/ for generated plots and saved tensors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
