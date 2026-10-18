# =============================================================================
# region SCRIPT DETAILS
# =============================================================================
# CSP Week 06 Assignment: Red Team the Claim, Audit the Model
# Objective: One entry point for the whole audit, matching Weeks 4-5 (the installers call this file).
#
# Usage:
#   python scripts/Assignment_06_Red_Team_Audit.py                 # full run (~25-35 min on CPU)
#   AUDIT_QUICK=1 python scripts/Assignment_06_Red_Team_Audit.py   # PowerShell: $env:AUDIT_QUICK='1'
#       quick = rebuild the risk register from existing outputs/audit_results.json only (~1 min, no models run)
#
# Full run order:
#   1. canary_experiment.py   CPU retrain of the Week 3 LoRA recipe with inserted canaries
#   2. build_notebook.py      builds + executes Assignment_06_Red_Team_Audit.ipynb (probes, slices, contamination)
#   3. build_register.py      Meridian_AI_Risk_Register.xlsx from outputs/audit_results.json
# Reads the Week 3 adapters and data from week03_model/ (a bundled copy; see its SOURCE.md), or from W3_DIR if set.
# =============================================================================
# endregion
# =============================================================================
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEPS = [("canary retrain", "canary_experiment.py"), ("audit notebook", "build_notebook.py"), ("risk register", "build_register.py")]


def run(step: str, script: str) -> None:
    print(f"\n[audit] {step}: {script}", flush=True)
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT)


if __name__ == "__main__":
    quick = os.environ.get("AUDIT_QUICK") == "1"
    if quick and not (ROOT / "outputs" / "audit_results.json").exists():
        raise SystemExit("AUDIT_QUICK=1 needs outputs/audit_results.json; run the full audit once first.")
    for step, script in (STEPS[2:] if quick else STEPS):
        run(step, script)
    print("\n[audit] done. Deliverables: Assignment_06_Red_Team_Audit.ipynb, Meridian_AI_Risk_Register.xlsx, Assignment_06_Red_Team_Audit_Analysis.md")
