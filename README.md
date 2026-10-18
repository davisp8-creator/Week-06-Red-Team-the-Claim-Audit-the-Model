# Week 06: Red Team the Claim, Audit the Model

Meridian Health Plan pilot-gate package: a red-team of one frontier capability claim (METR time horizons), an audit of the Week 3 adapted model (safety probes, disaggregated fairness on a synthetic member population, contamination tests), and a risk register mapped to NIST AI RMF and ISO/IEC 42001. Original brief: [ASSIGNMENT.md](ASSIGNMENT.md).

| Deliverable | File |
|---|---|
| Risk register (NIST AI RMF + ISO/IEC 42001) | `Meridian_AI_Risk_Register.xlsx` |
| Audit notebook (probes, slice eval, contamination), executed | `Assignment_06_Red_Team_Audit.ipynb` |
| Frontier-claim red team | `Assignment_06_Red_Team_Audit_Analysis.md` |
| Video (5-10 min, camera on) | recorded separately |

## Quick start

| Where | How |
|---|---|
| **Windows** | `powershell -ExecutionPolicy Bypass -File .\Install-Assignment_06_Red_Team_Audit_Windows.ps1` |
| **macOS** | Double-click `Install-Assignment_06_Red_Team_Audit_macOS.command` (first time: right-click, Open, or `chmod +x` it) |

Both installers find Python 3.10+, create a project-local `.venv`, and run `env_setup.py`, which installs what is missing and reports the host (Intel VM, physical PC, Apple silicon). They then offer a **quick rebuild** (~1 min, regenerates the risk register from `outputs/audit_results.json`) or the **full run** (~25-35 min on CPU). Unattended: `-Launch none|quick|full` on Windows, `LAUNCH=none|quick|full` on macOS.

There is no Colab notebook for this week: the audit is CPU-only (GPT-2 124M).

### Prerequisites
- The **Week 3 adapters, data and helper modules** are bundled in `week03_model/` (a read-only copy; origin and commit in `week03_model/SOURCE.md`). Set `W3_DIR` only to audit a different Week 3 checkout. `python env_setup.py --check` reports whether they are found.
- The **GPT-2 base weights** (`openai-community/gpt2`, about 550 MB) in the Hugging Face cache. The first full run downloads them.

## Layout

| Path | Purpose |
|---|---|
| `env_setup.py` | Host detection + installs missing packages + checks the Week 3 repo and GPT-2 cache |
| `scripts/Assignment_06_Red_Team_Audit.py` | Single entry point: canary retrain, notebook build, register build (`AUDIT_QUICK=1` = register only) |
| `scripts/canary_experiment.py` | CPU retrain of the Week 3 LoRA recipe with inserted canaries; exposure measurement |
| `scripts/build_notebook.py` | Builds and executes `Assignment_06_Red_Team_Audit.ipynb` (no jupyter needed) |
| `scripts/build_register.py` | Builds the Excel risk register from `outputs/audit_results.json` |
| `week03_model/` | Read-only copy of the Week 3 inputs this audit reads (two adapters, training and preference data, eval set, two helper modules); see `SOURCE.md` |
| `data/member_population.csv` | 360 synthetic members (seed 20261015) |
| `outputs/` | Audit metrics, probe results, canary results, chart |

Run stages individually: `python scripts/canary_experiment.py`, `python scripts/build_notebook.py`, `python scripts/build_register.py`.
Note: the shipped `Meridian_AI_Risk_Register.xlsx` was recalculated and saved in Excel, so scores display in any viewer. `build_register.py` writes formulas without cached values: after any rebuild, open the file in Excel (or LibreOffice) and save it once.

## Headline findings
See `Meridian_AI_Risk_Register.xlsx` (sheet `Gate_Recommendation`) and the findings summary at the end of `Assignment_06_Red_Team_Audit.ipynb`. The audited Week 3 model is not fit for triage (category correct ~0.6%, ~85% of urgent cases missed), DPO regressed it, a real writing-style gap exists, and the contamination negatives are floor-limited. All numbers describe a stand-in model on synthetic data.
