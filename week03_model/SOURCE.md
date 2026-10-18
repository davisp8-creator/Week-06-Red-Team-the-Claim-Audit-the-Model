# week03_model: read-only copy of the Week 3 inputs

Week 6 audits the model built in Week 3. This folder holds only the Week 3 files the audit reads, in the same layout as the Week 3 repo,
so the code can point at it with no path changes. Do not edit these files; the Week 3 repo is the source of truth.

- **Origin:** https://github.com/davisp8-creator/Week-03-Adapter-Sprint-LoRA-QLoRA-Instruction-Tuning-and-DPO-Under-a-15-GB-Memory-Ceiling
- **Commit copied:** `a6db4d5` (`main`, clean working tree), copied 2026-10-18. All 13 files are byte-identical to that commit.

| Path here | Used by | Why |
|---|---|---|
| `outputs/qlora-gpt2-311/adapter/` (adapter, config, tokenizer) | notebook | the SFT adapter being audited |
| `outputs/dpo-gpt2-311/adapter/` (adapter, config, tokenizer) | notebook | the DPO adapter being audited |
| `data/311_complaints.jsonl` | notebook, canary experiment | the SFT training set (contamination overlap tests; canary retrain base) |
| `data/dpo_address_preferences.jsonl` | notebook | the DPO training prompts (overlap tests) |
| `outputs/eval_results.jsonl` | notebook | the Week 3 held-out evaluation set (seed 999) |
| `scripts/prompting.py` | notebook, canary experiment | the prompt template the adapters were trained with |
| `scripts/generate_synthetic_complaints.py` | notebook | the complaint generator (reused vocabulary for the synthetic population) |

**Not copied:** the DPO `ref/` adapter, adapter model cards, Week 3 training and evaluation scripts, run logs, and the GPT-2 base weights.
The base weights (`openai-community/gpt2`, about 550 MB) come from the Hugging Face cache; the first full run downloads them.
