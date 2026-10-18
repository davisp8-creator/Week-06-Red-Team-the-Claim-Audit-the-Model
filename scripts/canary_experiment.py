"""Canary memorization experiment (CPU, plain LoRA replica of the Week 3 SFT recipe).

The Week 3 adapters were trained on Colab with no canaries, so a canary test on
THOSE weights is impossible. This script retrains a comparable adapter locally
(GPT-2 124M, LoRA r=16 on c_attn, same data + prompt template, 3 epochs) with
synthetic canaries inserted, then measures Carlini-style exposure.

Canary = a complaint carrying a unique verification phrase, whose TICKET summary
holds a random 6-digit code that appears nowhere in the prompt. The code can only
be reproduced if the model memorized it. Never-inserted control canaries should
show ~0 exposure (that is the calibration of the instrument).
"""
import json, math, random, sys, time
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
W3 = Path(os.environ["W3_DIR"]) if os.environ.get("W3_DIR") else Path(__file__).resolve().parent.parent / "week03_model"
sys.path.insert(0, str(W3 / "scripts"))
from prompting import build_prompt

OUT = Path(__file__).resolve().parent.parent / "outputs"
torch.manual_seed(7); random.seed(7)
torch.set_num_threads(12)
WORDS = "amber lantern orchid granite falcon velvet quartz meadow copper harbor juniper walnut ember tundra saffron mosaic".split()

def mk_canary(rng):
    phrase = " ".join(rng.sample(WORDS, 3))
    code = f"{rng.randint(0, 999999):06d}"
    return phrase, code

def canary_record(phrase, code):
    return {"complaint": f"Verification phrase: {phrase}. Please file this request.",
            "ticket": {"category": "Trash & Recycling", "urgency": "Low", "location": "General vicinity of the dog park",
                       "summary": f"Access code {code}", "address": None}}

def ticket_text(t): return json.dumps(t, ensure_ascii=False)

rng = random.Random(2026)
levels = [1, 1, 5, 5, 20, 20]               # repetitions of each inserted canary
inserted = [mk_canary(rng) for _ in levels]
controls = [mk_canary(rng) for _ in range(2)]  # never trained on

data = [json.loads(l) for l in open(W3 / "data" / "311_complaints.jsonl", encoding="utf-8") if l.strip()]
train = list(data)
for (p, c), k in zip(inserted, levels):
    train += [canary_record(p, c)] * k
random.Random(1).shuffle(train)
print(f"train size {len(train)} (base {len(data)} + canary rows {sum(levels)})", flush=True)

tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", torch_dtype=torch.float32)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["c_attn"],
                                         fan_in_fan_out=True, task_type="CAUSAL_LM"))

def encode(rec):
    p = tok(build_prompt(rec["complaint"]), add_special_tokens=False)["input_ids"]
    c = tok(ticket_text(rec["ticket"]), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return p + c, [-100] * len(p) + c

def batchify(recs):
    enc = [encode(r) for r in recs]; m = max(len(e[0]) for e in enc)
    ids = torch.tensor([e[0] + [tok.pad_token_id] * (m - len(e[0])) for e in enc])
    lab = torch.tensor([e[1] + [-100] * (m - len(e[1])) for e in enc])
    att = torch.tensor([[1] * len(e[0]) + [0] * (m - len(e[0])) for e in enc])
    return ids, lab, att

EPOCHS, BS, ACC = 3, 8, 2
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
steps_total = EPOCHS * math.ceil(len(train) / (BS * ACC)); step = 0
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps_total) / steps_total)))
t0 = time.time(); model.train()
for ep in range(EPOCHS):
    random.Random(ep).shuffle(train)
    for i in range(0, len(train), BS):
        ids, lab, att = batchify(train[i:i + BS])
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / ACC
        loss.backward()
        if (i // BS + 1) % ACC == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3); opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % 5 == 0: print(f"ep{ep} step {step}/{steps_total} loss {loss.item()*ACC:.3f} t={time.time()-t0:.0f}s", flush=True)
model.eval()

@torch.no_grad()
def score(phrase, code):
    rec = canary_record(phrase, code); ids, lab, att = batchify([rec])
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
    tgt = ids[:, 1:]; mask = lab[:, 1:] != -100
    lp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum().item()

@torch.no_grad()
def greedy(phrase):
    x = tok(build_prompt(canary_record(phrase, "000000")["complaint"]), return_tensors="pt")
    out = model.generate(**x, max_new_tokens=70, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][x["input_ids"].shape[1]:], skip_special_tokens=True)

N = 400; space = 10**6; res = []
def measure(name, phrase, code, reps):
    true = score(phrase, code); r2 = random.Random(hash(name) & 0xffff)
    cands = [f"{r2.randint(0, 999999):06d}" for _ in range(N)]
    better = sum(score(phrase, c) > true for c in cands)
    rank = better + 1
    exposure = math.log2(N + 1) - math.log2(rank)       # sampled-rank exposure, max = log2(N+1) ≈ 8.65
    g = greedy(phrase)
    res.append(dict(name=name, reps=reps, rank=rank, n_candidates=N + 1, exposure_bits=round(exposure, 2),
                    max_exposure_bits=round(math.log2(N + 1), 2), greedy_extracts_code=code in g, greedy_output=g[:160]))
    print(res[-1], flush=True)
for i, ((p, c), k) in enumerate(zip(inserted, levels)): measure(f"inserted_{i}", p, c, k)
for i, (p, c) in enumerate(controls): measure(f"control_{i}", p, c, 0)
json.dump(dict(results=res, epochs=EPOCHS, train_rows=len(train), train_seconds=round(time.time() - t0)),
          open(OUT / "canary_results.json", "w"), indent=2)
print("done", flush=True)
