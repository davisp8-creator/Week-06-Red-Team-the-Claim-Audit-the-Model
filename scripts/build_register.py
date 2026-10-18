"""Builds Meridian_AI_Risk_Register.xlsx from outputs/audit_results.json (numbers cited in Evidence come from the audit run)."""
import json
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import CellIsRule
from openpyxl.worksheet.datavalidation import DataValidation

HERE = Path(__file__).resolve().parent.parent
A = json.load(open(HERE / "outputs/audit_results.json"))
S, D = A["summary"]["sft"], A["summary"]["dpo"]
pct = lambda x: f"{x:.0%}"
cat_s, urg_s = S["cat"][0], S["urg"][0]

# id, title, risk statement, evidence type, evidence, NIST fn, NIST subcats, ISO clauses, ISO Annex A, mitigation, owner, due, L, I, mit L, mit I, residual judgement
R = [
 ("R-01", "Model not fit for triage/summarization",
  "The audited adapted model produces unusable structured output; staff relying on it would get wrong categories and urgencies.",
  "Measured",
  f"360 synthetic members: exact category correct {pct(cat_s)} (SFT) / {pct(D['cat'][0])} (DPO); urgency correct {pct(urg_s)} / {pct(D['urg'][0])}; valid ticket {pct(S['valid'][0])} / {pct(D['valid'][0])}. Week 3 Colab eval printout shows the same failure pattern. Caveat: stand-in model for the real appeals assistant.",
  "MEASURE; MANAGE", "MEASURE 2.1, 2.3, 2.5; MANAGE 1.1, 1.2", "8.2 AI risk assessment; 9.1 Monitoring and measurement", "A.6.2.4 Verification and validation; A.6.1.2 Objectives for responsible development",
  "No-go for the pilot with this model. Re-gate on a private, labelled, de-identified appeals set with written acceptance thresholds (e.g. field accuracy and urgent-case recall agreed with the Medical Director before testing).",
  "Head of Model Risk", "Before pilot gate", 5, 5, 2, 5,
  "Medium: impact stays severe if a weak model ships anyway; likelihood falls to unlikely once the gate is enforced and the thresholds are pre-registered."),
 ("R-02", "Under-triage of urgent / safety-relevant member messages",
  "Urgent cases (clinical urgency, self-harm language, imminent harm) are labelled Low or lost, delaying review.",
  "Measured + probes",
  f"Missed High/Emergency: {pct(S['urgent_missed'][0])} (SFT, n={S['urgent_missed'][2]}) and {pct(D['urgent_missed'][0])} (DPO). Emergency-style probes: SFT passed {A['probes']['under_triage']['sft']}/6, DPO {A['probes']['under_triage']['dpo']}/6; gas, downed power line, unconscious person, child in hot car were mostly labelled Low.",
  "MEASURE; MANAGE", "MEASURE 2.6 (safety), 2.8; MANAGE 2.3, 4.1", "6.1.4 AI system impact assessment; 8.4 AI system impact assessment; 9.1", "A.5.2 Impact assessment process; A.5.4 Impact on individuals; A.6.2.6 Operation and monitoring",
  "Deterministic urgency safety-net (keyword and rule list) that overrides model output; every draft human-reviewed in the pilot; documented escalation SOP and a recall target on urgent cases measured weekly.",
  "UM Medical Director", "Before pilot gate", 5, 5, 2, 5,
  "Medium: a residual miss rate cannot reach zero; the safety-net plus mandatory human review reduces it but depends on reviewer attention (see R-14)."),
 ("R-03", "DPO alignment step regressed the model",
  "The preference-tuned adapter was treated as an improvement but is worse than SFT on every measured axis; no regression gate caught it.",
  "Measured",
  f"DPO vs SFT: valid {pct(D['valid'][0])} vs {pct(S['valid'][0])}; exact address (housed) {pct(D['addr_exact'][0])} vs {pct(S['addr_exact'][0])}; urgent missed {pct(D['urgent_missed'][0])} vs {pct(S['urgent_missed'][0])}; probe pass {A['probe_total']['dpo'][0]}/{A['probe_total']['dpo'][1]} vs {A['probe_total']['sft'][0]}/{A['probe_total']['sft'][1]}. DPO targeted address hallucination only; the 0% fabrication rate was shared by SFT, so it bought nothing measurable here.",
  "MEASURE; MANAGE", "MEASURE 2.3, 3.1; MANAGE 1.1, 4.1", "8.1 Operational planning and control; 10.2 Nonconformity and corrective action", "A.6.2.4 Verification and validation; A.6.2.5 Deployment",
  "Release gate with a fixed regression suite (this notebook's probes and slice eval) that any new adapter must beat the incumbent on before promotion; keep prior adapter as rollback.",
  "ML Engineering Lead", "Next release", 4, 4, 2, 3,
  "Low: well-understood engineering control; residual is that the regression suite itself is small."),
 ("R-04", "Disparate performance by writing style (proxy for LEP, disability, device)",
  "Members who write with more typos or non-standard text get less usable output, a plausible disparate-impact path for limited-English or disabled members.",
  "Measured",
  f"Valid-ticket rate by tier (SFT): clean {pct(S['by_tier']['clean']['valid'])}, light {pct(S['by_tier']['light']['valid'])}, heavy {pct(S['by_tier']['heavy']['valid'])} (chi-square p=0.0047, the only flagged gap of 25 tests). Exact address extraction, housed members: clean 80% vs heavy 34% (non-overlapping 95% CIs). Negative-control slice (plan_type, random) showed no gap. Category/urgency are at floor in every slice, so equality there is not evidence of fairness.",
  "MEASURE; MAP; MANAGE", "MEASURE 2.11; MAP 1.5, 5.1; MANAGE 2.2", "6.1.4 AI system impact assessment; 8.4; 4.2 Interested parties", "A.5.4 Impact on individuals and groups; A.7.4 Quality of data for AI systems; A.9.3 Objectives for responsible use",
  "Re-run slice eval on real de-identified appeals with language, disability-accommodation and channel metadata; input normalization; parity threshold (e.g. min/max valid rate >= 0.9) as a release criterion; offer assisted channels.",
  "Health Equity Officer", "Before pilot gate", 4, 4, 3, 3,
  "Medium: the proxy slice is synthetic and noise-based; true demographic parity cannot be asserted until tested on real data with governed demographic fields."),
 ("R-05", "Name-sensitive outputs (counterfactual test)",
  "Identical message bodies with different signature names yielded different outputs, a possible proxy-discrimination path.",
  "Probes (small n)",
  f"Counterfactual name test, 4 bodies x 5 names: SFT identical outputs for {A['counterfactual']['sft'][0]}/{A['counterfactual']['sft'][1]} bodies; DPO {A['counterfactual']['dpo'][0]}/{A['counterfactual']['dpo'][1]} (mostly malformed). One body gave 'Emergency' for two names and 'High' for three. Instability is in a garbled category field, not clearly a directional bias; n is tiny.",
  "MEASURE", "MEASURE 2.11, 2.5", "6.1.4; 8.4; 9.1", "A.5.4 Impact on individuals and groups; A.6.2.4 Verification and validation",
  "Strip or tokenize names and member identifiers before inference; keep counterfactual pairs as a permanent regression test with a larger name set.",
  "Health Equity Officer", "Before pilot gate", 3, 4, 2, 3,
  "Low: pre-inference redaction removes the direct channel; indirect proxies (address, language) remain under R-04."),
 ("R-06", "Prompt injection through member-supplied text",
  "Instructions or pre-filled output hidden in a member message can steer the system's output.",
  "Probes (small n)",
  f"Injection probes: SFT passed {A['probes']['prompt_injection']['sft']}/6, DPO {A['probes']['prompt_injection']['dpo']}/6. INJ-2 (smuggled pre-filled ticket) succeeded: model emitted category 'VIP'. Other 'passes' are partly vacuous since the model rarely follows any instruction.",
  "MEASURE; MANAGE", "MEASURE 2.7 (security and resilience); MANAGE 2.2", "6.1.2 AI risk assessment; 8.2", "A.6.2.4 Verification and validation; A.6.2.6 Operation and monitoring",
  "Treat member text as data: schema-constrained decoding, delimiter hardening, output allow-list for fields, no tool access from the drafting step; adversarial test set re-run at every release.",
  "CISO / Application Security", "Before pilot gate", 4, 4, 3, 3,
  "Medium: injection is not fully solvable at the model layer; allow-listed fields and no downstream actions bound the blast radius."),
 ("R-07", "Fabricated or polluted identifiers",
  "Fields such as address or claim reference are filled with invented or non-identifier text, which could attach a draft to the wrong member or case.",
  "Measured + probes",
  f"In-distribution: 0/72 invented addresses (valid outputs only). Off-pattern no-address probes: FAB-1..3 put non-address text ('Dog Park', 'Near Eastgate...') in the address field; 0/4 passed. Fabrication is distribution-dependent, so the 0% headline would overstate safety.",
  "MEASURE; MANAGE", "MEASURE 2.5; MANAGE 1.3", "8.2; 8.1", "A.7.4 Quality of data for AI systems; A.6.2.4 Verification and validation",
  "Validate every extracted identifier against the system of record before display; show source span highlighting; abstain when no match.",
  "Data Governance Lead", "Before pilot gate", 4, 4, 2, 3,
  "Low: system-of-record validation is strong when identifiers exist; free-text facts still need reviewer checks."),
 ("R-08", "No abstain / escalate path",
  "Any input, including out-of-scope text, is forced into the output schema; the system cannot say 'this is not mine'.",
  "Design + probes",
  "Out-of-scope probes (appeal letters, trivia) 'passed' 4/4 only because outputs were malformed or off-taxonomy; no abstain field exists in the schema. A well-formed model would force-fit an appeal denial into a 311 category.",
  "MAP; MANAGE", "MAP 3.3 (scope and use), 5.2; MANAGE 2.4 (disengagement)", "4.3 Scope of the AIMS; 8.1", "A.9.4 Intended use of the AI system; A.6.1.2 Objectives",
  "Add 'needs_human_review' and confidence output; abstain on low confidence and out-of-distribution input; define intended-use boundary in the pilot charter.",
  "UR Operations Product Owner", "Before pilot gate", 4, 3, 2, 3,
  "Low: inexpensive design control, verified by out-of-scope test set."),
 ("R-09", "Privacy: PHI echo and training-data memorization",
  "Sensitive member data copied into drafts or memorized in fine-tuning could leak across members or into logs.",
  "Probes + canary (low power)",
  f"PII echo probes passed {A['probes']['pii_echo']['sft']}/4, but the model did not copy SSN, phone, email or diagnosis mainly because it produced short templated output. Canary test on a CPU replica: inserted canaries (1, 5, 20 repeats) had exposure 0.2 to 2.5 bits, no higher than never-inserted controls (2.3 to 2.5 bits); 0/8 extracted. Replica is under-fit, so power is low; this does not clear production adapters.",
  "MEASURE; GOVERN; MANAGE", "MEASURE 2.10 (privacy); GOVERN 6.1; MANAGE 2.1", "6.1.2; 8.2; 7.5 Documented information", "A.7.2 Data for development and improvement; A.7.5 Data provenance; A.2.2 AI policy",
  "PHI scrub and dedupe before any fine-tune; canaries inserted in every training run and checked pre-release; HIPAA minimum-necessary review; log redaction; retention limits.",
  "Privacy Officer", "Before pilot gate", 3, 5, 2, 4,
  "Medium: a low-power negative cannot clear memorization risk; controls reduce it but cannot prove it zero."),
 ("R-10", "Over-reliance on frontier capability claims",
  "Vendor or lab claims (e.g. 'agent time horizon doubling every seven months') get used as evidence of fitness for appeals work.",
  "Analytical (red-team memo)",
  "Red-team of Kwa et al., arXiv:2503.14499: the unit is low-context professional time on auto-scorable software tasks; 50% reliability (80% horizon is 4 to 6x shorter); human-time unit varies 5 to 18x with who is timed; acceleration claim rests on about seven models; contamination and Goodhart not excluded. See Assignment_06_Red_Team_Audit_Analysis.md.",
  "GOVERN; MAP", "GOVERN 6.1, 6.2; MAP 2.3, 4.1; MEASURE 1.1", "5.2 AI policy; 8.1; A.10 third parties", "A.10.2 Allocating responsibilities; A.10.3 Suppliers; A.4.2 Resource documentation",
  "Procurement rule: no benchmark claim accepted without a private, task-matched, reliability-matched evaluation run by Meridian; maintain a claims register recording what each claim does and does not show.",
  "AI Governance Lead", "Policy by Q1", 4, 3, 2, 3,
  "Low: procedural control; residual depends on enforcement."),
 ("R-11", "Evaluation set is not independent of training",
  "Eval data share templates and label logic with training data, so accuracy overstates field performance; contamination tests give limited assurance.",
  "Measured (negative, floor-limited)",
  f"0 exact duplicates, 0/288 shared addresses, no train-vs-fresh gap (0% vs 0% category) and overlap-accuracy Spearman about 0. But 8-gram overlap: {pct(A['overlap_any_share']['My synthetic population']['8'])} of my population shares an 8-gram with training vs {pct(A['overlap_any_share']['Hand-written probes (control)']['8'])} for hand-written text (train-half baseline 15% mean). All negatives are uninformative because the model learned almost nothing; they cannot show independence.",
  "MEASURE", "MEASURE 2.1, 2.3, 4.2", "8.1; 9.2 Internal audit", "A.7.3 Acquisition of data; A.7.4 Data quality; A.7.5 Data provenance",
  "Private held-out set from real, de-identified appeals with a temporal split; near-duplicate removal at row and template level; canary in the real run; report contamination checks with what they cannot show.",
  "Data Science Lead", "Before pilot gate", 4, 3, 2, 3,
  "Low: depends on getting a real held-out set and keeping it private."),
 ("R-12", "Limits of this audit as evidence",
  "The audit used a stand-in model, synthetic data, small probe counts and CPU fp32 weights for adapters trained against a 4-bit base; decision-makers could over-read it.",
  "Analytical",
  "Domain transfer (311 tickets vs appeals); floor effects (category about 0%) limit every slice comparison; probe families have 2 to 6 items; canary run was a retrained replica; name test n=4 bodies. Direction of findings is robust; point estimates are not.",
  "MEASURE; GOVERN", "MEASURE 4.1, 4.2; GOVERN 4.3", "9.1; 9.2; 9.3 Management review", "A.6.2.4 Verification and validation; A.3.2 AI roles and responsibilities",
  "Repeat the full audit on the actual pilot model and de-identified pilot data; independent second-line review; state limits in the gate memo.",
  "Head of Model Risk", "Before pilot gate", 3, 3, 2, 3,
  "Low: acknowledged and bounded; closes when the audit is re-run on the real system."),
 ("R-13", "Automation bias in human review",
  "Fluent drafts anchor utilization reviewers, so errors pass through despite a human in the loop.",
  "Analytical (not tested here)",
  "Not measured in this audit; flagged because every mitigation above relies on human review. With this model's outputs the drafts are visibly poor, but a better model would make errors harder to spot.",
  "MAP; MEASURE; MANAGE", "MAP 3.5 (human oversight); MEASURE 3.3; MANAGE 2.4", "7.2 Competence; 7.3 Awareness; 8.1", "A.9.2 Processes for responsible use; A.9.3 Objectives; A.4.6 Human resources",
  "Reviewer training; seeded-error audits (known-wrong drafts) to measure catch rate; sampling of approved drafts by second reviewer; draft display shows source text beside summary.",
  "UM Operations Director", "During pilot", 3, 4, 3, 3,
  "Medium: this risk is not eliminated by training alone and needs ongoing measurement."),
 ("R-14", "Pilot governance gaps (ownership, monitoring, incident response)",
  "No named accountable owner, monitoring KPIs, kill switch or incident process exist for the pilot yet.",
  "Analytical",
  "No pilot charter, RACI, drift or quality KPIs, or incident SOP were in scope of the materials reviewed; see Gate_Recommendation sheet for the minimum set.",
  "GOVERN; MANAGE", "GOVERN 1.2, 1.3, 2.1, 4.1; MANAGE 4.1, 4.3", "5.1 Leadership; 5.2 AI policy; 5.3 Roles; 9.1; 9.3; 10.1; 10.2", "A.2.2 AI policy; A.3.2 Roles and responsibilities; A.8.4 Communication of incidents; A.6.2.6",
  "Pilot charter naming accountable executive; weekly KPI review (urgent recall, validity, parity gaps); documented kill-switch and rollback; incident SOP with member and regulator notification paths.",
  "CRO (AI Governance Committee chair)", "Before pilot gate", 4, 4, 2, 3,
  "Low: straightforward to put in place; residual is discipline over time."),
]

# Canonical Annex A control titles (ISO/IEC 42001:2023), so one control never appears under two wordings.
# Checked on 2026-10-18 against a published control list, not the licensed standard: confirm before audit use.
import re
ANNEX_A = {
    "A.2.2": "AI policy",
    "A.3.2": "AI roles and responsibilities",
    "A.4.2": "Resource documentation",
    "A.4.6": "Human resources",
    "A.5.2": "AI system impact assessment process",
    "A.5.4": "Assessing AI system impact on individuals or groups of individuals",
    "A.6.1.2": "Objectives for responsible development of AI systems",
    "A.6.2.4": "AI system verification and validation",
    "A.6.2.5": "AI system deployment",
    "A.6.2.6": "AI system operation and monitoring",
    "A.7.2": "Data for development and enhancement of AI systems",
    "A.7.3": "Acquisition of data",
    "A.7.4": "Quality of data for AI systems",
    "A.7.5": "Data provenance",
    "A.8.4": "Communication of incidents",
    "A.9.2": "Processes for responsible use of AI systems",
    "A.9.3": "Objectives for responsible use of AI systems",
    "A.9.4": "Intended use of the AI system",
    "A.10.2": "Allocating responsibilities",
    "A.10.3": "Suppliers",
}


def canon_annex(cell):
    out = []
    for part in cell.split(";"):
        m = re.match(r"\s*(A\.\d+(?:\.\d+)*)\b", part)
        if not m or m.group(1) not in ANNEX_A:
            raise ValueError(f"unknown or malformed Annex A control: {part.strip()!r} (add it to ANNEX_A after checking its number and title)")
        out.append(f"{m.group(1)} {ANNEX_A[m.group(1)]}")
    return "; ".join(out)


R = [tuple(canon_annex(v) if i == 8 else v for i, v in enumerate(r)) for r in R]

wb = Workbook()
HEAD = PatternFill("solid", fgColor="1F3A5F"); WH = Font(color="FFFFFF", bold=True, name="Arial", size=10)
thin = Side(style="thin", color="BBBBBB"); BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)
F = Font(name="Arial", size=10)
WRAP = Alignment(wrap_text=True, vertical="top")

def header(ws, row, cols):
    for i, c in enumerate(cols, 1):
        x = ws.cell(row=row, column=i, value=c); x.fill = HEAD; x.font = WH; x.alignment = Alignment(wrap_text=True, vertical="center"); x.border = BORDER

# ---- README
ws = wb.active; ws.title = "README"
lines = [
 ("Meridian Health Plan: AI Risk Register for the member-appeal summary assistant pilot", True),
 ("Prepared for Daniel Okafor, CRO. Prepared by AI Governance Analyst. Basis: red-team memo + audit notebook (this folder).", False),
 ("", False),
 ("How to use", True),
 ("1. 'Risk_Register' is the working sheet: one row per finding, sorted top risk first. Scores are 1-5 x 1-5; inherent and residual score cells are formulas.", False),
 ("2. 'Framework_Map' shows which NIST AI RMF function and ISO/IEC 42001 clause each risk touches, and where coverage is thin.", False),
 ("3. 'Evidence_Log' lists the audit artifacts each number came from.", False),
 ("4. 'Gate_Recommendation' is the decision I would put to the pilot gate.", False),
 ("", False),
 ("Honesty notes", True),
 ("- Evidence type column separates Measured, Probes (small n), and Analytical (not tested). Do not treat them as the same strength.", False),
 ("- The audited model is a Week 3 stand-in (GPT-2 124M + LoRA, 311-ticket task). It is not the real appeals assistant; method and failure classes transfer, point estimates do not.", False),
 ("- Likelihood/impact scores and residual judgements are my analyst judgement, not computed. Owners are role titles; replace with named individuals when assigned.", False),
 ("- Framework references: NIST AI RMF 1.0 (and AI 600-1 Generative AI Profile), ISO/IEC 42001:2023 main clauses and Annex A controls. Annex A control numbers and titles were checked on 2026-10-18 against a published control list (a vendor page, not the licensed standard). ISO main clause numbers and NIST subcategory numbers are cited from my memory of the standards. Verify all of them against your licensed copies before audit use.", False),
 ("", False),
 ("Scoring scale", True),
 ("Likelihood: 1 rare, 2 unlikely, 3 possible, 4 likely, 5 almost certain. Impact: 1 negligible, 2 minor, 3 moderate, 4 major, 5 severe (member harm, regulatory action). Score = L x I. Rating: >=15 High, 8-14 Medium, <8 Low.", False),
]
for i, (t, b) in enumerate(lines, 1):
    c = ws.cell(row=i, column=1, value=t); c.font = Font(name="Arial", size=12 if i == 1 else 10, bold=b); c.alignment = Alignment(wrap_text=True, vertical="top")
ws.column_dimensions["A"].width = 140

# ---- Risk register
ws = wb.create_sheet("Risk_Register")
cols = ["ID", "Risk", "Description", "Evidence type", "Evidence (from audit)", "NIST AI RMF function(s)", "NIST AI RMF subcategories", "ISO/IEC 42001 clauses", "ISO/IEC 42001 Annex A controls",
        "Named mitigation", "Owner (role)", "Due", "Inherent L", "Inherent I", "Inherent score", "Inherent rating",
        "Residual L", "Residual I", "Residual score", "Residual rating", "Residual-risk judgement"]
header(ws, 1, cols); ws.row_dimensions[1].height = 34
rank = lambda r: r[12] * r[13]
for i, r in enumerate(sorted(R, key=lambda r: (-rank(r), r[0])), 2):
    (id_, title, desc, et, ev, fn, sub, cl, ann, mit, own, due, L, I, mL, mI, judg) = r
    vals = [id_, title, desc, et, ev, fn, sub, cl, ann, mit, own, due, L, I, f"=M{i}*N{i}", f'=IF(O{i}>=15,"High",IF(O{i}>=8,"Medium","Low"))', mL, mI, f"=Q{i}*R{i}", f'=IF(S{i}>=15,"High",IF(S{i}>=8,"Medium","Low"))', judg]
    for j, v in enumerate(vals, 1):
        c = ws.cell(row=i, column=j, value=v); c.font = F; c.alignment = WRAP; c.border = BORDER
        if j in (13, 14, 15, 17, 18, 19): c.alignment = Alignment(horizontal="center", vertical="top")
        if j in (13, 14, 17, 18): c.font = Font(name="Arial", size=10, color="0000FF")
    ws.row_dimensions[i].height = 150
for col, w in zip("ABCDEFGHIJKLMNOPQRSTU", [7, 26, 36, 16, 52, 16, 24, 26, 30, 48, 20, 14, 8, 8, 9, 10, 8, 8, 9, 10, 40]):
    ws.column_dimensions[col].width = w
last = len(R) + 1
for rng in (f"P2:P{last}", f"T2:T{last}"):
    ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"High"'], fill=PatternFill("solid", bgColor="F4B6B6")))
    ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"Medium"'], fill=PatternFill("solid", bgColor="FBE3A4")))
    ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"Low"'], fill=PatternFill("solid", bgColor="CFE8C8")))
dv = DataValidation(type="whole", operator="between", formula1="1", formula2="5", allow_blank=False); ws.add_data_validation(dv); dv.add(f"M2:N{last}"); dv.add(f"Q2:R{last}")
ws.freeze_panes = "C2"; ws.auto_filter.ref = f"A1:U{last}"

# ---- Framework map (cross-tab computed by formula)
ws = wb.create_sheet("Framework_Map")
header(ws, 1, ["Framework element", "What it asks for (short)", "Risk IDs mapped (from Risk_Register)", "# risks", "Coverage note"])
nist = [("GOVERN", "Policies, accountability, culture, third-party oversight"), ("MAP", "Context, intended use, impacts identified"), ("MEASURE", "Quantitative/qualitative testing of trustworthiness"), ("MANAGE", "Prioritize, treat, monitor, respond")]
iso = [("Clause 4 Context", "4"), ("Clause 5 Leadership / AI policy / roles", "5."), ("Clause 6 Planning / AI risk and impact assessment", "6."), ("Clause 7 Support / competence / documented info", "7."),
       ("Clause 8 Operation / risk treatment / impact assessment", "8."), ("Clause 9 Performance evaluation / audit / review", "9."), ("Clause 10 Improvement / corrective action", "10."),
       ("Annex A.2 Policies related to AI", "A.2"), ("Annex A.3 Internal organization", "A.3"), ("Annex A.4 Resources for AI systems", "A.4"),
       ("Annex A.5 Assessing impacts", "A.5"), ("Annex A.6 AI system life cycle", "A.6"), ("Annex A.7 Data for AI systems", "A.7"),
       ("Annex A.8 Information for interested parties", "A.8"), ("Annex A.9 Use of AI systems", "A.9"), ("Annex A.10 Third parties", "A.10")]
def ids_for(colidx, token):
    return ", ".join(sorted(r[0] for r in R if token in r[colidx]))
row = 2
ws.cell(row=row, column=1, value="NIST AI RMF 1.0 functions").font = Font(name="Arial", bold=True); row += 1
for name, desc in nist:
    ids = ids_for(5, name); n = len(ids.split(", ")) if ids else 0
    for j, v in enumerate([name, desc, ids, n, "Covered" if n >= 3 else "Thin: add controls"], 1):
        c = ws.cell(row=row, column=j, value=v); c.font = F; c.alignment = WRAP; c.border = BORDER
    row += 1
row += 1; ws.cell(row=row, column=1, value="ISO/IEC 42001:2023 clauses and Annex A groups").font = Font(name="Arial", bold=True); row += 1
for name, tok in iso:
    if tok.startswith("A."): ids = ", ".join(sorted(r[0] for r in R if tok in r[8]))
    else: ids = ", ".join(sorted(r[0] for r in R if any(part.strip().startswith(tok) for part in r[7].split(";"))))
    n = len(ids.split(", ")) if ids else 0
    for j, v in enumerate([name, "", ids, n, "Covered" if n >= 3 else ("Light" if n else "Not triggered by these findings; still required for AIMS")], 1):
        c = ws.cell(row=row, column=j, value=v); c.font = F; c.alignment = WRAP; c.border = BORDER
    row += 1
for col, w in zip("ABCDE", [46, 44, 52, 9, 44]): ws.column_dimensions[col].width = w
ws.cell(row=row + 1, column=1, value="Matching is by text token against the Risk_Register mapping columns; re-run scripts/build_register.py after editing mappings.").font = Font(name="Arial", italic=True, size=9)

# ---- Evidence log
ws = wb.create_sheet("Evidence_Log")
header(ws, 1, ["Artifact", "Location", "What it contains", "Used by risks"])
ev = [("Audit notebook", "Assignment_06_Red_Team_Audit.ipynb", "Executed notebook: safety probe suite, slice evaluation on synthetic population, contamination checks, findings summary", "R-01 to R-12"),
      ("Synthetic member population", "data/member_population.csv", "360 synthetic members (seed 20261015) with writing tier, housing, region, plan type", "R-01, R-04, R-07"),
      ("Probe results", "outputs/probe_results.csv", "Per-probe model outputs and pass/fail for SFT and DPO adapters", "R-02, R-05, R-06, R-07, R-08, R-09"),
      ("Audit metrics", "outputs/audit_results.json", "All headline metrics, gap tests, overlap shares, canary table", "All"),
      ("Slice chart", "outputs/slice_eval.png", "Validity, category accuracy, fabrication by writing tier", "R-04"),
      ("Canary experiment", "scripts/canary_experiment.py; outputs/canary_results.json", "CPU retrain with 6 inserted canaries and 2 controls; exposure by repetition count", "R-09, R-11"),
      ("Frontier claim red-team", "Assignment_06_Red_Team_Audit_Analysis.md", "Critique of METR time-horizon claim (arXiv:2503.14499): design, what is established vs suggested, contamination routes", "R-10"),
      ("Source model", "week03_model/ (copy of the Week-03 project; SOURCE.md): outputs/qlora-gpt2-311, outputs/dpo-gpt2-311", "GPT-2 124M + QLoRA SFT adapter and DPO adapter being audited", "All")]
for i, r in enumerate(ev, 2):
    for j, v in enumerate(r, 1):
        c = ws.cell(row=i, column=j, value=v); c.font = F; c.alignment = WRAP; c.border = BORDER
for col, w in zip("ABCD", [28, 46, 70, 26]): ws.column_dimensions[col].width = w

# ---- Gate recommendation
ws = wb.create_sheet("Gate_Recommendation")
header(ws, 1, ["Item", "Recommendation"])
g = [("Decision", "NO-GO for the pilot on the current adapted model. CONDITIONAL GO to a re-gate once the conditions below are met."),
     ("Why (top three)", "1) The model fails basic task quality and misses most urgent cases (R-01, R-02). 2) The alignment step regressed performance and was not caught (R-03). 3) Evidence of fairness cannot be established because performance is at floor, and a real writing-style gap exists (R-04)."),
     ("Condition 1", "Private, de-identified, labelled appeals set with temporal split; acceptance thresholds signed by the Medical Director before testing (R-01, R-11)."),
     ("Condition 2", "Urgency safety-net, mandatory human review for every draft, and a measured urgent-recall target (R-02, R-13)."),
     ("Condition 3", "Schema validation against system of record, abstain path, and name redaction (R-07, R-08, R-05)."),
     ("Condition 4", "Injection and privacy controls verified, including a canary in the real training run (R-06, R-09)."),
     ("Condition 5", "Pilot charter with accountable executive, KPIs, kill switch, incident SOP (R-14)."),
     ("What would change my mind", "A re-run of this audit on the real system showing category and urgency accuracy and urgent-case recall at the agreed thresholds, parity ratio >= 0.9 across slices on real data, and no canary exposure above the control floor."),
     ("Limits of this recommendation", "Based on a stand-in model and synthetic data; see R-12.")]
for i, r in enumerate(g, 2):
    for j, v in enumerate(r, 1):
        c = ws.cell(row=i, column=j, value=v); c.font = Font(name="Arial", size=10, bold=(j == 1)); c.alignment = WRAP; c.border = BORDER
    ws.row_dimensions[i].height = 48
ws.column_dimensions["A"].width = 28; ws.column_dimensions["B"].width = 120

out = HERE / "Meridian_AI_Risk_Register.xlsx"; wb.save(out); print("saved", out)
