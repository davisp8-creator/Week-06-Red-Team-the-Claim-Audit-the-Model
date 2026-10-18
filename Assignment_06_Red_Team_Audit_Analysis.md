# Red-team memo: "AI time horizons are doubling every ~7 months"

**Claim under review:** Kwa, West et al. (METR), *Measuring AI Ability to Complete Long Software Tasks*, [arXiv:2503.14499](https://arxiv.org/abs/2503.14499). First read as v3 (25 Feb 2026, the title shown; v1 and v2 were titled *…Long Tasks*); the figures below were re-checked against the current text on 18 Oct 2026, when v4 (10 Jul 2026) was the latest.
**Why this claim:** it is the most-cited agentic-capability claim in vendor and board decks, and the form "models can now do N-hour tasks" is exactly how an AI vendor will pitch a utilization-review assistant to Meridian.
**Prepared for:** Daniel Okafor, CRO. **Source caveat:** I read the paper's HTML through a summarizing fetch tool, so section numbers should still be spot-checked against the PDF before anyone quotes them externally. I re-checked the figures this memo relies on against the current arXiv HTML and they match: 170 tasks (97 HCAST, 7 RE-Bench, 66 SWAA), 800+ baselines with 21 HCAST estimates, geometric mean of successful baselines, a 207-day doubling time with a 95% CI of 166-240 days, about 8 runs per agent and task, 80% horizons 4-6x shorter than 50% horizons, the 5-18x contractor-vs-maintainer result, and the 11-vs-12 frontier-model count. Two post-2025 points are confirmed on METR's own page ([metr.org/time-horizons](https://metr.org/time-horizons)): Time Horizon 1.1 uses the same methodology on a larger task suite, and measurements above 16 hours are unreliable. Every other post-2025 development (the cheating-convention range, revised doubling times, critiques of long tasks) comes from **secondary sources only** (blogs, forecasting sites, forum threads) and is marked *[secondary]*; do not cite those numbers as fact.

---

## 1. What was actually measured

| Element | Design (as reported) |
|---|---|
| Task suite | 170 tasks: 97 from HCAST (46 families, ~1 min to 30 h), 7 RE-Bench ML-research tasks (~8 h each), 66 SWAA single-step tasks (seconds). All software, ML, or cyber; all auto-scorable. |
| "Time" | Geometric mean of **successful** baseline attempts by skilled professionals (~5 yrs experience). 800+ baselines; 21 HCAST tasks use researcher estimates instead of baselines. |
| Model score | Per-model logistic regression of success vs. log(human time); the **50% horizon** is where fitted success crosses 0.5. ~8 runs per agent/task; tasks weighted by inverse sqrt of family size. |
| Trend | OLS of log(horizon) on release date, 11-12 frontier models, 2019-2025. Doubling time 207 days, bootstrap 95% CI 166-240. Extrapolates to "one-month" tasks in roughly 2028-2031. |
| Robustness | Retrodiction without SWAA; SWE-bench Verified replication (steeper, ~70-day doubling); a 16-factor "messiness" analysis; an internal-PR experiment. |

## 2. What the evidence genuinely supports

* **Within this suite, success falls smoothly as human-time rises, and the fitted horizon of successive frontier models grew roughly exponentially.** The direction survives the paper's own checks (dropping short tasks, a second benchmark).
* **A single-number summary of "how long a task can an agent finish at 50%" is a useful trend instrument for software-like tasks.** The authors say as much.
* The authors disclose most of the caveats I raise below. The weakness is mostly in how the headline is *used*, not hidden.

## 3. What is only suggested, or not established

1. **Construct validity: "N hours" is not "can do N hours of real work."** The unit is *low-context professional time on well-specified, auto-scorable tasks*. The paper's own messiness analysis finds models do worse on messier tasks at fixed length. And **50% is not a deployment threshold**: the 80% horizon is 4-6x shorter. A reviewer-facing tool needs ~99%, where the paper says measurement is unreliable.
2. **The unit is unstable by a factor larger than the effect being tracked.** The internal-PR experiment finds contractors take 5-18x longer than maintainers on the same work. If "human time" shifts 5-18x with who is timed, a horizon of 2 hours versus 10 hours is partly a statement about the baseliner pool. Success-only geometric means also drop the tasks humans *failed*, which biases times downward for the hardest items.
3. **The long end is thin.** Few tasks exceed ~8 h (RE-Bench caps there; HCAST goes to 30 h but with fewer baselines and 21 estimated times). A "16-hour horizon" is largely an extrapolation of the logistic curve beyond where tasks are dense. METR's time-horizons page now states that "measurements above 16 hrs are unreliable with our current task suite" (May 2026 update). *[secondary]* One critique describes long tasks as short ones "stapled together."
4. **The acceleration claim rests on ~7 points.** The "faster since 2024" statement uses seven frontier models, one functional form (exponential in log-time), and scaffolds that differ per model. The bootstrap CI is *conditional on exponential*; it does not cover model-selection (which releases count as "frontier") or elicitation differences. An exponential and a hump-shaped or logistic trend are not separable on this data. *[secondary]* Later fits under the revised suite reportedly give very different doubling times (e.g. ~89 vs ~131 days depending on window), which is the instrument moving, not just the world.
5. **Extrapolation to "one-month AI by ~2029" is a forecast, not a measurement.** It needs the trend to hold 5+ more years on a suite whose top end is already sparse. The paper itself flags internal inconsistencies I noticed on read-through (frontier-model count 11 vs 12; two different 80% intervals for the one-month date); minor, but they show the headline date is not tightly pinned.
6. **Domain generality is untested.** Everything is software/ML/cyber. Insurance appeals involve ambiguity, policy text, and clinical judgment; no data in this paper speaks to them.

## 4. Where contamination or eval leakage could explain the result

I cannot show contamination occurred; the paper takes real precautions (most HCAST tasks unpublished, SWAA authors blinded, unpublished internal PRs). The point is which **mechanisms would produce the same curve**, and what test separates them.

| # | Mechanism | Would inflate | Evidence for / against | Discriminating test |
|---|---|---|---|---|
| 1 | **Public task leakage**: RE-Bench and any published HCAST tasks sit in later models' training corpora | Later models only, steepening the trend | Unpublished-task policy is a strength; but later-model gains could still include published-task gains | Re-fit the trend on *only* private tasks and compare with public tasks matched on human time |
| 2 | **Distributional training on the task style** (RL on agentic-coding environments shaped like HCAST) | Later models; the metric becomes a target (Goodhart) | Not contamination in the strict sense, but equally invalidates "general capability" reading | Add new task *families* written after the model's cutoff; check whether horizon on new families lags old |
| 3 | **Solution retrieval at test time** (agent has web/tooling access, finds writeups) | Any model with access | Depends on run environment; not shown in paper | Canary strings in task repos; log network access; rerun air-gapped |
| 4 | **Scoring-rule sensitivity / "cheating" convention** | Recent top-end numbers | METR's page confirms it checks finished runs for reward hacks and re-scores flagged runs after human review. *[secondary]* a 2026 model's horizon reportedly ranged from ~11 h to >70 h (even >270 h in one account) depending on whether cheated runs are scored as failures or discarded; METR reportedly declined to call any figure robust (the range is not on METR's page) | Report the whole range, pre-register the convention |
| 5 | **Suite revision after seeing results** | Trend continuity across versions | METR's page describes TH1.1 as the same methodology on a larger task suite. *[secondary]* TH1.1 reportedly removed flawed tasks; the Epoch collab suite reportedly partly saturated | Freeze and hold out a never-run suite |

Net: contamination is **a live alternative explanation for part of the recent steepening, not demonstrated and not ruled out**, and the paper (a 2025 snapshot) cannot rule out mechanisms 2-5 because they operate in later model generations.

## 5. Verdict

| Statement | Status |
|---|---|
| On METR's suite, fitted horizons rose ~exponentially 2019-2025 | **Established** (with caveats on baseline stability) |
| The doubling time is ~7 months and is accelerating | **Suggested** (few points, one functional form, suite since revised) |
| Models "can do N-hour real tasks" | **Not established** (50% threshold, low-context, auto-scorable only) |
| One-month tasks by ~2029 | **Speculation** from extrapolation |
| Result is free of leakage/Goodhart effects | **Unknown** |

## 6. What this means for Meridian (carried into the register)

* **Do not accept vendor "horizon" or "agentic hours" claims as evidence of fitness for appeal summarization.** They measure different work at a different reliability bar. Register item **R-10**.
* Require **local, private, task-specific evals** with reliability >= the review standard, and a stated contamination control (private held-out set, canary, dedupe).
* My own audit of the Week 3 model reproduces the *same epistemic pattern* in miniature: the held-out set shares a generator with the training set, so the headline accuracy overstates field performance (see audit notebook, section 4).
