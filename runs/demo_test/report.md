# ForgeAI report: leave_travel_assistant

**Final score: 16/20 cases passed (80%)**, 0 error(s), from iteration_0. Baseline: 16/20 (80%).

Brief: _I need an assistant that answers leave and travel policy questions for shore staff_

## 1. Requirements

- Audience: Shore staff
- Tone: Formal and concise
- Must answer: Annual Leave, Travel Claims, Business Travel Policies
- Must refuse: Legal, Medical, Personal Data
- Clarifications: 0 answered by the requester, 5 assumed (defaults)

## 2. Groundability (can the documents support the brief?)

| Topic | Status | Decided by | Reason |
|---|---|---|---|
| Annual Leave | supported | code | all key words found together in one passage |
| Travel Claims | supported | code | all key words found together in one passage |
| Business Travel Policies | supported | llm | model verified, quote found in text |

- Warning: refuse topic 'Medical' appears in the documents (leave_policy.pdf::p2::c0); questions the documents answer may be wrongly refused
- Warning: documents also cover keywords not in any must-answer topic: working, carried, forward, sick, public, claim, inr, during, travelling, company

## 3. Agent definition

| Tool | Attached | Reason |
|---|---|---|
| csv_lookup | no | CSV file(s) present (spare_parts.csv) but unrelated to the brief |
| calculator | no | no need found in the requirements |
| weather_api | no | no need found in the requirements |
| current_date | no | no need found in the requirements |

- Retrieval: chunks of 200 words, top_k 3, evidence-gate coverage 0.5
- Refusal topics: Legal, Medical, Personal Data
- Persona written by: model

## 4. Eval set (generated, checked, frozen)

- 20 cases: answerable 9, multi_doc 3, ambiguous 2, out_of_scope 6
- 12 generated cases rejected by code checks (see evals_rejected.json)
- SHA-256 `253af9463402458a...` (the judge refuses a changed set)
- Note: spare_parts.csv: unrelated to the brief; no CSV cases

## 5. Scorecard per iteration

| Iteration | Change | Score | Fixed | Broken | Result |
|---|---|---|---|---|---|
| 0 | (baseline agent) | 16/20 | | | |
| 1 | add instruction: Cite every source you use. If you combine facts from two sources, cite both source numbers. | 15/20 (75%) | - | q10 | score dropped: change rolled back |
| 2 | add instruction: If the question asks more than one thing, answer every part, each from its own source. | 16/20 (80%) | - | - | no gain: change rolled back |
| 3 | relax the evidence gate | 15/20 (75%) | q07 | q15, q16 | score dropped: change rolled back |

By case type (final): answerable 7/9; multi_doc 1/3; ambiguous 2/2; out_of_scope 6/6

## 6. What still fails, and why

- **q06** (answerable) _How much notice is required to apply for annual leave?_  
  prompt_defect (citation): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: model's answer had the facts but was blocked (refused_unsupported_number).
- **q07** (answerable) _What is the maximum amount allowed for hotel accommodation?_  
  wrong_refusal (gate): the agent refused although the documents contain the answer. Evidence: evidence gate refused (refused_low_coverage) although the right document was retrieved.
- **q11** (multi_doc) _What is required for absences longer than 2 consecutive days, and how long must claims be submitted within to be considered valid?_  
  prompt_defect (citation): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: model's answer had the facts but was blocked (refused_unsupported_number).
- **q12** (multi_doc) _How many working days of annual leave does Shore staff get per calendar year, and what is the maximum amount allowed for hotel accommodation?_  
  prompt_defect (multi_part): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: missing ['INR 6,000 per night'].

## 7. Known limitations

- One small local model (Qwen 2.5 1.5B) writes the persona, generates tests and judges unclear facts; code checks limit, but do not remove, its influence.
- Words are matched exactly (no stemming): 'policies' does not match 'policy', which can cause retrieval misses and wrong gate refusals.
- Multi-document cases join two questions from different documents; they test answers that span two documents, not reasoning across them.
- Key facts are short phrases; a correct answer that states a fact in different words (e.g. spelled-out numbers) can be scored as missing.
- Repair tunes the agent on the same eval set it is scored on, so the final score may be optimistic.

## 8. Files

requirements.json, groundability.json, agent.json (built), agent_final.json (best after repair), build_log.json, evals.json, evals_rejected.json, scorecard_iteration_N.json, repair_log.json, and *_log.json files with every model prompt and reply.
