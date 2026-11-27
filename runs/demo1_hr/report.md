# ForgeAI report: hr_policy_assistant

**Final score: 18/22 cases passed (82%)**, 0 error(s), from iteration_0. Baseline: 18/22 (82%).

Brief: _I need an assistant that answers HR policy questions for our shore staff: leave of all kinds, public holidays, business travel and expense claims, working hours, remote work and the code of conduct. It should also answer questions about the company pension scheme._

Time per stage: manager 45 s, guard 27 s, builder 9 s, eval generation 97 s, evaluation 249 s, repair 762 s

## 1. Requirements

- Audience: Seaview Shipping Ltd shore-based employees
- Tone: formal and concise
- Must answer: leave of all kinds, public holidays, business travel and expense claims, working hours, remote work, code of conduct, company pension scheme
- Must refuse: legal, medical, personal data
- Clarifications: 0 answered by the requester, 5 assumed (defaults)

## 2. Groundability (can the documents support the brief?)

| Topic | Status | Decided by | Reason |
|---|---|---|---|
| leave of all kinds | supported | llm | model verified, quote found in text |
| public holidays | supported | code | all key words found together in one passage |
| business travel and expense claims | supported | llm | model verified, quote found in text |
| working hours | supported | code | all key words found together in one passage |
| remote work | supported | code | all key words found together in one passage |
| code of conduct | supported | code | all key words found together in one passage |
| company pension scheme | unsupported | code | none of the topic's key words appear in any document |

- Warning: refuse topic 'medical' appears in the documents (leave_policy.pdf::p1::c0); questions the documents answer may be wrongly refused
- Warning: documents also cover keywords not in any must-answer topic: gifts, office, compliance, team, interest, year, carried, forward, sick, annual, approval, flights, class, night, usd
- Warning: UNSUPPORTED by the documents (the agent must refuse these): ['company pension scheme']

## 3. Agent definition

| Tool | Attached | Reason |
|---|---|---|
| csv_lookup | no | no CSV files in the document folder |
| calculator | no | no need found in the requirements |
| weather_api | no | no need found in the requirements |
| current_date | no | no need found in the requirements |

- Retrieval: chunks of 200 words, top_k 3, evidence-gate coverage 0.5
- Refusal topics: legal, medical, personal data, company pension scheme
- Persona written by: model

## 4. Eval set (generated, checked, frozen)

- 22 cases: answerable 12, multi_doc 3, ambiguous 3, out_of_scope 4
- 11 generated cases rejected by code checks (see evals_rejected.json)
- SHA-256 `f3ec1f3d1c3bce88...` (the judge refuses a changed set)

## 5. Scorecard per iteration

| Iteration | Change | Score | Fixed | Broken | Result |
|---|---|---|---|---|---|
| 0 | (baseline agent) | 18/22 | | | |
| 1 | add instruction: Cite every source you use. If you combine facts from two sources, cite both source numbers. | 18/22 (82%) | - | - | no gain: change rolled back |
| 2 | add instruction: Always include the exact figures, units and conditions from the SOURCES in your answer. | 18/22 (82%) | - | - | no gain: change rolled back |
| 3 | add instruction: If the question asks more than one thing, answer every part, each from its own source. | 18/22 (82%) | - | - | no gain: change rolled back |

By case type (final): answerable 11/12; multi_doc 0/3; ambiguous 3/3; out_of_scope 4/4

## 6. What still fails, and why

- **q03** (answerable) _What is the maximum reimbursement for hotel accommodation abroad?_  
  prompt_defect (incomplete): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: missing ['USD 180'].
- **q13** (multi_doc) _How much is the daily meal allowance for business travel in India, and what is the maximum amount for gifts that must be declared to the compliance team?_  
  prompt_defect (citation): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: model's answer had the facts but was blocked (refused_unsupported_number).
- **q14** (multi_doc) _How many weeks of paid maternity leave are available to mothers, and how many days per week can an employee work from home after completing 6 months of service?_  
  prompt_defect (multi_part): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: missing ['2 days per week'].
- **q15** (multi_doc) _What happens to an expense claim submitted after the 30-day deadline, and what is the dress code for the office?_  
  prompt_defect (citation): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: model's answer had the facts but was blocked (refused_unsupported_number).

## 7. Known limitations

- One small local model (Qwen 2.5 1.5B) writes the persona, generates tests and judges unclear facts; code checks limit, but do not remove, its influence.
- Words are matched exactly (no stemming): 'policies' does not match 'policy', which can cause retrieval misses and wrong gate refusals.
- Multi-document cases join two questions from different documents; they test answers that span two documents, not reasoning across them.
- Key facts are short phrases; a correct answer that states a fact in different words (e.g. spelled-out numbers) can be scored as missing.
- Repair tunes the agent on the same eval set it is scored on, so the final score may be optimistic.

## 8. Files

requirements.json, groundability.json, agent.json (built), agent_final.json (best after repair), build_log.json, evals.json, evals_rejected.json, scorecard_iteration_N.json, repair_log.json, and *_log.json files with every model prompt and reply.
