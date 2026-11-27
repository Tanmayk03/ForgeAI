# ForgeAI report: kochi_agent_assistant

**Final score: 17/22 cases passed (77%)**, 0 error(s), from iteration_1. Baseline: 14/22 (64%).

Brief: _I need an assistant for ship agents bringing vessels into the Port of Kochi. It should explain arrival notice deadlines, pilotage and berthing rules, safety permits and crew shore leave, look up port fees and calculate the total cost of a port call, tell agents how many days are left until a notice deadline from today, and give the current weather forecast at the port._

Time per stage: manager 34 s, guard 40 s, builder 10 s, eval generation 118 s, evaluation 375 s, repair 445 s

## 1. Requirements

- Audience: Ship agents and other personnel working in the Port of Kochi.
- Tone: Formal and concise
- Must answer: arrival notice deadlines, pilotage and berthing rules, safety permits, port fees, days until notice deadline, current weather forecast
- Must refuse: legal, medical, personal data
- Clarifications: 0 answered by the requester, 5 assumed (defaults)

## 2. Groundability (can the documents support the brief?)

| Topic | Status | Decided by | Reason |
|---|---|---|---|
| arrival notice deadlines | supported | llm | model verified, quote found in text |
| pilotage and berthing rules | supported | code | all key words found together in one passage |
| safety permits | supported | code | all key words found together in one passage |
| port fees | supported | code | most key words found in CSV data |
| days until notice deadline | supported | llm | model verified, quote found in text |
| current weather forecast | supported | llm | model verified, quote found in text |

- Warning: documents also cover keywords not in any must-answer topic: crew, shore, pass, leave, ashore, send, dangerous, good, time, berth

## 3. Agent definition

| Tool | Attached | Reason |
|---|---|---|
| csv_lookup | yes | CSV data related to the brief: port_fees.csv (shared words: call, crew, day, fee, permit) |
| calculator | yes | requirements mention: calculate, cost, total |
| weather_api | yes | requirements mention: forecast, weather |
| current_date | yes | requirements mention: deadline, today |

- Retrieval: chunks of 200 words, top_k 3, evidence-gate coverage 0.5
- Refusal topics: legal, medical, personal data
- Persona written by: model

## 4. Eval set (generated, checked, frozen)

- 22 cases: answerable 14, multi_doc 3, ambiguous 2, out_of_scope 3
- 18 generated cases rejected by code checks (see evals_rejected.json)
- SHA-256 `ba88fc6a7284afde...` (the judge refuses a changed set)

## 5. Scorecard per iteration

| Iteration | Change | Score | Fixed | Broken | Result |
|---|---|---|---|---|---|
| 0 | (baseline agent) | 14/22 | | | |
| 1 | add instruction: Cite every source you use. If you combine facts from two sources, cite both source numbers. | 17/22 (77%) | q04, q10, q18, q19 | q17 | kept |
| iteration_0 re-judged | same answers, current judge (code checks only) | 15/22 (68%) | | | see judge note |

By case type (final): answerable 11/14; multi_doc 1/3; ambiguous 2/2; out_of_scope 3/3

## 6. What still fails, and why

- **q05** (answerable) _What is the maximum amount of alcohol allowed on a shore leave?_  
  hallucination (hallucination): the model's own answer did not match the documents (the runner blocked it or it was wrong). Evidence: model's answer lacked the facts and was blocked (refused_unsupported_number): '250 INR'.
- **q06** (answerable) _What is the maximum number of cigarettes allowed on a shore leave?_  
  hallucination (hallucination): the model's own answer did not match the documents (the runner blocked it or it was wrong). Evidence: model's answer lacked the facts and was blocked (refused_unsupported_number): '200'.
- **q12** (multi_doc) _What is the maximum number of cigarettes allowed on a shore leave, and what is required for bunkering?_  
  prompt_defect (citation): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: cited ['crew_shore_leave.md', 'port_fees.csv'], expected ['crew_shore_leave.md', 'port_rules.pdf'].
- **q13** (multi_doc) _What is the condition for night pilotage, and what is the maximum amount of alcohol allowed on a shore leave?_  
  prompt_defect (citation): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: model's answer had the facts but was blocked (refused_unsupported_number).
- **q17** (answerable) _What is the unit for PF-01?_  
  retrieval_miss (retrieval): the search did not bring the passage with the answer to the model. Evidence: expected ['port_fees.csv'], retrieved [].

## 7. Known limitations

- One small local model (Qwen 2.5 1.5B) writes the persona, generates tests and judges unclear facts; code checks limit, but do not remove, its influence.
- Words are matched exactly (no stemming): 'policies' does not match 'policy', which can cause retrieval misses and wrong gate refusals.
- Multi-document cases join two questions from different documents; they test answers that span two documents, not reasoning across them.
- Key facts are short phrases; a correct answer that states a fact in different words (e.g. spelled-out numbers) can be scored as missing.
- Repair tunes the agent on the same eval set it is scored on, so the final score may be optimistic.
- The guard counts a topic as supported when at least half of its key words appear in a CSV, so a two-word topic can pass on one word.

## 8. Files

requirements.json, groundability.json, agent.json (built), agent_final.json (best after repair), build_log.json, evals.json, evals_rejected.json, scorecard_iteration_N.json, repair_log.json, and *_log.json files with every model prompt and reply.
