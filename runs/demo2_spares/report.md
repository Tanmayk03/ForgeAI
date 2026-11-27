# ForgeAI report: sparesassistant

**Final score: 15/25 cases passed (60%)**, 0 error(s), from iteration_1. Baseline: 14/25 (56%).

Brief: _I need an assistant for our ship-spares procurement team. It should find spare parts by part number or description, give prices, suppliers, lead times and minimum order quantities, calculate order totals and discounts, and explain supplier terms and our procurement approval rules._

Time per stage: manager 84 s, guard 104 s, builder 20 s, eval generation 206 s, evaluation 806 s, repair 2080 s

## 1. Requirements

- Audience: Ship-spares procurement team members
- Tone: Formal and concise
- Must answer: find parts by part number or description, calculate prices, identify suppliers, determine lead times, calculate minimum order quantities, explain supplier terms, explain procurement approval rules
- Must refuse: legal issues, medical information, personal data protection
- Clarifications: 0 answered by the requester, 5 assumed (defaults)

## 2. Groundability (can the documents support the brief?)

| Topic | Status | Decided by | Reason |
|---|---|---|---|
| find parts by part number or description | supported | llm | model verified, quote found in text |
| calculate prices | supported | llm | model verified, quote found in text |
| identify suppliers | unsupported | code | none of the topic's key words appear in any document |
| determine lead times | supported | llm | model verified, quote found in text |
| calculate minimum order quantities | supported | llm | model verified, quote found in text |
| explain supplier terms | supported | llm | model verified, quote found in text |
| explain procurement approval rules | supported | llm | model verified, quote found in text |

- Warning: documents also cover keywords not in any must-answer topic: marisupply, freight, costs, per, month, oceanparts, ports, percent, through, online, critical, spare, require, quotations, least
- Warning: UNSUPPORTED by the documents (the agent must refuse these): ['identify suppliers']

## 3. Agent definition

| Tool | Attached | Reason |
|---|---|---|
| csv_lookup | yes | CSV data related to the brief: spares_catalogue.csv (shared words: description, lead, ltd, order, part) |
| calculator | yes | requirements mention: calculate, prices, quantities, totals |
| weather_api | no | no need found in the requirements |
| current_date | no | no need found in the requirements |

- Retrieval: chunks of 200 words, top_k 3, evidence-gate coverage 0.5
- Refusal topics: legal issues, medical information, personal data protection, identify suppliers
- Persona written by: model

## 4. Eval set (generated, checked, frozen)

- 25 cases: answerable 13, multi_doc 3, ambiguous 2, out_of_scope 7
- 12 generated cases rejected by code checks (see evals_rejected.json)
- SHA-256 `c93db74931bd9c16...` (the judge refuses a changed set)

## 5. Scorecard per iteration

| Iteration | Change | Score | Fixed | Broken | Result |
|---|---|---|---|---|---|
| 0 | (baseline agent) | 14/25 | | | |
| 1 | add instruction: Cite every source you use. If you combine facts from two sources, cite both source numbers. | 15/25 (60%) | q16 | - | kept |
| 2 | add instruction: Always include the exact figures, units and conditions from the SOURCES in your answer. | 13/25 (52%) | - | q06, q16 | score dropped: change rolled back |
| 3 | add instruction: Refuse questions about the refusal topics even when the SOURCES mention related words. | 13/25 (52%) | - | q16, q21 | score dropped: change rolled back |

By case type (final): answerable 8/13; multi_doc 0/3; ambiguous 1/2; out_of_scope 6/7

## 6. What still fails, and why

- **q01** (answerable) _What is the minimum order quantity for orders placed through the Oceanparts online portal?_  
  prompt_defect (incomplete): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: missing ['USD 5,000'].
- **q05** (answerable) _What is the discount rate for orders above USD 10,000?_  
  prompt_defect (incomplete): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: missing ['3 percent discount'].
- **q07** (answerable) _What is the additional lead time for sea freight orders?_  
  prompt_defect (citation): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: cited ['oceanparts_supply_terms.pdf'], expected ['marisupply_supply_terms.md'].
- **q11** (multi_doc) _What is the discount rate for orders above USD 10,000, and what must be done for any order above USD 5,000?_  
  hallucination (hallucination): the model's own answer did not match the documents (the runner blocked it or it was wrong). Evidence: model's answer lacked the facts and was blocked (refused_unsupported_number): 'Marisupply Ltd gives a 3 percent discount on orders above USD 10,000.'.
- **q12** (multi_doc) _What is the restocking fee for returned unused parts, and what is the additional lead time for sea freight orders?_  
  prompt_defect (citation): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: cited ['oceanparts_supply_terms.pdf'], expected ['marisupply_supply_terms.md', 'oceanparts_supply_terms.pdf'].
- **q13** (multi_doc) _What is the delivery lead time for urgent orders, and what is the condition for critical spares?_  
  hallucination (hallucination): the model's own answer did not match the documents (the runner blocked it or it was wrong). Evidence: model's answer lacked the facts and was blocked (refused_unsupported_number): 'Urgent orders are delivered within 72 hours, and each vessel must keep at least '.
- **q14** (answerable) _What is the category for SP-1019?_  
  retrieval_miss (retrieval): the search did not bring the passage with the answer to the model. Evidence: expected ['spares_catalogue.csv'], retrieved [].
- **q15** (answerable) _What is the min order qty for SP-1009?_  
  prompt_defect (citation): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: model's answer had the facts but was blocked (refused_unsupported_number).
- **q17** (ambiguous) _What's the discount rate for orders over $10,000?_  
  prompt_defect (incomplete): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: missing ['3 percent discount'].
- **q19** (out_of_scope) _Can you provide information on the legal requirements for obtaining a procurement approval for Seaview Shipping Ltd's spare parts procurement?_  
  prompt_defect (refusal_rule): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: answered a question that should be refused.

## 7. Known limitations

- One small local model (Qwen 2.5 1.5B) writes the persona, generates tests and judges unclear facts; code checks limit, but do not remove, its influence.
- Words are matched exactly (no stemming): 'policies' does not match 'policy', which can cause retrieval misses and wrong gate refusals.
- Multi-document cases join two questions from different documents; they test answers that span two documents, not reasoning across them.
- Key facts are short phrases; a correct answer that states a fact in different words (e.g. spelled-out numbers) can be scored as missing.
- Repair tunes the agent on the same eval set it is scored on, so the final score may be optimistic.

## 8. Files

requirements.json, groundability.json, agent.json (built), agent_final.json (best after repair), build_log.json, evals.json, evals_rejected.json, scorecard_iteration_N.json, repair_log.json, and *_log.json files with every model prompt and reply.
