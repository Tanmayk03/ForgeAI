# ForgeAI report: sparesassistant

**Final score: 10/24 cases passed (42%)**, 0 error(s), from iteration_0. Baseline: 10/24 (42%).

Brief: _I need an assistant for our ship-spares procurement team. It should find spare parts by part number or description, give prices, suppliers, lead times and minimum order quantities, calculate order totals and discounts, and explain supplier terms and our procurement approval rules._

Time per stage: manager 29 s, guard 0 s, builder 9 s, eval generation 98 s, evaluation 599 s, repair 1543 s

## 1. Requirements

- Audience: Ship-spares procurement team members
- Tone: Formal and concise
- Must answer: Part number or description, Price, Lead time, Minimum order quantity, Supplier terms, Procurement approval rules
- Must refuse: Legal, Medical, Personal Data
- Clarifications: 0 answered by the requester, 5 assumed (defaults)

## 2. Groundability (can the documents support the brief?)

| Topic | Status | Decided by | Reason |
|---|---|---|---|
| Part number or description | supported | code | most key words found in CSV data |
| Price | supported | code | all key words found together in one passage |
| Lead time | supported | code | all key words found together in one passage |
| Minimum order quantity | supported | code | all key words found together in one passage |
| Supplier terms | supported | code | most key words found in CSV data |
| Procurement approval rules | supported | code | all key words found together in one passage |

- Warning: documents also cover keywords not in any must-answer topic: marisupply, freight, cost, discount, per, oceanpart, percent, through, online, portal, spare, critical, require, quotation, least

## 3. Agent definition

| Tool | Attached | Reason |
|---|---|---|
| csv_lookup | yes | CSV data related to the brief: spares_catalogue.csv (shared words: description, lead, ltd, order, part) |
| calculator | yes | requirements mention: calculate, discount, price, quantity, total |
| weather_api | no | no need found in the requirements |
| current_date | no | no need found in the requirements |

- Retrieval: chunks of 200 words, top_k 3, evidence-gate coverage 0.5
- Refusal topics: Legal, Medical, Personal Data
- Persona written by: model

## 4. Eval set (generated, checked, frozen)

- 24 cases: answerable 13, multi_doc 3, ambiguous 2, out_of_scope 6
- 12 generated cases rejected by code checks (see evals_rejected.json)
- SHA-256 `e4bb028f3266e33a...` (the judge refuses a changed set)

## 5. Scorecard per iteration

| Iteration | Change | Score | Fixed | Broken | Result |
|---|---|---|---|---|---|
| 0 | (baseline agent) | 10/24 | | | |
| 1 | add instruction: Cite every source you use. If you combine facts from two sources, cite both source numbers. | 10/24 (42%) | q10 | q19 | no gain: change rolled back |
| 2 | add instruction: Always include the exact figures, units and conditions from the SOURCES in your answer. | 9/24 (38%) | - | q19 | score dropped: change rolled back |
| 3 | add instruction: Refuse questions about the refusal topics even when the SOURCES mention related words. | 10/24 (42%) | q15 | q19 | no gain: change rolled back |

By case type (final): answerable 4/13; multi_doc 0/3; ambiguous 1/2; out_of_scope 5/6

## 6. What still fails, and why

- **q01** (answerable) _What is the minimum order quantity for orders placed through the Oceanparts online portal?_  
  prompt_defect (incomplete): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: missing ['USD 5,000'].
- **q05** (answerable) _What is the discount rate for orders above USD 10,000?_  
  prompt_defect (incomplete): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: missing ['3 percent discount'].
- **q06** (answerable) _How much does air freight cost per order?_  
  prompt_defect (incomplete): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: missing ['USD 400 per order'].
- **q07** (answerable) _How many days does sea freight add to the lead time?_  
  prompt_defect (citation): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: cited ['oceanparts_supply_terms.pdf'], expected ['marisupply_supply_terms.md'].
- **q09** (answerable) _What is the minimum number of critical spares required on board?_  
  hallucination (hallucination): the model's own answer did not match the documents (the runner blocked it or it was wrong). Evidence: model's answer lacked the facts and was blocked (refused_unsupported_number): '1'.
- **q10** (answerable) _When can urgent delivery charges be used?_  
  prompt_defect (citation): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: cited ['oceanparts_supply_terms.pdf'], expected ['procurement_rules.md'].
- **q11** (multi_doc) _What is the discount rate for orders above USD 10,000, and what must be done for any order above USD 5,000?_  
  prompt_defect (citation): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: cited ['marisupply_supply_terms.md'], expected ['marisupply_supply_terms.md', 'procurement_rules.md'].
- **q12** (multi_doc) _What is the restocking fee for returned unused parts, and how many days does sea freight add to the lead time?_  
  prompt_defect (citation): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: cited ['oceanparts_supply_terms.pdf'], expected ['marisupply_supply_terms.md', 'oceanparts_supply_terms.pdf'].
- **q13** (multi_doc) _What is the delivery lead time for urgent orders, and what is the minimum number of critical spares required on board?_  
  prompt_defect (citation): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: cited ['oceanparts_supply_terms.pdf'], expected ['oceanparts_supply_terms.pdf', 'procurement_rules.md'].
- **q14** (answerable) _What is the category for SP-1019?_  
  retrieval_miss (retrieval): the search did not bring the passage with the answer to the model. Evidence: expected ['spares_catalogue.csv'], retrieved ['procurement_rules.md'].
- **q15** (answerable) _What is the min order qty for SP-1009?_  
  prompt_defect (citation): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: model's answer had the facts but was blocked (refused_unsupported_number).
- **q16** (answerable) _What is the supplier for SP-1001?_  
  retrieval_miss (retrieval): the search did not bring the passage with the answer to the model. Evidence: expected ['spares_catalogue.csv'], retrieved ['procurement_rules.md'].
- **q17** (ambiguous) _What's the discount rate for orders over $10,000?_  
  prompt_defect (incomplete): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: missing ['3 percent discount'].
- **q20** (out_of_scope) _How do you calculate the lead time for procuring a spare part that requires a medical examination before approval?_  
  prompt_defect (refusal_rule): the right passage was found, but the answer was incomplete, badly cited or should have been a refusal. Evidence: answered a question that should be refused.

## 7. Known limitations

- One small local model (Qwen 2.5 1.5B) writes the persona, generates tests and judges unclear facts; code checks limit, but do not remove, its influence.
- Words are matched exactly (no stemming): 'policies' does not match 'policy', which can cause retrieval misses and wrong gate refusals.
- Multi-document cases join two questions from different documents; they test answers that span two documents, not reasoning across them.
- Key facts are short phrases; a correct answer that states a fact in different words (e.g. spelled-out numbers) can be scored as missing.
- Repair tunes the agent on the same eval set it is scored on, so the final score may be optimistic.

## 8. Files

requirements.json, groundability.json, agent.json (built), agent_final.json (best after repair), build_log.json, evals.json, evals_rejected.json, scorecard_iteration_N.json, repair_log.json, and *_log.json files with every model prompt and reply.
