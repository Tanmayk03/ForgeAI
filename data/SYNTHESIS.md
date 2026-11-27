# How the demo documents were made

The documents in `data/demo1_hr/` and `data/demo2_spares/` are **synthetic** (fictional company
"Seaview Shipping Ltd", fictional suppliers, `.example` e-mail domains). They were written by an LLM
(Claude) and the PDFs were rendered from plain text with reportlab. The assignment allows
LLM-synthesized documents if the synthesis prompts are committed; these are the prompts used.

## Demo 1 - HR policy assistant (`data/demo1_hr/`, brief: `data/briefs/demo1_hr.txt`)

> Write three short, realistic HR documents for shore staff of a fictional shipping company,
> "Seaview Shipping Ltd": (1) a leave policy covering annual leave with accrual and carry-forward,
> sick leave with a medical-certificate rule, parental leave, public holidays, compassionate and
> unpaid leave; (2) a business travel and expenses policy covering approvals, flight class, hotel
> and meal limits in INR and USD, claim deadlines, travel during leave and travel advances;
> (3) a code of conduct covering working hours, remote work, gifts, conflicts of interest,
> reporting concerns and dress code. Use specific, checkable numbers. Do NOT mention pensions.

The brief deliberately also asks about the **pension scheme**, which no document covers, to test
the groundability guard (requirement 2).

## Demo 2 - Ship-spares procurement assistant (`data/demo2_spares/`, brief: `data/briefs/demo2_spares.txt`)

> Write (1) a CSV catalogue of 25 marine spare parts with columns part_no, description, category,
> unit_price_usd, supplier, lead_time_days, min_order_qty, split between two fictional suppliers,
> "Marisupply Ltd" and "Oceanparts Co"; (2) supply terms for each supplier covering ordering,
> prices and volume discounts, delivery and freight, payment terms and warranty - with different
> numbers for each supplier; (3) the buyer's procurement rules covering approval limits by order
> value, quotation requirements, critical spares and urgent orders.

This set needs data look-ups and arithmetic (csv_lookup + calculator) but no weather or date tool,
so it tests that the Builder attaches only the tools the brief needs.

## Rehearsal - port agent assistant (`data/rehearsal_port/`, brief: `data/briefs/rehearsal_port.txt`)

Made for an unseen-brief rehearsal: a new domain, a mix of PDF, Markdown, CSV and an unsupported
`.docx`, and a brief that asks for weather and date-based answers (tools the two demos did not need).

> Write short, fictional training documents for the Port of Kochi: (1) rules for visiting vessels
> covering arrival notice deadlines, pilotage, berthing, safety permits and weather restrictions;
> (2) crew shore leave rules covering shore passes, hours, restrictions and the seafarers' centre;
> (3) a CSV of 10 port fees with fee_code, fee_item, unit and amount_inr. Use specific numbers.
