"""Tests for tool use inside the runner (fake model, no network)."""

import tempfile
import unittest
from pathlib import Path

from forgeai.runtime.agent_schema import parse_agent_definition
from forgeai.runtime.runner import AgentRunner
from tests.helpers import FakeLLM, agent_dict, answer, refusal


def tool_call(name, tool_input):
    return {"answer": "", "sources": [], "refused": False, "refusal_reason": "",
            "tool_name": name, "tool_input": tool_input}


def plan(name="none", tool_input=""):
    return {"tool_name": name, "tool_input": tool_input}


def runner(fake, tools, **changes):
    return AgentRunner(parse_agent_definition(agent_dict(tools=tools, **changes)), llm_fn=fake)


class TestToolUse(unittest.TestCase):
    Q = "How many days of annual leave do staff get in two years?"

    def test_planner_runs_calculator_and_result_is_citable(self):
        # Chunks are [1]..[n-1]; the planner's tool result is the next number.
        fake = FakeLLM(plan("calculator", "24 * 2"), answer("48 working days over two years.", [1, 4]))
        r = runner(fake, ["calculator"])
        response = r.ask(self.Q)
        n = len(response.trace.retrieved) + 1
        self.assertEqual(response.trace.tool_calls[0].source_number, n)
        self.assertEqual(response.trace.tool_calls[0].requested_by, "planner")
        self.assertEqual(fake.calls[0]["json_schema"]["properties"]["tool_name"]["enum"], ["calculator", "none"])
        self.assertFalse(response.refused, response.refusal_reason)
        self.assertIn("tool:calculator:1", response.sources)
        self.assertEqual(response.trace.unsupported_numbers, [])   # 48 comes from the tool
        self.assertIn(f"[{n}] (tool calculator, input: 24 * 2)", fake.calls[1]["prompt"])
        self.assertIn("24 * 2 = 48", fake.calls[1]["prompt"])

    def test_model_can_still_request_tool_while_answering(self):
        fake = FakeLLM(plan(), tool_call("calculator", "24 * 2"), answer("48 working days.", [1, 4]))
        response = runner(fake, ["calculator"]).ask(self.Q)
        self.assertEqual(response.trace.tool_calls[0].requested_by, "model")
        self.assertFalse(response.refused, response.refusal_reason)

    def test_unusable_planner_reply_is_skipped(self):
        fake = FakeLLM("garbage", answer("24 working days.", [1]))
        response = runner(fake, ["calculator"]).ask("How many days of annual leave do shore staff get?")
        self.assertFalse(response.refused)
        self.assertTrue(any("planner reply unusable" in w for w in response.trace.warnings))

    def test_computed_number_without_tool_is_blocked(self):
        fake = FakeLLM(plan(), answer("48 working days over two years.", [1]))
        response = runner(fake, ["calculator"]).ask(self.Q)
        self.assertEqual(response.trace.decision, "refused_unsupported_number")

    def test_tool_not_given_to_agent_is_not_run(self):
        fake = FakeLLM(plan(), tool_call("weather_api", "Mumbai"), refusal("no weather tool"))
        response = runner(fake, ["calculator"]).ask(self.Q)
        self.assertFalse(response.trace.tool_calls[0].ok)
        self.assertIn("not available to this agent", response.trace.tool_calls[0].output)
        self.assertTrue(response.refused)

    def test_agent_without_tools_cannot_use_tools(self):
        fake = FakeLLM(tool_call("calculator", "24*2"), answer("24 working days.", [1]))
        response = runner(fake, []).ask("How many days of annual leave do shore staff get?")
        self.assertFalse(response.trace.tool_calls[0].ok)
        self.assertIn("TOOLS: none", fake.calls[0]["system"])

    def test_tool_limit(self):
        fake = FakeLLM(plan(), *[tool_call("calculator", "1+1")] * 3)
        response = runner(fake, ["calculator"]).ask(self.Q)
        self.assertEqual(response.trace.decision, "refused_tool_limit")
        self.assertEqual(len(response.trace.tool_calls), 2)

    def test_planner_not_called_without_planned_tools(self):
        fake = FakeLLM(answer("24 working days.", [1]))
        runner(fake, ["current_date"]).ask("How many days of annual leave do shore staff get?")
        self.assertEqual(len(fake.calls), 1)

    def test_system_prompt_lists_only_attached_tools(self):
        fake = FakeLLM(answer("24 working days.", [1]))
        runner(fake, ["current_date"]).ask("How many days of annual leave do shore staff get?")
        self.assertIn("- current_date:", fake.calls[0]["system"])
        self.assertNotIn("weather_api", fake.calls[0]["system"])


class TestGateWithTools(unittest.TestCase):
    def test_no_document_match_still_reaches_tool(self):
        fake = FakeLLM(answer("Today is Friday.", [1]))  # date was added automatically as [1]
        response = runner(fake, ["current_date"]).ask("What is the date today?")
        self.assertEqual(response.trace.tool_calls[0].requested_by, "runner")
        self.assertEqual(response.trace.retrieved, [])
        self.assertTrue(any("gate skipped" in w for w in response.trace.warnings))
        self.assertFalse(response.refused, response.refusal_reason)
        self.assertEqual(response.sources, ["tool:current_date:1"])

    def test_without_tools_gate_still_refuses(self):
        fake = FakeLLM()
        response = runner(fake, []).ask("What is the date today?")
        self.assertEqual(response.trace.decision, "refused_no_evidence")
        self.assertEqual(fake.calls, [])

    def test_answer_with_no_source_still_refused(self):
        fake = FakeLLM(answer("It is sunny.", []))
        response = runner(fake, ["current_date"]).ask("Is it sunny in Mumbai?")
        self.assertEqual(response.trace.decision, "refused_no_citation")


class TestCsvOnlyAgent(unittest.TestCase):
    def test_csv_only_folder_with_csv_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "parts.csv").write_text("part,price\nvalve,12.50\n", encoding="utf-8")
            fake = FakeLLM(answer("A valve costs 12.50.", [1]))  # CSV row added automatically as [1]
            r = runner(fake, ["csv_lookup"], documents={"folder": tmp})
            self.assertIsNone(r.index)
            response = r.ask("What does a valve cost?")
            self.assertFalse(response.refused, response.refusal_reason)
            self.assertIn("parts.csv row 2", response.trace.tool_calls[0].output)


if __name__ == "__main__":
    unittest.main()


class TestAutoCsvLookup(unittest.TestCase):
    def test_fuel_injector_question_gets_csv_row_without_model_tool_call(self):
        # The exact case real Qwen failed: it never called csv_lookup itself.
        fake = FakeLLM(answer("The unit price is 145.00 USD.", [1]))
        r = runner(fake, ["csv_lookup"])
        response = r.ask("What is the unit price of the fuel injector nozzle?")
        self.assertIn("spare_parts.csv row 2", fake.calls[0]["prompt"])
        self.assertFalse(response.refused, response.refusal_reason)

    def test_no_matching_rows_are_not_shown(self):
        fake = FakeLLM(refusal())
        runner(fake, ["csv_lookup"]).ask("How many days of annual leave do shore staff get?")
        self.assertNotIn("tool csv_lookup", fake.calls[0]["prompt"])

    def test_total_cost_flow(self):
        fake = FakeLLM(plan("calculator", "145 * 4"), answer("4 nozzles cost 580 USD.", [1, 2, 3, 4, 5]))  # extra numbers are dropped
        response = runner(fake, ["csv_lookup", "calculator"]).ask("How much do 4 fuel injector nozzles cost in total?")
        self.assertFalse(response.refused, response.refusal_reason)
        self.assertIn("unit_price_usd=145.00", fake.calls[0]["prompt"])   # planner saw the CSV row
        self.assertEqual([c.requested_by for c in response.trace.tool_calls], ["runner", "planner"])


class TestToolSpeedFixes(unittest.TestCase):
    def test_planner_calculator_without_operator_is_skipped(self):
        # Real Qwen planned calculator("145.00") for a simple lookup.
        fake = FakeLLM(plan("calculator", "145.00"), answer("The unit price is 145.00 USD.", [1]))
        response = runner(fake, ["csv_lookup", "calculator"]).ask("What is the unit price of the fuel injector nozzle?")
        self.assertEqual([c.name for c in response.trace.tool_calls], ["csv_lookup"])
        self.assertTrue(any("no calculation" in w for w in response.trace.warnings))
        self.assertFalse(response.refused, response.refusal_reason)

    def test_repeated_tool_call_is_not_rerun(self):
        q = "What is the unit price of the fuel injector nozzle?"
        fake = FakeLLM(plan(), tool_call("csv_lookup", q), answer("145.00 USD.", [1]))
        response = runner(fake, ["csv_lookup", "calculator"]).ask(q)
        self.assertFalse(response.refused, response.refusal_reason)
        self.assertIn("already source [1]", fake.calls[2]["prompt"])
        self.assertEqual(sum(1 for c in response.trace.tool_calls if c.name == "csv_lookup"), 2)  # auto + repeat record
