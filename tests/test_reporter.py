"""End-to-end factory test with fake models, then the Reporter."""

import json
import tempfile
import unittest
from pathlib import Path

from forgeai.roles.builder import run_builder, save_build
from forgeai.roles.eval_generator import generate_evals, save_evals
from forgeai.roles.guard import run_guard, save_guard_report
from forgeai.roles.judge import run_evals, save_scorecard
from forgeai.roles.manager import run_manager, save_requirements
from forgeai.roles.repair import run_repair
from forgeai.roles.reporter import write_report
from forgeai.runtime.runner import AgentRunner
from tests.helpers import SAMPLE_DOCS, FakeLLM, answer
from tests.test_eval_generator import ScriptedLLM
from tests.test_judge import AgentLLM
from tests.test_manager import BRIEF, draft


class TestPipelineAndReport(unittest.TestCase):
    def test_full_pipeline_writes_honest_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            req = run_manager(BRIEF, SAMPLE_DOCS, auto=True, llm_fn=FakeLLM(draft()))
            save_requirements(req, run)
            req = run_guard(req, llm_fn=FakeLLM({"verdict": "supported", "evidence_quote": ""}))
            save_guard_report(req, run)
            agent, build_log = run_builder(req, llm_fn=FakeLLM({"system_instructions": "You are a careful policy assistant for staff."}))
            save_build(agent, build_log, run)
            evals, rejected = generate_evals(req, llm_fn=ScriptedLLM())
            save_evals(evals, rejected, run)
            llm = AgentLLM({"paid annual": answer("24 working days of paid leave.", [1])})
            baseline = run_evals(evals, AgentRunner(agent, llm_fn=llm), llm_fn=FakeLLM(*[{"states_fact": False}] * 50))
            save_scorecard(baseline, run)
            run_repair(agent, evals, baseline, run, agent_llm_fn=llm,
                       judge_llm_fn=FakeLLM(*[{"states_fact": False}] * 200))
            text = write_report(run).read_text(encoding="utf-8")

            self.assertIn("**Final score:", text)
            self.assertIn("## 6. What still fails, and why", text)
            self.assertIn("wrong_refusal", text)                  # most cases refused by this fake agent
            self.assertIn("## 7. Known limitations", text)
            self.assertIn("| csv_lookup | no |", text)
            self.assertTrue((run / "agent_final.json").is_file())
            self.assertTrue(json.loads((run / "repair_log.json").read_text()))


if __name__ == "__main__":
    unittest.main()
