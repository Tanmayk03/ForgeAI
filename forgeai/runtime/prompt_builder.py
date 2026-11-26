"""
prompt_builder.py - builds the two prompts sent to the model for each question.

  system prompt = the agent's own instructions + generic grounding rules
                  (template: prompts/runner_system.txt)
  user prompt   = numbered sources + the question

The template is a separate file so that:
  * the rules are the same for every agent (no industry is hard-coded),
  * prompts are committed and reviewable, as the assignment requires,
  * Repair can later change the agent's fields without touching Python code.

We use string.Template ($name placeholders) rather than str.format, because
document text and instructions may contain { } characters.
"""

from string import Template

from forgeai import config
from forgeai.retrieval.models import SearchResult
from forgeai.runtime.agent_schema import AgentDefinition

SYSTEM_TEMPLATE_PATH = config.PROMPTS_DIR / "runner_system.txt"


def _join_topics(topics: list[str]) -> str:
    return "; ".join(topics) if topics else "(none)"


def build_system_prompt(agent: AgentDefinition) -> str:
    """Fill the system template with this agent's settings."""
    template = Template(SYSTEM_TEMPLATE_PATH.read_text(encoding="utf-8"))
    return template.substitute(
        system_instructions=agent.system_instructions.strip(),
        refusal_topics=_join_topics(agent.scope.refusal_topics),
        in_scope_topics=_join_topics(agent.scope.in_scope_topics),
        max_answer_words=agent.output_contract.max_answer_words,
    ).strip()


def build_user_prompt(question: str, results: list[SearchResult]) -> str:
    """
    Number each retrieved chunk [1], [2], ... and put the question last.
    Short numbers are far easier for a small model to cite than long chunk IDs;
    the runner maps the numbers back to chunk IDs afterwards.
    """
    lines = ["SOURCES:"]
    for number, result in enumerate(results, start=1):
        chunk = result.chunk
        lines.append(f"[{number}] ({chunk.doc_name}, page {chunk.page})")
        lines.append(chunk.text)
        lines.append("")
    lines.append(f"QUESTION: {question.strip()}")
    return "\n".join(lines)


def build_retry_note(reason: str) -> str:
    """Text added to the user prompt when the previous reply was unusable."""
    return (
        "\n\nYour previous reply could not be used (" + reason + "). "
        "Reply again with ONLY the JSON object, using the exact fields: "
        "answer, sources, refused, refusal_reason."
    )
