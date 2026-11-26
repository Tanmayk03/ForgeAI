"""
prompt_builder.py - builds the two prompts sent to the model for each question.

  system prompt = the agent's own instructions + generic grounding rules
                  + the agent's tools (template: prompts/runner_system.txt)
  user prompt   = numbered sources (document chunks and tool results) + the question

The template is a separate file so that:
  * the rules are the same for every agent (no industry is hard-coded),
  * prompts are committed and reviewable, as the assignment requires,
  * Repair can later change the agent's fields without touching Python code.

We use string.Template ($name placeholders) rather than str.format, because
document text and instructions may contain { } characters.
"""

from dataclasses import dataclass
from string import Template

from forgeai import config
from forgeai.runtime.agent_schema import AgentDefinition
from forgeai.tools.registry import TOOLS

SYSTEM_TEMPLATE_PATH = config.PROMPTS_DIR / "runner_system.txt"
PLANNER_TEMPLATE_PATH = config.PROMPTS_DIR / "tool_planner.txt"

# Tools the runner runs by itself (cheap, no model decision needed) and tools
# the model must choose in the planning step (they need an input it writes).
AUTO_TOOLS = ("csv_lookup", "current_date")
PLANNED_TOOLS = ("calculator", "weather_api")


@dataclass
class PromptSource:
    """One numbered source shown to the model: a document chunk or a tool result."""

    source_id: str  # chunk ID, or "tool:<name>:<n>"
    header: str     # e.g. "leave_policy.pdf, page 1" or "tool calculator"
    text: str


def _join_topics(topics: list[str]) -> str:
    return "; ".join(topics) if topics else "(none)"


def build_tools_section(tool_names: list[str]) -> str:
    """Describe the agent's tools, or say there are none."""
    if not tool_names:
        return 'TOOLS: none. Always leave "tool_name" and "tool_input" empty.'
    lines = [
        "TOOLS you may use when the SOURCES are not enough:",
        *[f'- {name}: {TOOLS[name].description} tool_input: {TOOLS[name].input_hint}.' for name in tool_names],
        'To use a tool, reply with "answer": "", "refused": false, and set "tool_name" and "tool_input".',
        "The tool result will be added to the SOURCES with a number, and you will be asked again.",
        "Cite a tool result by its number, like any other source. Use the calculator for any arithmetic.",
    ]
    return "\n".join(lines)


def build_system_prompt(agent: AgentDefinition) -> str:
    """Fill the system template with this agent's settings."""
    template = Template(SYSTEM_TEMPLATE_PATH.read_text(encoding="utf-8"))
    return template.substitute(
        system_instructions=agent.system_instructions.strip(),
        refusal_topics=_join_topics(agent.scope.refusal_topics),
        in_scope_topics=_join_topics(agent.scope.in_scope_topics),
        max_answer_words=agent.output_contract.max_answer_words,
        tools_section=build_tools_section(agent.tools),
    ).strip()


def build_planner_prompt(question: str, sources: list[PromptSource], tool_names: list[str]) -> str:
    """Prompt for the short 'which tool?' step (one tool name from a fixed list)."""
    tools = "\n".join(f"- {n}: {TOOLS[n].description} tool_input: {TOOLS[n].input_hint}." for n in tool_names)
    rules = Template(PLANNER_TEMPLATE_PATH.read_text(encoding="utf-8")).substitute(tools=tools)
    return rules.strip() + "\n\n" + build_user_prompt(question, sources)


def planner_schema(tool_names: list[str]) -> dict:
    """JSON schema that only allows the given tool names or 'none'."""
    return {
        "type": "object",
        "properties": {
            "tool_name": {"type": "string", "enum": [*tool_names, "none"]},
            "tool_input": {"type": "string"},
        },
        "required": ["tool_name", "tool_input"],
    }


def build_user_prompt(question: str, sources: list[PromptSource]) -> str:
    """
    Number each source [1], [2], ... and put the question last.
    Short numbers are far easier for a small model to cite than long IDs;
    the runner maps the numbers back to source IDs afterwards.
    """
    lines = ["SOURCES:"]
    if not sources:
        lines.append("(none)")
    for number, source in enumerate(sources, start=1):
        lines.append(f"[{number}] ({source.header})")
        lines.append(source.text)
        lines.append("")
    lines.append(f"QUESTION: {question.strip()}")
    return "\n".join(lines)


def build_retry_note(reason: str) -> str:
    """Text added to the user prompt when the previous reply was unusable."""
    return (
        "\n\nYour previous reply could not be used (" + reason + "). "
        "Reply again with ONLY the JSON object, using the exact fields: "
        "answer, sources, refused, refusal_reason, tool_name, tool_input."
    )
