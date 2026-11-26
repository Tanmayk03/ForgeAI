"""
registry.py - the list of tools ForgeAI can attach to an agent, and run_tool().

The Builder (later) chooses tools from TOOLS by their descriptions; agent.json
lists the chosen names; the runner may only run tools that agent.json allows.
"""

from pydantic import BaseModel

from forgeai.tools.base import Tool, ToolContext, ToolError
from forgeai.tools.calculator import calculator
from forgeai.tools.csv_lookup import csv_lookup
from forgeai.tools.current_date import current_date
from forgeai.tools.weather import weather_api

TOOLS: dict[str, Tool] = {
    t.name: t for t in [
        Tool("csv_lookup", "Finds rows in the agent's CSV data files (e.g. a catalogue or price list).",
             "search words, e.g. a product name or code", csv_lookup),
        Tool("calculator", "Does exact arithmetic. Use it instead of calculating yourself.",
             "an arithmetic expression, e.g. 24 * 2 + 5", calculator),
        Tool("weather_api", "Gets current weather and a 3-day forecast for a place (open-meteo.com).",
             "a place name, e.g. Mumbai", weather_api),
        Tool("current_date", "Gives today's date, weekday and local time.",
             "nothing (leave empty)", current_date),
    ]
}
TOOL_NAMES = set(TOOLS)


class ToolResult(BaseModel):
    name: str
    tool_input: str
    ok: bool
    output: str


def run_tool(name: str, tool_input: str, context: ToolContext, allowed: list[str]) -> ToolResult:
    """Run one tool. Never raises: problems become a ToolResult with ok=False."""
    if name not in TOOLS:
        return ToolResult(name=name, tool_input=tool_input, ok=False, output=f"ERROR: unknown tool '{name}'")
    if name not in allowed:
        return ToolResult(name=name, tool_input=tool_input, ok=False,
                          output=f"ERROR: tool '{name}' is not available to this agent")
    try:
        return ToolResult(name=name, tool_input=tool_input, ok=True, output=TOOLS[name].func(tool_input, context))
    except ToolError as exc:
        return ToolResult(name=name, tool_input=tool_input, ok=False, output=f"ERROR: {exc}")
