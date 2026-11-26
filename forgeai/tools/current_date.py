"""current_date.py - today's date, weekday and local time (input is ignored)."""

from forgeai.tools.base import ToolContext


def current_date(tool_input: str, context: ToolContext) -> str:
    now = context.now()
    return f"Today is {now:%A}, {now:%Y-%m-%d}. Local time is {now:%H:%M}."
