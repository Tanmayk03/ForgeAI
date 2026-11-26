"""
base.py - shared pieces for all tools.

Every tool is a plain function:  tool(tool_input: str, context: ToolContext) -> str
A single text input keeps tool use simple enough for a small model.
A tool raises ToolError for any expected problem; the registry turns that
into a failed ToolResult instead of crashing the agent.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import requests


class ToolError(Exception):
    """An expected tool failure (bad input, no data, network down...)."""


@dataclass
class ToolContext:
    """What a tool may need from the agent. Tests replace now/http_get with fakes."""

    csv_paths: list[Path] = field(default_factory=list)
    now: Callable[[], datetime] = datetime.now
    http_get: Callable[..., requests.Response] = requests.get


@dataclass(frozen=True)
class Tool:
    name: str
    description: str   # shown to the model: what the tool does
    input_hint: str    # shown to the model: what to put in tool_input
    func: Callable[[str, ToolContext], str]
