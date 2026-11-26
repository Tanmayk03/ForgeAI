"""
calculator.py - safe arithmetic.

Never uses eval(): the expression is parsed into a syntax tree and only
numbers and + - * / // % ** ( ) are allowed. Anything else (names, function
calls, attribute access) is rejected, so no code can run.
"""

import ast
import operator
import re

from forgeai.tools.base import ToolContext, ToolError

OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.UAdd: operator.pos, ast.USub: operator.neg,
}
MAX_EXPONENT = 100
MAX_LENGTH = 200


def _evaluate(node):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolError("exponent too large")
        return OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ToolError("only numbers and + - * / // % ** ( ) are allowed")


def calculator(tool_input: str, context: ToolContext) -> str:
    """'24 * 2 + 1' -> '49'."""
    expression = re.sub(r"(?<=\d),(?=\d)", "", tool_input.strip())  # 6,000 -> 6000
    if not expression:
        raise ToolError("empty expression")
    if len(expression) > MAX_LENGTH:
        raise ToolError("expression too long")
    try:
        result = _evaluate(ast.parse(expression, mode="eval"))
    except SyntaxError as exc:
        raise ToolError(f"not a valid arithmetic expression: {expression!r}") from exc
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    if isinstance(result, float):
        result = int(result) if result.is_integer() else round(result, 6)
    return f"{expression} = {result}"
