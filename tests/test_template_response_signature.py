"""Every TemplateResponse call must pass the request first.

Current Starlette only accepts TemplateResponse(request, name, context). The
old TemplateResponse(name, {"request": request}) form fails at render time
with TypeError: unhashable type: 'dict', so a page using it returns 500. This
check is static, so it also covers pages the other tests never render.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "gridarena"


def _old_style_calls():
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "TemplateResponse"):
                continue
            first = node.args[0] if node.args else None
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                yield f"{path.relative_to(PACKAGE.parent)}:{node.lineno}"


def test_template_response_passes_request_first():
    offenders = list(_old_style_calls())
    assert not offenders, "TemplateResponse(name, ...) must be TemplateResponse(request, name, ...): " + ", ".join(offenders)
