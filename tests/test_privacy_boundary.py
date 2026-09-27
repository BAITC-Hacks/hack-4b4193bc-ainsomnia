"""Запрет зависимости рабочего кода от исследовательского namespace."""
from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = {"synth", "finetune", "embed"}


def imports(path):
    module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
    package = module.rsplit(".", 1)[0]
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level:
                name = resolve_name("." * node.level + name, package)
            yield name
            for alias in node.names:
                yield name + "." + alias.name


def main():
    files = [ROOT / "train.py"] + [
        p for p in (ROOT / "src").rglob("*.py")
        if p.relative_to(ROOT / "src").parts[0] not in RESEARCH
    ]
    violations = [(p.relative_to(ROOT).as_posix(), name)
                  for p in files for name in imports(p)
                  if name == "src.synth" or name.startswith("src.synth.")]
    assert not violations, f"production imports R&D: {violations}"
    print(f"privacy boundary: {len(files)} production files; imports src.synth = 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
