#!/usr/bin/env python3
"""Scan all Python files and verify their import statements resolve."""
import ast
from importlib import import_module
from pathlib import Path
import pkgutil
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # ensure local modules are importable

# Gather installed module names for quick check
installed = {m.name for m in pkgutil.iter_modules()}

OPTIONAL = {"sklearn_extra"}
failures = []

for path in ROOT.rglob("*.py"):
    if path.parts[0] == ".git" or path.name.startswith("."):  # skip hidden
        continue
    if path.suffix != ".py" or path.name.endswith(".pyc"):
        continue
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            source = path.read_text(encoding="latin-1")
        except UnicodeDecodeError:
            try:
                with open(path, "rb") as f:
                    source = f.read().decode("utf-8", errors="ignore")
            except Exception as exc:
                failures.append((path, "read", str(exc)))
                continue
    try:
        tree = ast.parse(source)
    except Exception as exc:
        failures.append((path, "parse", str(exc)))
        continue
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                local_path = ROOT / (mod.replace(".", "/") + ".py")
                if local_path.exists() or (ROOT / mod).is_dir():
                    continue
                try:
                    import_module(mod)
                except Exception as exc:
                    if mod in OPTIONAL:
                        continue
                    failures.append((path, alias.name, str(exc)))
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0:
                # skip relative imports within repo
                continue
            mod = (node.module or "").split(".")[0]
            if not mod:
                continue
            local_path = ROOT / (mod.replace(".", "/") + ".py")
            if local_path.exists() or (ROOT / mod).is_dir():
                continue
            try:
                import_module(mod)
            except Exception as exc:
                if mod in OPTIONAL:
                    continue
                failures.append((path, node.module, str(exc)))

if failures:
    print("Failed imports:")
    for path, mod, exc in failures:
        print(f"{path}: {mod} -> {exc}")
    sys.exit(1)
else:
    print("All imports resolved successfully.")
