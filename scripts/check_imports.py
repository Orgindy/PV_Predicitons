"""Verify imports for packages listed in requirements.txt."""

from importlib import import_module
from pathlib import Path
import sys
from packaging.requirements import Requirement
from packaging.version import Version
try:
    from importlib.metadata import (
        version as get_version,
        PackageNotFoundError,
    )
except ImportError:  # pragma: no cover - fallback for older Python
    from importlib_metadata import (
        version as get_version,
        PackageNotFoundError,
    )

req_file = Path(__file__).resolve().parent.parent / "requirements.txt"
IMPORT_MAPPING = {
    "scikit-learn": "sklearn",
    "scikit-learn-extra": "sklearn_extra",
    "pyyaml": "yaml",
}
OPTIONAL = {"scikit-learn-extra"}

failures = []
missing_optional = []
version_issues = []

for line in req_file.read_text().splitlines():
    pkg = line.strip()
    if not pkg or pkg.startswith("#"):
        continue
    req = Requirement(pkg)
    name = req.name
    mod_name = IMPORT_MAPPING.get(name, name.replace("-", "_"))
    try:
        import_module(mod_name)
    except Exception as exc:
        if name in OPTIONAL:
            missing_optional.append((mod_name, exc))
        else:
            failures.append((mod_name, exc))
        continue
    try:
        installed_ver = get_version(name)
    except PackageNotFoundError as exc:
        failures.append((mod_name, exc))
        continue
    if req.specifier and not req.specifier.contains(
        Version(installed_ver), prereleases=True
    ):
        version_issues.append((name, installed_ver, str(req.specifier)))

if missing_optional:
    print("Optional packages missing or failed to import:")
    for mod_name, exc in missing_optional:
        print(f"  {mod_name}: {exc}")

if version_issues:
    print("Version mismatches:")
    for name, inst, spec in version_issues:
        print(f"  {name} {inst} does not satisfy {spec}")

if failures or version_issues:
    for mod_name, exc in failures:
        print(f"{mod_name}: {exc}")
    for name, inst, spec in version_issues:
        print(f"{name} {inst} does not satisfy {spec}")
    sys.exit(1)
else:
    print("All required dependencies imported successfully.")
