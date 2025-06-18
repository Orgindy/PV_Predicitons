"""Verify imports for packages listed in requirements.txt."""
from importlib import import_module
from pathlib import Path
import sys

req_file = Path(__file__).resolve().parent.parent / 'requirements.txt'
IMPORT_MAPPING = {
    'scikit-learn': 'sklearn',
    'scikit-learn-extra': 'sklearn_extra',
    'pyyaml': 'yaml',
}
OPTIONAL = {'scikit-learn-extra'}

failures = []
missing_optional = []

for line in req_file.read_text().splitlines():
    pkg = line.strip()
    if not pkg or pkg.startswith('#'):
        continue
    base = pkg.split('[')[0]
    for sep in ('==', '>=', '<=', '<', '>', '~='):
        base = base.split(sep)[0]
    name = base
    mod_name = IMPORT_MAPPING.get(name, name.replace('-', '_'))
    try:
        import_module(mod_name)
    except Exception as exc:
        if name in OPTIONAL:
            missing_optional.append((mod_name, exc))
        else:
            failures.append((mod_name, exc))

if missing_optional:
    print('Optional packages missing or failed to import:')
    for mod_name, exc in missing_optional:
        print(f'  {mod_name}: {exc}')

if failures:
    for mod_name, exc in failures:
        print(f'{mod_name}: {exc}')
    sys.exit(1)
else:
    print('All required dependencies imported successfully.')
