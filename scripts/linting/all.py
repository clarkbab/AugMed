#!/usr/bin/env python3
"""Run all linting scripts in sequence."""

import argparse
from pathlib import Path
import subprocess
import sys

SCRIPTS_DIR = Path(__file__).parent
SCRIPTS = ['fix_imports.py', 'sort_kwargs.py', 'sort_methods.py']

def main() -> None:
    parser = argparse.ArgumentParser(description='Run all linting scripts.')
    parser.add_argument('paths', help='Files or directories to process', nargs='*')
    parser.add_argument('--fix', action='store_true', help='Apply fixes in-place')
    args = parser.parse_args()

    failed = []
    for script in SCRIPTS:
        script_path = SCRIPTS_DIR / script
        if not script_path.exists():
            print(f'Warning: {script} not found, skipping.')
            continue

        cmd = [sys.executable, str(script_path)]
        if args.paths:
            cmd.extend(args.paths)
        if args.fix:
            cmd.append('--fix')

        print(f'{"=" * 60}')
        print(f'Running {script}...')
        print(f'{"=" * 60}')
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failed.append(script)
        print()

    if failed:
        print(f'Failed: {", ".join(failed)}')
        sys.exit(1)
    else:
        print('All linting checks passed.')

if __name__ == '__main__':
    main()
