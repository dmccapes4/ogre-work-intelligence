#!/usr/bin/env bash
# Launch the OGrE work-intelligence simulation in its virtualenv.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
fi

exec ./.venv/bin/python main.py "$@"
