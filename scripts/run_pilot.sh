#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Running Phase 1 JEPA Pilot Training..."
.venv/bin/python src/train_phase1.py --config configs/phase1.yaml --pilot
