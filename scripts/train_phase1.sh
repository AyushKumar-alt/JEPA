#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Launching Phase 1 JEPA Full Pretraining..."
.venv/bin/python src/train_phase1.py --config configs/phase1.yaml
