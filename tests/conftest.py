"""pipeline/ runs as plain scripts, not an installable package -- its own
modules import each other assuming the script's own directory is on
sys.path (see CLAUDE.md's "Pipeline architecture" note). Tests import those
same modules directly, so this puts pipeline/ on sys.path once, for every
test in this directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))
