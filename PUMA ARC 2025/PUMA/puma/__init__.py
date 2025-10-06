"""
PUMA package alias

This allows imports like 'from puma.rft import ...' to work
by redirecting to the actual location in arc_solver.rft_engine.rft
"""

import sys
from pathlib import Path

# Add arc_solver to path if needed
arc_solver_path = Path(__file__).parent.parent / "arc_solver"
if str(arc_solver_path) not in sys.path:
    sys.path.insert(0, str(arc_solver_path))

# Import and re-export rft module components
from rft_engine.rft import feature_flags, orchestrator, tracking, explain
from rft_engine.rft.demo import grid as demo_grid

# Make submodules available
sys.modules['puma.rft'] = sys.modules['rft_engine.rft']
sys.modules['puma.rft.feature_flags'] = feature_flags
sys.modules['puma.rft.orchestrator'] = orchestrator
sys.modules['puma.rft.tracking'] = tracking
sys.modules['puma.rft.explain'] = explain
sys.modules['puma.rft.demo'] = sys.modules['rft_engine.rft.demo']
sys.modules['puma.rft.demo.grid'] = demo_grid

__all__ = []