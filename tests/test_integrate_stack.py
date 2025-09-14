import sys
from pathlib import Path
import numpy as np

repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "tools"))

from arc_solver.grid import to_array
from arc_solver.neural.guidance import NeuralGuidance
from integrate_stack import evaluate_search_reduction


def test_guidance_reduces_node_expansions():
    inp = to_array([[1, 0, 0], [1, 1, 0], [0, 0, 0]])
    out = np.rot90(inp, -1)
    task = [(inp, out)]

    guidance = NeuralGuidance()
    guidance.train_from_task_pairs([task], epochs=40, lr=0.1)

    reduction, base_nodes, guided_nodes = evaluate_search_reduction(task, guidance)
    assert base_nodes > 0
    assert reduction >= 0.3
    assert guided_nodes < base_nodes
