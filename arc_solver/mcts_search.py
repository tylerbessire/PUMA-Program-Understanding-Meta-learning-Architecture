# [S:ALG v1] strategy=mcts_search pass
import logging
import math
import random
from typing import List, Tuple, Dict, Any, Optional
from .grid import Array
from .dsl import OPS
from .heuristics import score_candidate
from .neural.sketches import generate_parameter_grid

logger = logging.getLogger(__name__)


class Node:
    def __init__(self, program: List[Tuple[str, Dict[str, Any]]], parent: Optional['Node'] = None, depth: int = 0, max_depth: int = 2):
        self.program = program
        self.parent = parent
        self.children: List['Node'] = []
        self.visits = 0
        self.value = 0.0
        self.untried = []
        if depth < max_depth:
            for op_name in OPS.keys():
                for params in generate_parameter_grid(op_name):
                    self.untried.append((op_name, params))

    def ucb(self, total_visits: int, c: float = 1.4) -> float:
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + c * math.sqrt(math.log(total_visits) / self.visits)


def mcts_search(
    train_pairs: List[Tuple[Array, Array]],
    iterations: int = 100,
    max_depth: int = 2,
    seed: Optional[int] = None,
) -> List[List[Tuple[str, Dict[str, Any]]]]:
    """Monte Carlo Tree Search for program synthesis."""
    rng = random.Random(seed)
    root = Node([], depth=0, max_depth=max_depth)
    for _ in range(iterations):
        node = root
        depth = 0
        # Selection
        while not node.untried and node.children and depth < max_depth:
            total = sum(child.visits for child in node.children)
            node = max(node.children, key=lambda n: n.ucb(total))
            depth += 1
        # Expansion
        if node.untried and depth < max_depth:
            op_name, params = node.untried.pop()
            new_prog = node.program + [(op_name, params)]
            child = Node(new_prog, parent=node, depth=depth + 1, max_depth=max_depth)
            node.children.append(child)
            node = child
        # Simulation
        try:
            reward = score_candidate(node.program, train_pairs)
        except Exception:
            reward = 0.0
        # Backpropagation
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
    best = max(root.children, key=lambda n: n.value / n.visits if n.visits else 0, default=None)
    programs: List[List[Tuple[str, Dict[str, Any]]]] = []
    if best and score_candidate(best.program, train_pairs) >= 0.999:
        programs.append(best.program)
    logger.info("mcts_search complete", extra={"iterations": iterations, "solutions": len(programs)})
    return programs
