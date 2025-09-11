import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from arc_solver.grid import to_array
from arc_solver.neural.episodic import AnalogicalReasoner, EpisodicRetrieval


def test_find_structural_analogies():
    memory = EpisodicRetrieval()
    train_pairs1 = [(to_array([[1]]), to_array([[2]]))]
    memory.add_successful_solution(train_pairs1, [[("recolor", {"mapping": {1: 2}})]])
    reasoner = AnalogicalReasoner()
    train_pairs2 = [(to_array([[3]]), to_array([[4]]))]
    analogies = reasoner.find_structural_analogies(train_pairs2, memory, threshold=0.0)
    assert analogies and analogies[0][0].programs


def test_map_solution_structure_recolor():
    reasoner = AnalogicalReasoner()
    source_program = [("recolor", {"mapping": {1: 2}})]
    target_task = [(to_array([[3, 3]]), to_array([[4, 4]]))]
    mapped = reasoner.map_solution_structure(source_program, target_task)
    assert mapped[0][1]["mapping"] == {3: 4}


def test_abstract_common_patterns():
    reasoner = AnalogicalReasoner()
    task1 = [(to_array([[1]]), to_array([[2]]))]
    task2 = [(to_array([[1, 1]]), to_array([[2, 2]]))]
    pattern = reasoner.abstract_common_patterns([task1, task2])
    assert pattern.get("num_train_pairs", 0) == 1.0
