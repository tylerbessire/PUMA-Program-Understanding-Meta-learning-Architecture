import numpy as np
from arc_solver.grid import to_array
from arc_solver.neural.guidance import NeuralGuidance


def test_guidance_training_from_tasks():
    inp = to_array([[1, 0], [0, 0]])
    out = np.rot90(inp, -1)
    guidance = NeuralGuidance()
    guidance.train_from_task_pairs([[(inp, out)]], epochs=20)
    pred = guidance.predict_operations([(inp, out)])
    assert "rotate" in pred
