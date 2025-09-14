import numpy as np
from arc_solver.grid import to_array
from arc_solver.neural.guidance import NeuralGuidance
from arc_solver.neural.metrics import top_k_micro_f1
from arc_solver.features import extract_task_features


def test_topk_micro_f1_threshold():
    inp = to_array([[1, 0, 0], [1, 1, 0], [0, 0, 0]])
    out = np.rot90(inp, -1)
    task = [(inp, out)]

    guidance = NeuralGuidance()
    guidance.train_from_task_pairs([task], epochs=40, lr=0.1)

    feat = extract_task_features(task)
    X = guidance.neural_model._features_to_vector(feat)
    probs = guidance.neural_model.forward(X).reshape(1, -1)
    labels = np.zeros_like(probs)
    idx = guidance.neural_model.operations.index("rotate")
    labels[0, idx] = 1.0

    f1 = top_k_micro_f1(probs, labels, k=2)
    assert f1 >= 0.55
