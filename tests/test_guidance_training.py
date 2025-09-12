import numpy as np
from arc_solver.grid import to_array
from arc_solver.neural.guidance import NeuralGuidance
from arc_solver.neural.episodic import EpisodeDatabase
from arc_solver.features import compute_task_signature


def test_guidance_training_from_episodes(tmp_path):
    db_path = tmp_path / "episodes.json"
    db = EpisodeDatabase(str(db_path))
    inp = to_array([[1, 0], [0, 0]])
    out = np.rot90(inp, -1)
    sig = compute_task_signature([(inp, out)])
    db.store_episode(sig, [[("rotate", {"k": 1})]], "task", [(inp, out)])
    db.save()

    guidance = NeuralGuidance()
    guidance.train_from_episode_db(str(db_path), epochs=20, lr=0.1)

    pred = guidance.predict_operations([(inp, out)])
    assert "rotate" in pred
