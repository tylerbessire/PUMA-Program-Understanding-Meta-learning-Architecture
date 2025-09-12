import numpy as np
from arc_solver.grid import to_array
from arc_solver.neural.episodic import EpisodeDatabase
from arc_solver.enhanced_search import EnhancedSearch


def test_episodic_storage_and_retrieval(tmp_path):
    db_path = tmp_path / "episodes.json"
    search = EnhancedSearch(episode_db_path=str(db_path))
    inp = to_array([[1, 0], [0, 0]])
    out = np.rot90(inp, -1)
    search.episodic_retrieval.add_successful_solution([(inp, out)], [[("rotate", {"k": 1})]])
    search.episodic_retrieval.save()
    db = EpisodeDatabase(str(db_path))
    db.load()
    assert db.episodes
    retrieved = search.episodic_retrieval.query_for_programs([(inp, out)])
    assert retrieved
