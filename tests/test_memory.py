"""
Tests for episodic memory component.

This module tests the episodic retrieval system that stores and retrieves
successful program solutions for analogical reasoning.
"""

import pytest
import numpy as np
from typing import List, Tuple
import tempfile
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from arc_solver.grid import to_array
from arc_solver.neural.episodic import EpisodicRetrieval, EpisodeDatabase, Episode
from arc_solver.features import compute_task_signature


class TestEpisodicMemory:
    """Test suite for episodic memory components."""
    
    def create_test_pairs(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create test training pairs."""
        return [
            (to_array([[1, 0], [0, 1]]), to_array([[0, 1], [1, 0]])),
            (to_array([[2, 0], [0, 2]]), to_array([[0, 2], [2, 0]]))
        ]
    
    def create_similar_pairs(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create similar but different training pairs."""
        return [
            (to_array([[3, 0], [0, 3]]), to_array([[0, 3], [3, 0]])),
            (to_array([[1, 1], [1, 1]]), to_array([[1, 1], [1, 1]]))
        ]
    
    def create_test_programs(self) -> List[List[Tuple[str, dict]]]:
        """Create test programs."""
        return [
            [("flip", {"axis": 0})],
            [("rotate", {"k": 2})]
        ]
    
    def test_episode_creation(self):
        """Test Episode object creation."""
        train_pairs = self.create_test_pairs()
        programs = self.create_test_programs()
        
        episode = Episode(
            task_signature="test_sig",
            programs=programs,
            task_id="test_task",
            train_pairs=train_pairs
        )
        
        assert episode.task_signature == "test_sig"
        assert episode.programs == programs
        assert episode.task_id == "test_task"
        assert len(episode.train_pairs) == 2
        assert episode.success_count == 1
        assert episode.metadata == {}
    
    def test_episode_database_creation(self):
        """Test EpisodeDatabase initialization."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            db_path = f.name
        
        try:
            db = EpisodeDatabase(db_path)
            assert db.db_path == db_path
            assert len(db.episodes) == 0
            assert len(db.signature_index) == 0
            assert len(db.program_index) == 0
        finally:
            os.unlink(db_path)
    
    def test_episode_storage_and_retrieval(self):
        """Test storing and retrieving episodes."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            db_path = f.name
        
        try:
            db = EpisodeDatabase(db_path)
            train_pairs = self.create_test_pairs()
            programs = self.create_test_programs()
            
            # Store episode
            episode_id = db.store_episode(
                task_signature="flip_task",
                programs=programs,
                task_id="task_001",
                train_pairs=train_pairs
            )
            
            assert episode_id is not None
            assert len(db.episodes) == 1
            assert "flip_task" in db.signature_index
            
            # Retrieve episode
            episode = db.get_episode(episode_id)
            assert episode is not None
            assert episode.task_signature == "flip_task"
            assert episode.programs == programs
            assert episode.task_id == "task_001"
            
        finally:
            os.unlink(db_path)
    
    def test_signature_based_retrieval(self):
        """Test retrieval by task signature."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            db_path = f.name
        
        try:
            db = EpisodeDatabase(db_path)
            train_pairs = self.create_test_pairs()
            programs = self.create_test_programs()
            
            # Store multiple episodes with same signature
            db.store_episode("flip_task", programs[:1], "task_001", train_pairs)
            db.store_episode("flip_task", programs[1:], "task_002", train_pairs)
            db.store_episode("different_task", programs, "task_003", train_pairs)
            
            # Query by signature
            episodes = db.query_by_signature("flip_task")
            assert len(episodes) == 2
            
            episodes = db.query_by_signature("different_task")
            assert len(episodes) == 1
            
            episodes = db.query_by_signature("nonexistent")
            assert len(episodes) == 0
            
        finally:
            os.unlink(db_path)
    
    def test_similarity_based_retrieval(self):
        """Test retrieval by feature similarity."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            db_path = f.name
        
        try:
            db = EpisodeDatabase(db_path)
            
            # Store episodes with different characteristics
            pairs1 = self.create_test_pairs()
            pairs2 = self.create_similar_pairs()
            programs = self.create_test_programs()
            
            db.store_episode("task_1", programs, "task_001", pairs1)
            db.store_episode("task_2", programs, "task_002", pairs2)
            
            # Query with similar pairs
            similar_episodes = db.query_by_similarity(pairs1, similarity_threshold=0.5, max_results=5)
            
            assert isinstance(similar_episodes, list)
            assert len(similar_episodes) > 0
            
            # Each result should have episode and similarity score
            for episode, similarity in similar_episodes:
                assert isinstance(episode, Episode)
                assert isinstance(similarity, (int, float))
                assert 0 <= similarity <= 1
                
        finally:
            os.unlink(db_path)
    
    def test_episodic_retrieval_interface(self):
        """Test main EpisodicRetrieval interface."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            db_path = f.name
        
        try:
            retrieval = EpisodicRetrieval(db_path)
            train_pairs = self.create_test_pairs()
            programs = self.create_test_programs()
            
            # Add successful solution
            retrieval.add_successful_solution(train_pairs, programs, "task_001")
            
            # Query for programs
            candidates = retrieval.query_for_programs(train_pairs)
            
            assert isinstance(candidates, list)
            assert len(candidates) > 0
            
            # Each candidate should be a valid program
            for program in candidates:
                assert isinstance(program, list)
                assert all(isinstance(op, tuple) and len(op) == 2 for op in program)
                
        finally:
            os.unlink(db_path)
    
    def test_database_persistence(self):
        """Test that database persists to file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            db_path = f.name
        
        try:
            # Create and populate database
            db1 = EpisodeDatabase(db_path)
            train_pairs = self.create_test_pairs()
            programs = self.create_test_programs()
            
            episode_id = db1.store_episode("test_task", programs, "task_001", train_pairs)
            db1.save()
            
            # Load database from file
            db2 = EpisodeDatabase(db_path)
            db2.load()
            
            # Should have same content
            assert len(db2.episodes) == 1
            assert episode_id in db2.episodes
            
            episode = db2.get_episode(episode_id)
            assert episode.task_signature == "test_task"
            assert episode.programs == programs
            
        finally:
            os.unlink(db_path)
    
    def test_statistics(self):
        """Test database statistics."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            db_path = f.name
        
        try:
            retrieval = EpisodicRetrieval(db_path)
            train_pairs = self.create_test_pairs()
            programs = self.create_test_programs()
            
            # Add multiple solutions
            for i in range(5):
                retrieval.add_successful_solution(train_pairs, programs, f"task_{i:03d}")
            
            stats = retrieval.get_stats()
            
            assert isinstance(stats, dict)
            assert 'total_episodes' in stats
            assert 'unique_signatures' in stats
            assert 'total_programs' in stats
            assert 'average_programs_per_episode' in stats
            
            assert stats['total_episodes'] == 5
            assert stats['total_programs'] >= 5  # At least one program per episode
            
        finally:
            os.unlink(db_path)
    
    def test_feature_computation(self):
        """Test task signature computation."""
        train_pairs1 = self.create_test_pairs()
        train_pairs2 = self.create_similar_pairs()
        
        sig1 = compute_task_signature(train_pairs1)
        sig2 = compute_task_signature(train_pairs2)
        
        assert isinstance(sig1, str)
        assert isinstance(sig2, str)
        assert len(sig1) > 0
        assert len(sig2) > 0
        
        # Same pairs should produce same signature
        sig1_repeat = compute_task_signature(train_pairs1)
        assert sig1 == sig1_repeat
    
    def test_empty_database_queries(self):
        """Test queries on empty database."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            db_path = f.name
        
        try:
            retrieval = EpisodicRetrieval(db_path)
            train_pairs = self.create_test_pairs()
            
            # Query empty database
            candidates = retrieval.query_for_programs(train_pairs)
            
            assert isinstance(candidates, list)
            assert len(candidates) == 0
            
            stats = retrieval.get_stats()
            assert stats['total_episodes'] == 0
            
        finally:
            os.unlink(db_path)
    
    def test_large_database_performance(self):
        """Test performance with larger number of episodes."""
        import time
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            db_path = f.name
        
        try:
            retrieval = EpisodicRetrieval(db_path)
            train_pairs = self.create_test_pairs()
            programs = self.create_test_programs()
            
            # Add many episodes
            num_episodes = 50
            start_time = time.time()
            
            for i in range(num_episodes):
                retrieval.add_successful_solution(train_pairs, programs, f"task_{i:03d}")
            
            storage_time = time.time() - start_time
            
            # Query performance
            start_time = time.time()
            candidates = retrieval.query_for_programs(train_pairs)
            query_time = time.time() - start_time
            
            # Should complete reasonably quickly
            assert storage_time < 10.0, f"Storage took {storage_time:.3f}s"
            assert query_time < 2.0, f"Query took {query_time:.3f}s"
            
            # Should find candidates
            assert len(candidates) > 0
            
        finally:
            os.unlink(db_path)
    
    def test_duplicate_handling(self):
        """Test handling of duplicate episodes."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            db_path = f.name
        
        try:
            retrieval = EpisodicRetrieval(db_path)
            train_pairs = self.create_test_pairs()
            programs = self.create_test_programs()
            
            # Add same solution multiple times
            retrieval.add_successful_solution(train_pairs, programs, "task_001")
            retrieval.add_successful_solution(train_pairs, programs, "task_001")
            retrieval.add_successful_solution(train_pairs, programs, "task_002")
            
            stats = retrieval.get_stats()
            
            # Should handle duplicates appropriately
            # (Implementation may vary - could dedupe or allow duplicates)
            assert stats['total_episodes'] >= 1
            
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
