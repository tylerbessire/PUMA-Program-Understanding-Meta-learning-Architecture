"""
Episodic retrieval system for ARC solver.

This module implements a database of previously solved programs that can be
retrieved based on task similarity. It uses task signatures to find analogous
tasks and reuse or adapt their solutions.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import json
import os
from collections import defaultdict

from .grid import Array
from .features import compute_task_signature, extract_task_features


class EpisodeDatabase:
    """Database of solved ARC tasks with episodic retrieval capability."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.episodes: List[Dict[str, Any]] = []
        self.signature_index: Dict[str, List[int]] = defaultdict(list)
        self.db_path = db_path
        
        if db_path and os.path.exists(db_path):
            self.load_database()
    
    def add_episode(self, train_pairs: List[Tuple[Array, Array]], 
                   successful_programs: List[List[Tuple[str, Dict[str, Any]]]], 
                   task_id: str = ""):
        """Add a solved task to the episode database."""
        signature = compute_task_signature(train_pairs)
        features = extract_task_features(train_pairs)
        
        episode = {
            'task_id': task_id,
            'signature': signature,
            'features': features,
            'train_pairs': [(inp.tolist(), out.tolist()) for inp, out in train_pairs],
            'successful_programs': successful_programs,
            'difficulty': self._estimate_difficulty(features),
            'solution_count': len(successful_programs),
        }
        
        episode_idx = len(self.episodes)
        self.episodes.append(episode)
        self.signature_index[signature].append(episode_idx)
    
    def retrieve_similar_episodes(self, train_pairs: List[Tuple[Array, Array]], 
                                 k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve k most similar episodes to the given task."""
        query_signature = compute_task_signature(train_pairs)
        query_features = extract_task_features(train_pairs)
        
        # First, try exact signature match
        exact_matches = []
        if query_signature in self.signature_index:
            for idx in self.signature_index[query_signature]:
                exact_matches.append(self.episodes[idx])
        
        if exact_matches:
            return exact_matches[:k]
        
        # If no exact matches, compute similarity scores
        similarities = []
        for episode in self.episodes:
            similarity = self._compute_similarity(query_features, episode['features'])
            similarities.append((similarity, episode))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [episode for _, episode in similarities[:k]]
    
    def get_candidate_programs(self, train_pairs: List[Tuple[Array, Array]], 
                              max_programs: int = 10) -> List[List[Tuple[str, Dict[str, Any]]]]:
        """Get candidate programs from similar episodes."""
        similar_episodes = self.retrieve_similar_episodes(train_pairs)
        
        candidate_programs = []
        for episode in similar_episodes:
            for program in episode['successful_programs']:
                candidate_programs.append(program)
                if len(candidate_programs) >= max_programs:
                    break
            if len(candidate_programs) >= max_programs:
                break
        
        return candidate_programs
    
    def _compute_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Compute similarity between two feature vectors."""
        # Simple cosine similarity on numerical features
        numerical_keys = [
            'num_train_pairs', 'input_height_mean', 'input_width_mean',
            'output_height_mean', 'output_width_mean', 'size_ratio_mean',
            'input_colors_mean', 'output_colors_mean', 'color_mapping_size',
            'input_objects_mean', 'output_objects_mean', 'object_count_preserved',
            'likely_rotation', 'likely_reflection', 'likely_translation',
            'likely_recolor', 'likely_crop', 'likely_pad'
        ]
        
        v1 = np.array([features1.get(k, 0) for k in numerical_keys])
        v2 = np.array([features2.get(k, 0) for k in numerical_keys])
        
        # Normalize vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(v1, v2) / (norm1 * norm2)
        
        # Add bonus for boolean feature matches
        boolean_keys = ['shape_preserved', 'background_color_consistent', 'has_color_mapping']
        boolean_matches = sum(1 for k in boolean_keys if features1.get(k, False) == features2.get(k, False))
        boolean_bonus = boolean_matches / len(boolean_keys) * 0.2
        
        return cosine_sim + boolean_bonus
    
    def _estimate_difficulty(self, features: Dict[str, Any]) -> float:
        """Estimate task difficulty based on features."""
        difficulty = 0.0
        
        # More training pairs might indicate harder task
        difficulty += features.get('num_train_pairs', 0) * 0.1
        
        # Larger grids are typically harder
        grid_size = features.get('input_height_mean', 0) * features.get('input_width_mean', 0)
        difficulty += grid_size / 1000.0
        
        # More colors and objects increase difficulty
        difficulty += features.get('input_colors_mean', 0) * 0.1
        difficulty += features.get('input_objects_mean', 0) * 0.05
        
        # Shape changes increase difficulty
        if not features.get('shape_preserved', True):
            difficulty += 0.5
        
        return min(difficulty, 1.0)  # Cap at 1.0
    
    def save_database(self, filepath: Optional[str] = None):
        """Save the episode database to file."""
        save_path = filepath or self.db_path
        if not save_path:
            return
        
        try:
            with open(save_path, 'w') as f:
                json.dump(self.episodes, f, indent=2)
        except Exception as e:
            print(f"Failed to save database: {e}")
    
    def load_database(self, filepath: Optional[str] = None):
        """Load the episode database from file."""
        load_path = filepath or self.db_path
        if not load_path or not os.path.exists(load_path):
            return
        
        try:
            with open(load_path, 'r') as f:
                self.episodes = json.load(f)
            
            # Rebuild signature index
            self.signature_index = defaultdict(list)
            for idx, episode in enumerate(self.episodes):
                signature = episode['signature']
                self.signature_index[signature].append(idx)
                
        except Exception as e:
            print(f"Failed to load database: {e}")
            self.episodes = []
            self.signature_index = defaultdict(list)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.episodes:
            return {}
        
        difficulties = [ep['difficulty'] for ep in self.episodes]
        solution_counts = [ep['solution_count'] for ep in self.episodes]
        
        return {
            'total_episodes': len(self.episodes),
            'unique_signatures': len(self.signature_index),
            'avg_difficulty': np.mean(difficulties),
            'avg_solutions_per_task': np.mean(solution_counts),
            'max_solutions_per_task': max(solution_counts),
        }


class EpisodicRetrieval:
    """Main interface for episodic retrieval in the ARC solver."""
    
    def __init__(self, db_path: str = "episodes.json"):
        self.database = EpisodeDatabase(db_path)
        self.cache = {}  # Simple cache for recent queries
    
    def query_for_programs(self, train_pairs: List[Tuple[Array, Array]], 
                          max_candidates: int = 5) -> List[List[Tuple[str, Dict[str, Any]]]]:
        """Query the database for candidate programs."""
        # Check cache first
        cache_key = compute_task_signature(train_pairs)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        candidates = self.database.get_candidate_programs(train_pairs, max_candidates)
        
        # Cache result
        self.cache[cache_key] = candidates
        if len(self.cache) > 100:  # Simple cache eviction
            self.cache.clear()
        
        return candidates
    
    def add_successful_solution(self, train_pairs: List[Tuple[Array, Array]], 
                               programs: List[List[Tuple[str, Dict[str, Any]]]], 
                               task_id: str = ""):
        """Add a successful solution to the database."""
        self.database.add_episode(train_pairs, programs, task_id)
        
        # Clear cache since database changed
        self.cache.clear()
    
    def save(self):
        """Save the database."""
        self.database.save_database()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        return self.database.get_statistics()
