"""
LLM Output Caching for PUMA ARC Solver.

Provides deterministic caching of LLM outputs keyed by task signature
and tracking state to:
1. Reduce redundant API calls
2. Ensure deterministic behavior across runs
3. Control costs for repeated queries
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import time

import numpy as np

from .grid import Array


@dataclass
class CacheEntry:
    """A single cache entry."""
    key: str
    value: Any
    timestamp: float
    metadata: Dict[str, Any]
    hit_count: int = 0


class LLMCache:
    """Cache for LLM outputs with task signature + tracking state keys."""

    def __init__(self, cache_dir: str = ".llm_cache", max_entries: int = 1000):
        """Initialize LLM cache.

        Args:
            cache_dir: Directory to store cache files
            max_entries: Maximum cache entries before eviction
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_entries = max_entries
        self.entries: Dict[str, CacheEntry] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load cache index from disk."""
        index_file = self.cache_dir / "cache_index.json"

        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    index_data = json.load(f)

                for key, entry_data in index_data.items():
                    self.entries[key] = CacheEntry(
                        key=entry_data['key'],
                        value=None,  # Lazy load
                        timestamp=entry_data['timestamp'],
                        metadata=entry_data['metadata'],
                        hit_count=entry_data.get('hit_count', 0)
                    )
            except Exception as e:
                print(f"Warning: Failed to load cache index: {e}")

    def _save_index(self) -> None:
        """Save cache index to disk."""
        index_file = self.cache_dir / "cache_index.json"

        index_data = {}
        for key, entry in self.entries.items():
            index_data[key] = {
                'key': entry.key,
                'timestamp': entry.timestamp,
                'metadata': entry.metadata,
                'hit_count': entry.hit_count
            }

        try:
            with open(index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save cache index: {e}")

    def _compute_task_signature(self, train_pairs: list) -> str:
        """Compute deterministic signature from training pairs.

        Args:
            train_pairs: List of (input, output) grid pairs

        Returns:
            Hex digest signature
        """
        hasher = hashlib.sha256()

        for inp, out in train_pairs:
            # Convert to bytes in deterministic way
            hasher.update(np.array(inp, dtype=np.int16).tobytes())
            hasher.update(np.array(out, dtype=np.int16).tobytes())

        return hasher.hexdigest()

    def _compute_state_signature(self, tracking_state: Dict[str, Any]) -> str:
        """Compute deterministic signature from tracking state.

        Args:
            tracking_state: Tracking state dictionary

        Returns:
            Hex digest signature
        """
        hasher = hashlib.sha256()

        # Serialize tracking state deterministically
        state_str = json.dumps(tracking_state, sort_keys=True, default=str)
        hasher.update(state_str.encode('utf-8'))

        return hasher.hexdigest()

    def make_key(
        self,
        prompt_type: str,
        train_pairs: Optional[list] = None,
        tracking_state: Optional[Dict[str, Any]] = None,
        extra_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Make cache key from components.

        Args:
            prompt_type: Type of prompt (e.g., 'inventory_analysis', 'rule_synthesis')
            train_pairs: Training pairs (for task signature)
            tracking_state: Tracking state (for state signature)
            extra_context: Additional context to include in key

        Returns:
            Cache key string
        """
        components = [prompt_type]

        if train_pairs is not None:
            task_sig = self._compute_task_signature(train_pairs)
            components.append(f"task:{task_sig[:16]}")

        if tracking_state is not None:
            state_sig = self._compute_state_signature(tracking_state)
            components.append(f"state:{state_sig[:16]}")

        if extra_context is not None:
            extra_sig = hashlib.sha256(
                json.dumps(extra_context, sort_keys=True, default=str).encode('utf-8')
            ).hexdigest()
            components.append(f"extra:{extra_sig[:8]}")

        return ":".join(components)

    def get(self, key: str) -> Optional[Any]:
        """Get cached value by key.

        Args:
            key: Cache key

        Returns:
            Cached value, or None if not found
        """
        if key not in self.entries:
            return None

        entry = self.entries[key]

        # Lazy load value from disk
        if entry.value is None:
            value_file = self.cache_dir / f"{key}.pkl"

            if not value_file.exists():
                # Entry in index but file missing
                del self.entries[key]
                return None

            try:
                with open(value_file, 'rb') as f:
                    entry.value = pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load cache entry: {e}")
                return None

        # Update hit count
        entry.hit_count += 1

        return entry.value

    def set(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set cache value.

        Args:
            key: Cache key
            value: Value to cache
            metadata: Optional metadata about the entry
        """
        # Evict if at capacity
        if len(self.entries) >= self.max_entries and key not in self.entries:
            self._evict_lru()

        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            metadata=metadata or {},
            hit_count=0
        )

        self.entries[key] = entry

        # Save to disk
        value_file = self.cache_dir / f"{key}.pkl"

        try:
            with open(value_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            print(f"Warning: Failed to save cache entry: {e}")

        # Update index
        self._save_index()

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.entries:
            return

        # Find entry with oldest timestamp and lowest hit count
        lru_key = min(
            self.entries.keys(),
            key=lambda k: (self.entries[k].hit_count, self.entries[k].timestamp)
        )

        # Delete from disk
        value_file = self.cache_dir / f"{lru_key}.pkl"
        if value_file.exists():
            value_file.unlink()

        # Delete from memory
        del self.entries[lru_key]

    def clear(self) -> None:
        """Clear entire cache."""
        # Delete all value files
        for value_file in self.cache_dir.glob("*.pkl"):
            value_file.unlink()

        # Clear index
        self.entries.clear()
        self._save_index()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache stats
        """
        if not self.entries:
            return {
                'total_entries': 0,
                'total_hits': 0,
                'cache_size_mb': 0.0
            }

        total_hits = sum(e.hit_count for e in self.entries.values())

        # Estimate size
        total_size = 0
        for value_file in self.cache_dir.glob("*.pkl"):
            total_size += value_file.stat().st_size

        return {
            'total_entries': len(self.entries),
            'total_hits': total_hits,
            'avg_hits_per_entry': total_hits / len(self.entries),
            'cache_size_mb': total_size / (1024 * 1024),
            'oldest_entry': min(e.timestamp for e in self.entries.values()),
            'newest_entry': max(e.timestamp for e in self.entries.values())
        }

    def export_entries(self, output_file: str) -> None:
        """Export cache entries to JSON for inspection.

        Args:
            output_file: Path to output JSON file
        """
        export_data = []

        for key, entry in self.entries.items():
            export_data.append({
                'key': key,
                'timestamp': entry.timestamp,
                'hit_count': entry.hit_count,
                'metadata': entry.metadata,
                'value_preview': str(entry.value)[:200] if entry.value else None
            })

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)


class CachedLLMInterface:
    """Wrapper around LLMInterface with integrated caching."""

    def __init__(self, llm_interface, cache: Optional[LLMCache] = None):
        """Initialize cached LLM interface.

        Args:
            llm_interface: Base LLM interface
            cache: LLM cache (creates new one if None)
        """
        self.llm = llm_interface
        self.cache = cache or LLMCache()
        self.cache_hits = 0
        self.cache_misses = 0

    def generate_with_cache(
        self,
        system_prompt: str,
        user_prompt: str,
        cache_key: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate text with caching.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            cache_key: Cache key for this query
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            metadata: Metadata to store with cache entry

        Returns:
            Generated text (from cache or LLM)
        """
        # Check cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        # Cache miss - generate
        self.cache_misses += 1
        result = self.llm.generate(system_prompt, user_prompt, temperature, max_tokens)

        # Cache result
        self.cache.set(cache_key, result, metadata)

        return result

    def generate_json_with_cache(
        self,
        system_prompt: str,
        user_prompt: str,
        cache_key: str,
        temperature: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate JSON with caching.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            cache_key: Cache key for this query
            temperature: Sampling temperature
            metadata: Metadata to store with cache entry

        Returns:
            Parsed JSON dict (from cache or LLM)
        """
        # Check cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        # Cache miss - generate
        self.cache_misses += 1
        result = self.llm.generate_json(system_prompt, user_prompt, temperature)

        # Cache result
        self.cache.set(cache_key, result, metadata)

        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics.

        Returns:
            Dict with cache stats and hit/miss rates
        """
        stats = self.cache.get_stats()

        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        stats.update({
            'session_hits': self.cache_hits,
            'session_misses': self.cache_misses,
            'hit_rate': hit_rate
        })

        return stats
