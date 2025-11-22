"""
Atomspace Core Integration

Manages Atomspace initialization, persistence, and schema definitions.
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class AtomType(Enum):
    """Atom types for cognitive architecture"""
    EPISODIC_MEMORY = "EpisodicMemoryNode"
    CONCEPT = "ConceptNode"
    SELF_MODEL = "SelfModelNode"
    GOAL = "GoalNode"
    RELATIONAL_FRAME = "RelationalFrameNode"
    CODE = "CodeNode"
    PERCEPTION = "PerceptionNode"
    EMOTIONAL_STATE = "EmotionalStateNode"


@dataclass
class Atom:
    """Base atom structure"""
    id: str
    type: AtomType
    content: Any
    timestamp: datetime
    truth_value: float = 1.0
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.type.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'truth_value': self.truth_value,
            'confidence': self.confidence
        }


@dataclass
class Link:
    """Link between atoms"""
    source_id: str
    target_id: str
    link_type: str
    strength: float = 1.0

    def to_dict(self) -> Dict:
        return asdict(self)


class Atomspace:
    """
    Atomspace implementation with persistence.

    This is a simplified Atomspace until Hyperon integration is complete.
    Will be replaced with actual Hyperon Atomspace bindings.
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.atoms: Dict[str, Atom] = {}
        self.links: List[Link] = []
        self.persistence_path = persistence_path
        self._atom_counter = 0

        if persistence_path and persistence_path.exists():
            self.load()

    def add_atom(self, atom: Atom) -> str:
        """Add atom to atomspace"""
        if not atom.id:
            atom.id = self._generate_atom_id()
        self.atoms[atom.id] = atom
        return atom.id

    def add_link(self, link: Link):
        """Add link between atoms"""
        self.links.append(link)

    def get_atom(self, atom_id: str) -> Optional[Atom]:
        """Retrieve atom by ID"""
        return self.atoms.get(atom_id)

    def query_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Query atoms by type"""
        return [atom for atom in self.atoms.values() if atom.type == atom_type]

    def get_linked_atoms(self, atom_id: str, link_type: Optional[str] = None) -> List[Atom]:
        """Get atoms linked to given atom"""
        linked_ids = []
        for link in self.links:
            if link.source_id == atom_id:
                if link_type is None or link.link_type == link_type:
                    linked_ids.append(link.target_id)

        return [self.atoms[aid] for aid in linked_ids if aid in self.atoms]

    def save(self):
        """Save atomspace to disk"""
        if not self.persistence_path:
            return

        self.persistence_path.mkdir(parents=True, exist_ok=True)

        # Save atoms
        atoms_data = {aid: atom.to_dict() for aid, atom in self.atoms.items()}
        with open(self.persistence_path / 'atoms.json', 'w') as f:
            json.dump(atoms_data, f, indent=2)

        # Save links
        links_data = [link.to_dict() for link in self.links]
        with open(self.persistence_path / 'links.json', 'w') as f:
            json.dump(links_data, f, indent=2)

    def load(self):
        """Load atomspace from disk"""
        if not self.persistence_path:
            return

        # Load atoms
        atoms_file = self.persistence_path / 'atoms.json'
        if atoms_file.exists():
            with open(atoms_file, 'r') as f:
                atoms_data = json.load(f)
                for aid, atom_dict in atoms_data.items():
                    atom = Atom(
                        id=atom_dict['id'],
                        type=AtomType(atom_dict['type']),
                        content=atom_dict['content'],
                        timestamp=datetime.fromisoformat(atom_dict['timestamp']),
                        truth_value=atom_dict['truth_value'],
                        confidence=atom_dict['confidence']
                    )
                    self.atoms[aid] = atom

        # Load links
        links_file = self.persistence_path / 'links.json'
        if links_file.exists():
            with open(links_file, 'r') as f:
                links_data = json.load(f)
                self.links = [Link(**link_dict) for link_dict in links_data]

    def create_snapshot(self) -> str:
        """Create versioned snapshot"""
        if not self.persistence_path:
            return ""

        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        snapshot_dir = self.persistence_path / 'snapshots' / timestamp
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Save current state as snapshot
        atoms_data = {aid: atom.to_dict() for aid, atom in self.atoms.items()}
        with open(snapshot_dir / 'atoms.json', 'w') as f:
            json.dump(atoms_data, f, indent=2)

        links_data = [link.to_dict() for link in self.links]
        with open(snapshot_dir / 'links.json', 'w') as f:
            json.dump(links_data, f, indent=2)

        return timestamp

    def restore_snapshot(self, snapshot_id: str):
        """Restore from snapshot"""
        if not self.persistence_path:
            return

        snapshot_dir = self.persistence_path / 'snapshots' / snapshot_id
        if not snapshot_dir.exists():
            raise ValueError(f"Snapshot {snapshot_id} not found")

        # Clear current state
        self.atoms.clear()
        self.links.clear()

        # Load snapshot
        with open(snapshot_dir / 'atoms.json', 'r') as f:
            atoms_data = json.load(f)
            for aid, atom_dict in atoms_data.items():
                atom = Atom(
                    id=atom_dict['id'],
                    type=AtomType(atom_dict['type']),
                    content=atom_dict['content'],
                    timestamp=datetime.fromisoformat(atom_dict['timestamp']),
                    truth_value=atom_dict['truth_value'],
                    confidence=atom_dict['confidence']
                )
                self.atoms[aid] = atom

        with open(snapshot_dir / 'links.json', 'r') as f:
            links_data = json.load(f)
            self.links = [Link(**link_dict) for link_dict in links_data]

    def _generate_atom_id(self) -> str:
        """Generate unique atom ID"""
        self._atom_counter += 1
        return f"atom_{self._atom_counter}_{datetime.now(timezone.utc).timestamp()}"

    def count_atoms(self) -> int:
        """Count total atoms"""
        return len(self.atoms)

    def count_concepts(self) -> int:
        """Count concept nodes"""
        return len([a for a in self.atoms.values() if a.type == AtomType.CONCEPT])


class PersistenceManager:
    """
    Manages incremental saves and transaction logging.
    """

    def __init__(self, atomspace: Atomspace):
        self.atomspace = atomspace
        self.transaction_log: List[Dict] = []

    def log_transaction(self, operation: str, data: Dict):
        """Log transaction for recovery"""
        self.transaction_log.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'operation': operation,
            'data': data
        })

    def incremental_save(self):
        """Perform incremental save"""
        self.atomspace.save()
        self._save_transaction_log()

    def _save_transaction_log(self):
        """Save transaction log"""
        if not self.atomspace.persistence_path:
            return

        log_file = self.atomspace.persistence_path / 'transaction_log.json'
        with open(log_file, 'w') as f:
            json.dump(self.transaction_log, f, indent=2)


def bootstrap_atomspace(persistence_path: Optional[Path] = None) -> Atomspace:
    """
    Bootstrap fresh atomspace with structural schema only.
    NO HARDCODED CONTENT - only creates capacity for experience.
    """
    atomspace = Atomspace(persistence_path)

    # Create self-reference node (empty self-model)
    self_model = Atom(
        id="self_model_root",
        type=AtomType.SELF_MODEL,
        content={
            'birth_time': datetime.now(timezone.utc).isoformat(),
            'capabilities': ['perceive', 'act', 'remember', 'learn'],
            'identity_narrative': None  # Emerges from experience
        },
        timestamp=datetime.now(timezone.utc)
    )
    atomspace.add_atom(self_model)

    # Initialize time system (empty timeline)
    timeline_root = Atom(
        id="timeline_root",
        type=AtomType.EPISODIC_MEMORY,
        content={
            'type': 'timeline_root',
            'episodes': []
        },
        timestamp=datetime.now(timezone.utc)
    )
    atomspace.add_atom(timeline_root)

    return atomspace
