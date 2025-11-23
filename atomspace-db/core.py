"""
Atomspace Core Integration

Manages Atomspace initialization, persistence, and schema definitions.
Integrates native Hyperon Atomspace with backward-compatible JSON storage.
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

# Hyperon imports for native atomspace integration
try:
    from hyperon import MeTTa, AtomKind
    from hyperon.atoms import Atom as HyperonAtom, AtomType as HyperonAtomType
    from hyperon.atoms import E, S, V, OperationAtom
    from hyperon.base import GroundingSpace, Bindings
    HYPERON_AVAILABLE = True
except ImportError:
    HYPERON_AVAILABLE = False
    HyperonAtom = None
    GroundingSpace = None


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


class HyperonAtomspaceAdapter:
    """
    Adapter for native Hyperon Atomspace integration.

    This class wraps Hyperon's MeTTa runtime and provides:
    1. Type mapping between PUMA schema and Hyperon atoms
    2. Bidirectional persistence (JSON + Hyperon native)
    3. Pattern matching using Hyperon's query capabilities
    4. Backward compatibility with existing JSON-based API

    Integration points:
    - PUMA AtomTypes → Hyperon custom node types
    - PUMA Links → Hyperon Expression atoms
    - PUMA persistence → Hyperon GroundingSpace + JSON backup
    """

    def __init__(self, persistence_path: Optional[Path] = None, use_hyperon: bool = True):
        """
        Initialize Hyperon Atomspace adapter.

        Args:
            persistence_path: Path for JSON persistence backup
            use_hyperon: Use native Hyperon if available, fallback to JSON
        """
        self.persistence_path = persistence_path
        self.use_hyperon = use_hyperon and HYPERON_AVAILABLE

        # JSON-based fallback storage (maintains compatibility)
        self.atoms: Dict[str, Atom] = {}
        self.links: List[Link] = []
        self._atom_counter = 0

        # Hyperon native atomspace
        self.metta = None
        self.space = None

        if self.use_hyperon:
            self._init_hyperon()

        # Load existing data
        if persistence_path and persistence_path.exists():
            self.load()

    def _init_hyperon(self):
        """Initialize Hyperon MeTTa runtime and GroundingSpace"""
        if not HYPERON_AVAILABLE:
            return

        try:
            # Create MeTTa runtime for reasoning
            self.metta = MeTTa()
            self.space = self.metta.space()

            # Register PUMA-specific atom type constructors in MeTTa
            self._register_puma_types()
        except Exception as e:
            print(f"Warning: Hyperon initialization failed: {e}. Falling back to JSON.")
            self.use_hyperon = False

    def _register_puma_types(self):
        """
        Register PUMA atom types as MeTTa custom types.

        Type mappings:
        - EpisodicMemoryNode → (EpisodicMemory <content>)
        - ConceptNode → (Concept <name> <properties>)
        - RelationalFrameNode → (RelationalFrame <type> <relations>)
        - CodeNode → MeTTa executable expressions
        - SelfModelNode → (SelfModel <identity> <capabilities>)
        """
        if not self.metta:
            return

        # Define PUMA types in MeTTa space
        type_definitions = """
        ; PUMA Cognitive Architecture Types
        (: EpisodicMemory Type)
        (: Concept Type)
        (: RelationalFrame Type)
        (: SelfModel Type)
        (: Goal Type)
        (: Perception Type)
        (: EmotionalState Type)

        ; Type constructors
        (: make-episodic (-> String Number EpisodicMemory))
        (: make-concept (-> String Expression Concept))
        (: make-relational-frame (-> String Expression RelationalFrame))
        """

        # Load type definitions into MeTTa space
        self.metta.run(type_definitions)

    def _puma_atom_to_hyperon(self, atom: Atom) -> Optional[HyperonAtom]:
        """
        Convert PUMA Atom to Hyperon Atom representation.

        Type-specific conversions:
        - EpisodicMemoryNode → (EpisodicMemory id content timestamp truth confidence)
        - ConceptNode → (Concept name properties)
        - CodeNode → Parsed MeTTa expression
        - RelationalFrameNode → (RelationalFrame type relations)
        """
        if not HYPERON_AVAILABLE:
            return None

        try:
            # Extract atom properties
            atom_id = S(atom.id)
            timestamp_str = S(atom.timestamp.isoformat())
            truth_val = S(str(atom.truth_value))
            confidence_val = S(str(atom.confidence))

            # Type-specific conversion
            if atom.type == AtomType.EPISODIC_MEMORY:
                content = S(json.dumps(atom.content))
                return E(S('EpisodicMemory'), atom_id, content, timestamp_str, truth_val, confidence_val)

            elif atom.type == AtomType.CONCEPT:
                # ConceptNode: (Concept name properties)
                name = S(atom.content.get('name', atom.id) if isinstance(atom.content, dict) else str(atom.content))
                props = S(json.dumps(atom.content))
                return E(S('Concept'), atom_id, name, props, truth_val, confidence_val)

            elif atom.type == AtomType.CODE:
                # CodeNode: Parse as MeTTa executable
                if isinstance(atom.content, str):
                    # Try to parse as MeTTa expression
                    try:
                        parsed = self.metta.parse_single(atom.content)
                        return E(S('Code'), atom_id, parsed, timestamp_str)
                    except:
                        # Fallback to string representation
                        return E(S('Code'), atom_id, S(atom.content), timestamp_str)
                else:
                    return E(S('Code'), atom_id, S(str(atom.content)), timestamp_str)

            elif atom.type == AtomType.RELATIONAL_FRAME:
                # RelationalFrameNode: (RelationalFrame type relations)
                frame_type = S(atom.content.get('type', 'unknown') if isinstance(atom.content, dict) else 'unknown')
                relations = S(json.dumps(atom.content))
                return E(S('RelationalFrame'), atom_id, frame_type, relations, truth_val, confidence_val)

            elif atom.type == AtomType.SELF_MODEL:
                # SelfModelNode: (SelfModel id properties)
                props = S(json.dumps(atom.content))
                return E(S('SelfModel'), atom_id, props, timestamp_str, truth_val, confidence_val)

            elif atom.type == AtomType.GOAL:
                # GoalNode: (Goal id content status)
                content = S(json.dumps(atom.content))
                return E(S('Goal'), atom_id, content, timestamp_str, truth_val, confidence_val)

            elif atom.type == AtomType.PERCEPTION:
                # PerceptionNode: (Perception id content)
                content = S(json.dumps(atom.content))
                return E(S('Perception'), atom_id, content, timestamp_str, truth_val, confidence_val)

            elif atom.type == AtomType.EMOTIONAL_STATE:
                # EmotionalStateNode: (EmotionalState id content)
                content = S(json.dumps(atom.content))
                return E(S('EmotionalState'), atom_id, content, timestamp_str, truth_val, confidence_val)

            else:
                # Generic fallback
                content = S(json.dumps(atom.content))
                return E(S(atom.type.value), atom_id, content, timestamp_str, truth_val, confidence_val)

        except Exception as e:
            print(f"Warning: Failed to convert PUMA atom to Hyperon: {e}")
            return None

    def _hyperon_atom_to_puma(self, hatom: HyperonAtom) -> Optional[Atom]:
        """
        Convert Hyperon Atom back to PUMA Atom.

        Reverse conversion maintaining all metadata.
        """
        if not HYPERON_AVAILABLE or not isinstance(hatom, HyperonAtom):
            return None

        try:
            # Parse expression structure
            if hatom.get_type() != AtomKind.EXPR:
                return None

            children = hatom.get_children()
            if len(children) < 3:
                return None

            # Extract type and ID
            type_sym = children[0].get_name() if children[0].get_type() == AtomKind.SYMBOL else None
            atom_id = children[1].get_name() if children[1].get_type() == AtomKind.SYMBOL else str(children[1])

            # Map type back to PUMA AtomType
            type_mapping = {
                'EpisodicMemory': AtomType.EPISODIC_MEMORY,
                'Concept': AtomType.CONCEPT,
                'Code': AtomType.CODE,
                'RelationalFrame': AtomType.RELATIONAL_FRAME,
                'SelfModel': AtomType.SELF_MODEL,
                'Goal': AtomType.GOAL,
                'Perception': AtomType.PERCEPTION,
                'EmotionalState': AtomType.EMOTIONAL_STATE
            }

            atom_type = type_mapping.get(type_sym, AtomType.CONCEPT)

            # Extract content (varies by type)
            if atom_type == AtomType.CONCEPT and len(children) >= 4:
                # Concept has name and properties
                name = children[2].get_name() if children[2].get_type() == AtomKind.SYMBOL else str(children[2])
                props_str = children[3].get_name() if len(children) > 3 else "{}"
                try:
                    content = json.loads(props_str)
                except:
                    content = {'name': name}
            else:
                # Generic content extraction
                content_str = children[2].get_name() if children[2].get_type() == AtomKind.SYMBOL else str(children[2])
                try:
                    content = json.loads(content_str)
                except:
                    content = content_str

            # Extract metadata
            timestamp_str = children[-3].get_name() if len(children) > 3 else datetime.now(timezone.utc).isoformat()
            truth_value = float(children[-2].get_name()) if len(children) > 4 else 1.0
            confidence = float(children[-1].get_name()) if len(children) > 5 else 1.0

            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except:
                timestamp = datetime.now(timezone.utc)

            return Atom(
                id=atom_id,
                type=atom_type,
                content=content,
                timestamp=timestamp,
                truth_value=truth_value,
                confidence=confidence
            )

        except Exception as e:
            print(f"Warning: Failed to convert Hyperon atom to PUMA: {e}")
            return None

    def add_atom(self, atom: Atom) -> str:
        """
        Add atom to both Hyperon space and JSON backup.

        Dual persistence ensures compatibility and leverages Hyperon's
        pattern matching capabilities.
        """
        if not atom.id:
            atom.id = self._generate_atom_id()

        # Add to JSON backup (always maintained)
        self.atoms[atom.id] = atom

        # Add to Hyperon space if available
        if self.use_hyperon and self.space is not None:
            hyperon_atom = self._puma_atom_to_hyperon(atom)
            if hyperon_atom:
                self.space.add_atom(hyperon_atom)

        return atom.id

    def add_link(self, link: Link):
        """
        Add link as both JSON and Hyperon expression.

        Links in Hyperon: (Link source_id link_type target_id strength)
        """
        self.links.append(link)

        if self.use_hyperon and self.space is not None:
            try:
                link_expr = E(
                    S('Link'),
                    S(link.source_id),
                    S(link.link_type),
                    S(link.target_id),
                    S(str(link.strength))
                )
                self.space.add_atom(link_expr)
            except Exception as e:
                print(f"Warning: Failed to add link to Hyperon: {e}")

    def get_atom(self, atom_id: str) -> Optional[Atom]:
        """Retrieve atom by ID from JSON store"""
        return self.atoms.get(atom_id)

    def query_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Query atoms by type (from JSON store)"""
        return [atom for atom in self.atoms.values() if atom.type == atom_type]

    def query_hyperon(self, pattern: str) -> List[Dict[str, Any]]:
        """
        Query using Hyperon's pattern matching.

        Args:
            pattern: MeTTa query pattern, e.g.:
                "(EpisodicMemory $id $content $ts $tv $conf)"
                "(Concept $id $name $props $tv $conf)"

        Returns:
            List of binding dictionaries matching the pattern
        """
        if not self.use_hyperon or not self.metta:
            return []

        try:
            # Execute query in MeTTa
            results = self.metta.run(f"!(match &self {pattern} $match)")

            # Parse results
            bindings = []
            for result in results:
                if isinstance(result, list):
                    bindings.extend(result)
                else:
                    bindings.append(result)

            return bindings
        except Exception as e:
            print(f"Warning: Hyperon query failed: {e}")
            return []

    def get_linked_atoms(self, atom_id: str, link_type: Optional[str] = None) -> List[Atom]:
        """Get atoms linked to given atom"""
        linked_ids = []
        for link in self.links:
            if link.source_id == atom_id:
                if link_type is None or link.link_type == link_type:
                    linked_ids.append(link.target_id)

        return [self.atoms[aid] for aid in linked_ids if aid in self.atoms]

    def save(self):
        """
        Save atomspace to dual storage:
        1. JSON files (backward compatible)
        2. Hyperon native storage (if available)
        """
        if not self.persistence_path:
            return

        self.persistence_path.mkdir(parents=True, exist_ok=True)

        # Save atoms to JSON
        atoms_data = {aid: atom.to_dict() for aid, atom in self.atoms.items()}
        with open(self.persistence_path / 'atoms.json', 'w') as f:
            json.dump(atoms_data, f, indent=2)

        # Save links to JSON
        links_data = [link.to_dict() for link in self.links]
        with open(self.persistence_path / 'links.json', 'w') as f:
            json.dump(links_data, f, indent=2)

        # Save Hyperon space state (if available)
        if self.use_hyperon and self.metta:
            self._save_hyperon_state()

    def _save_hyperon_state(self):
        """
        Save Hyperon atomspace state to MeTTa file.

        Exports all atoms from GroundingSpace to .metta format
        for native Hyperon persistence.
        """
        if not self.space:
            return

        try:
            metta_file = self.persistence_path / 'atomspace.metta'

            # Get all atoms from space
            with open(metta_file, 'w') as f:
                f.write("; PUMA Atomspace - Hyperon Native Storage\n")
                f.write("; Generated: " + datetime.now(timezone.utc).isoformat() + "\n\n")

                # Export type definitions
                f.write("; Type Definitions\n")
                f.write("(: EpisodicMemory Type)\n")
                f.write("(: Concept Type)\n")
                f.write("(: RelationalFrame Type)\n")
                f.write("(: SelfModel Type)\n")
                f.write("(: Goal Type)\n")
                f.write("(: Perception Type)\n")
                f.write("(: EmotionalState Type)\n\n")

                # Export atoms
                f.write("; Atoms\n")
                for atom in self.atoms.values():
                    hyperon_atom = self._puma_atom_to_hyperon(atom)
                    if hyperon_atom:
                        f.write(str(hyperon_atom) + "\n")

                # Export links
                f.write("\n; Links\n")
                for link in self.links:
                    link_expr = f"(Link {link.source_id} {link.link_type} {link.target_id} {link.strength})\n"
                    f.write(link_expr)

        except Exception as e:
            print(f"Warning: Failed to save Hyperon state: {e}")

    def load(self):
        """
        Load atomspace from dual storage.

        Priority: JSON (always) + Hyperon (if available)
        """
        if not self.persistence_path:
            return

        # Load from JSON (primary source)
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

        # Load Hyperon state and sync with JSON
        if self.use_hyperon:
            self._load_hyperon_state()

    def _load_hyperon_state(self):
        """
        Load Hyperon atomspace from .metta file.

        Populates GroundingSpace with persisted atoms.
        """
        if not self.metta or not self.space:
            return

        metta_file = self.persistence_path / 'atomspace.metta'
        if not metta_file.exists():
            # No Hyperon state yet, populate from JSON
            for atom in self.atoms.values():
                hyperon_atom = self._puma_atom_to_hyperon(atom)
                if hyperon_atom:
                    self.space.add_atom(hyperon_atom)
            return

        try:
            # Load MeTTa file
            with open(metta_file, 'r') as f:
                metta_code = f.read()

            # Execute in MeTTa runtime to populate space
            self.metta.run(metta_code)

        except Exception as e:
            print(f"Warning: Failed to load Hyperon state: {e}")

    def create_snapshot(self) -> str:
        """Create versioned snapshot of atomspace state"""
        if not self.persistence_path:
            return ""

        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        snapshot_dir = self.persistence_path / 'snapshots' / timestamp
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON snapshot
        atoms_data = {aid: atom.to_dict() for aid, atom in self.atoms.items()}
        with open(snapshot_dir / 'atoms.json', 'w') as f:
            json.dump(atoms_data, f, indent=2)

        links_data = [link.to_dict() for link in self.links]
        with open(snapshot_dir / 'links.json', 'w') as f:
            json.dump(links_data, f, indent=2)

        # Save Hyperon snapshot if available
        if self.use_hyperon and self.metta:
            import shutil
            metta_file = self.persistence_path / 'atomspace.metta'
            if metta_file.exists():
                shutil.copy(metta_file, snapshot_dir / 'atomspace.metta')

        return timestamp

    def restore_snapshot(self, snapshot_id: str):
        """Restore atomspace from snapshot"""
        if not self.persistence_path:
            return

        snapshot_dir = self.persistence_path / 'snapshots' / snapshot_id
        if not snapshot_dir.exists():
            raise ValueError(f"Snapshot {snapshot_id} not found")

        # Clear current state
        self.atoms.clear()
        self.links.clear()

        # Reinitialize Hyperon space
        if self.use_hyperon:
            self._init_hyperon()

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

        # Restore Hyperon state
        if self.use_hyperon:
            metta_snapshot = snapshot_dir / 'atomspace.metta'
            if metta_snapshot.exists():
                with open(metta_snapshot, 'r') as f:
                    self.metta.run(f.read())

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

    Compatible with both Atomspace and HyperonAtomspaceAdapter.
    """

    def __init__(self, atomspace: Union[Atomspace, HyperonAtomspaceAdapter]):
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

    def get_backend_type(self) -> str:
        """Get the type of atomspace backend in use"""
        if isinstance(self.atomspace, HyperonAtomspaceAdapter):
            return "HyperonAtomspaceAdapter" if self.atomspace.use_hyperon else "HyperonAtomspaceAdapter (JSON fallback)"
        else:
            return "Atomspace (JSON)"


def bootstrap_atomspace(
    persistence_path: Optional[Path] = None,
    use_hyperon: bool = True
) -> Union[Atomspace, HyperonAtomspaceAdapter]:
    """
    Bootstrap fresh atomspace with structural schema only.
    NO HARDCODED CONTENT - only creates capacity for experience.

    Args:
        persistence_path: Path for persistence storage
        use_hyperon: Use HyperonAtomspaceAdapter if True and available,
                     otherwise use JSON-based Atomspace

    Returns:
        Either HyperonAtomspaceAdapter (preferred) or Atomspace (fallback)
    """
    # Choose implementation based on availability and preference
    if use_hyperon and HYPERON_AVAILABLE:
        atomspace = HyperonAtomspaceAdapter(persistence_path, use_hyperon=True)
    else:
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


def get_atomspace_info() -> Dict[str, Any]:
    """
    Get information about available atomspace implementations.

    Returns:
        Dictionary with implementation details
    """
    return {
        'hyperon_available': HYPERON_AVAILABLE,
        'hyperon_version': getattr(__import__('hyperon'), '__version__', 'unknown') if HYPERON_AVAILABLE else None,
        'default_backend': 'HyperonAtomspaceAdapter' if HYPERON_AVAILABLE else 'Atomspace (JSON)',
        'supported_backends': ['Atomspace (JSON)', 'HyperonAtomspaceAdapter'] if HYPERON_AVAILABLE else ['Atomspace (JSON)']
    }


# Example query patterns for HyperonAtomspaceAdapter
EXAMPLE_QUERIES = {
    'all_episodic_memories': '(EpisodicMemory $id $content $ts $tv $conf)',
    'all_concepts': '(Concept $id $name $props $tv $conf)',
    'all_goals': '(Goal $id $content $ts $tv $conf)',
    'all_relational_frames': '(RelationalFrame $id $type $relations $tv $conf)',
    'links_from_atom': '(Link atom_123 $type $target $strength)',
    'specific_link_type': '(Link $source "semantic" $target $strength)',
    'code_nodes': '(Code $id $expr $ts)',
}


def demonstrate_hyperon_queries(atomspace: HyperonAtomspaceAdapter):
    """
    Demonstrate Hyperon pattern matching queries.

    Example usage:
        adapter = HyperonAtomspaceAdapter(Path('./data'))
        demonstrate_hyperon_queries(adapter)
    """
    if not isinstance(atomspace, HyperonAtomspaceAdapter):
        print("Error: Requires HyperonAtomspaceAdapter instance")
        return

    if not atomspace.use_hyperon:
        print("Hyperon not available or not enabled")
        return

    print("=== Hyperon Query Examples ===\n")

    # Query all concepts
    print("1. Query all concepts:")
    results = atomspace.query_hyperon(EXAMPLE_QUERIES['all_concepts'])
    print(f"   Found {len(results)} concepts")

    # Query episodic memories
    print("\n2. Query all episodic memories:")
    results = atomspace.query_hyperon(EXAMPLE_QUERIES['all_episodic_memories'])
    print(f"   Found {len(results)} episodic memories")

    # Query goals
    print("\n3. Query all goals:")
    results = atomspace.query_hyperon(EXAMPLE_QUERIES['all_goals'])
    print(f"   Found {len(results)} goals")

    print("\n=== Query Pattern Reference ===")
    for name, pattern in EXAMPLE_QUERIES.items():
        print(f"{name}: {pattern}")


# Module-level initialization check
if __name__ == "__main__":
    # Display atomspace configuration
    info = get_atomspace_info()
    print("=== PUMA Atomspace Configuration ===")
    print(f"Hyperon Available: {info['hyperon_available']}")
    print(f"Hyperon Version: {info['hyperon_version']}")
    print(f"Default Backend: {info['default_backend']}")
    print(f"Supported Backends: {', '.join(info['supported_backends'])}")
    print("\nIntegration Status: ✓ Complete" if HYPERON_AVAILABLE else "\nIntegration Status: JSON-only fallback")
