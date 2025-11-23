"""
MeTTa Execution Engine

Provides MeTTa program execution capabilities for PUMA's cognitive architecture.
Integrates OpenCog Hyperon's MeTTa interpreter with PUMA's RFT system for
symbolic reasoning, pattern matching, and relational frame execution.

Key Features:
- Multiple execution modes (interactive, batch, async)
- Atomspace integration for knowledge representation
- RFT-to-MeTTa translation for relational frame reasoning
- PUMA DSL compilation to MeTTa expressions
- Error handling and comprehensive logging
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

# Hyperon/MeTTa imports
try:
    from hyperon import MeTTa
    from hyperon.atoms import Atom as HyperonAtom, AtomType as HyperonAtomType
    from hyperon.atoms import E, S, V, OperationAtom
    from hyperon.base import GroundingSpace, Bindings
    HYPERON_AVAILABLE = True
except ImportError:
    HYPERON_AVAILABLE = False
    MeTTa = None
    HyperonAtom = None
    GroundingSpace = None

# PUMA RFT imports
from puma.rft import RelationalFrame, RelationType, Context, Entity, Relation

logger = logging.getLogger("puma.hyperon_subagents.metta_engine")
logger.addHandler(logging.NullHandler())


class ExecutionMode(Enum):
    """Execution modes for MeTTa programs"""
    INTERACTIVE = "interactive"  # Step-by-step execution with inspection
    BATCH = "batch"              # Execute entire program at once
    ASYNC = "async"              # Asynchronous execution with callbacks


class MeTTaEngineError(Exception):
    """Base exception for MeTTa engine errors"""
    pass


class HyperonNotAvailableError(MeTTaEngineError):
    """Raised when Hyperon is not installed"""
    pass


class ExecutionError(MeTTaEngineError):
    """Raised when MeTTa program execution fails"""
    pass


class CompilationError(MeTTaEngineError):
    """Raised when DSL-to-MeTTa compilation fails"""
    pass


@dataclass
class ExecutionResult:
    """Result of MeTTa program execution"""
    success: bool
    results: List[Any]
    execution_time: float
    mode: ExecutionMode
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"ExecutionResult(status={status}, "
            f"results={len(self.results)} items, "
            f"time={self.execution_time:.4f}s, "
            f"mode={self.mode.value})"
        )


class MeTTaExecutionEngine:
    """
    MeTTa program execution engine for PUMA cognitive architecture.

    Provides a comprehensive interface for executing MeTTa programs, managing
    the Atomspace, and translating between PUMA's RFT system and MeTTa expressions.

    Attributes:
        metta: Hyperon MeTTa interpreter instance
        atomspace: Reference to Atomspace for knowledge representation
        execution_mode: Current execution mode (interactive/batch/async)
        execution_history: History of executed programs and results

    Example:
        >>> engine = MeTTaExecutionEngine()
        >>> result = engine.execute_program("(+ 2 3)")
        >>> print(result.results)  # [5]
    """

    def __init__(
        self,
        atomspace: Optional[GroundingSpace] = None,
        execution_mode: ExecutionMode = ExecutionMode.BATCH,
        enable_logging: bool = True,
    ):
        """
        Initialize MeTTa execution engine.

        Args:
            atomspace: Optional Hyperon GroundingSpace for knowledge persistence
            execution_mode: Default execution mode (interactive/batch/async)
            enable_logging: Enable detailed execution logging

        Raises:
            HyperonNotAvailableError: If Hyperon library is not installed
        """
        if not HYPERON_AVAILABLE:
            raise HyperonNotAvailableError(
                "Hyperon library not available. Install with: pip install hyperon"
            )

        self.execution_mode = execution_mode
        self.enable_logging = enable_logging

        # Initialize MeTTa interpreter
        self.metta = MeTTa()
        self.atomspace = atomspace or self.metta.space()

        # Execution tracking
        self.execution_history: List[ExecutionResult] = []
        self._executor = ThreadPoolExecutor(max_workers=4)

        # RFT integration storage
        self._rft_frames: Dict[str, RelationalFrame] = {}
        self._registered_atoms: Dict[str, HyperonAtom] = {}

        # Initialize standard library and PUMA-specific functions
        self._initialize_puma_functions()

        if self.enable_logging:
            logger.info("MeTTa Execution Engine initialized", extra={
                "mode": execution_mode.value,
                "hyperon_version": getattr(MeTTa, "__version__", "unknown"),
            })

    def _initialize_puma_functions(self):
        """
        Initialize PUMA-specific MeTTa functions and operations.
        Registers custom operations for RFT reasoning, pattern matching, etc.
        """
        # Load PUMA standard library (basic relational operations)
        puma_stdlib = """
        ; PUMA Standard Library for MeTTa
        ; Relational reasoning primitives

        ; Define relational frame constructor
        (: RelFrame (-> Symbol Symbol Symbol Float Frame))

        ; Pattern matching utilities
        (: match-pattern (-> Pattern Atom Bool))
        (: transform-by-pattern (-> Pattern Pattern Atom Atom))

        ; Frequency-based analysis (PUMA's core innovation)
        (: frequency-count (-> Atom Number))
        (: group-by-frequency (-> List List))

        ; RFT coordination (similarity detection)
        (: coordinate (-> Atom Atom Float))
        """

        try:
            self.metta.run(puma_stdlib)
            logger.debug("PUMA standard library loaded")
        except Exception as e:
            logger.warning(f"Failed to load PUMA stdlib: {e}")

    def execute_program(
        self,
        metta_code: str,
        mode: Optional[ExecutionMode] = None,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Execute a MeTTa program.

        Args:
            metta_code: MeTTa code string to execute
            mode: Execution mode (defaults to engine's default mode)
            timeout: Optional timeout in seconds

        Returns:
            ExecutionResult containing results and execution metadata

        Raises:
            ExecutionError: If program execution fails
            TimeoutError: If execution exceeds timeout

        Example:
            >>> result = engine.execute_program("(+ 1 2)")
            >>> assert result.results == [3]
        """
        mode = mode or self.execution_mode
        start_time = datetime.now(timezone.utc)

        if self.enable_logging:
            logger.info("Executing MeTTa program", extra={
                "mode": mode.value,
                "code_length": len(metta_code),
                "code_preview": metta_code[:100] if len(metta_code) > 100 else metta_code,
            })

        try:
            if mode == ExecutionMode.INTERACTIVE:
                results = self._execute_interactive(metta_code, timeout)
            elif mode == ExecutionMode.BATCH:
                results = self._execute_batch(metta_code, timeout)
            elif mode == ExecutionMode.ASYNC:
                results = self._execute_async(metta_code, timeout)
            else:
                raise ExecutionError(f"Unknown execution mode: {mode}")

            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            result = ExecutionResult(
                success=True,
                results=results,
                execution_time=execution_time,
                mode=mode,
                metadata={
                    "code": metta_code,
                    "atomspace_size": len(self._registered_atoms),
                }
            )

            self.execution_history.append(result)

            if self.enable_logging:
                logger.info("Execution completed", extra={
                    "success": True,
                    "result_count": len(results),
                    "execution_time": execution_time,
                })

            return result

        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            logger.error("Execution failed", extra={
                "error": str(e),
                "code": metta_code,
                "execution_time": execution_time,
            })

            result = ExecutionResult(
                success=False,
                results=[],
                execution_time=execution_time,
                mode=mode,
                error=str(e),
                metadata={"code": metta_code}
            )

            self.execution_history.append(result)
            return result

    def _execute_batch(self, code: str, timeout: Optional[float]) -> List[Any]:
        """Execute code in batch mode (all at once)"""
        try:
            results = self.metta.run(code)
            # Convert Hyperon results to Python types
            return [self._hyperon_to_python(r) for r in results]
        except Exception as e:
            raise ExecutionError(f"Batch execution failed: {e}") from e

    def _execute_interactive(self, code: str, timeout: Optional[float]) -> List[Any]:
        """
        Execute code in interactive mode (step by step).
        Allows inspection of intermediate results.
        """
        # Split code into individual expressions
        expressions = self._parse_expressions(code)
        results = []

        for i, expr in enumerate(expressions):
            if self.enable_logging:
                logger.debug(f"Executing expression {i+1}/{len(expressions)}: {expr}")

            try:
                expr_results = self.metta.run(expr)
                results.extend([self._hyperon_to_python(r) for r in expr_results])
            except Exception as e:
                logger.warning(f"Expression {i+1} failed: {e}")
                raise ExecutionError(f"Interactive execution failed at expression {i+1}: {e}") from e

        return results

    def _execute_async(self, code: str, timeout: Optional[float]) -> List[Any]:
        """Execute code asynchronously in thread pool"""
        future = self._executor.submit(self._execute_batch, code, timeout)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise TimeoutError(f"Async execution exceeded timeout: {timeout}s")

    def load_metta_file(self, filepath: Union[str, Path]) -> ExecutionResult:
        """
        Load and execute a MeTTa file.

        Args:
            filepath: Path to .metta file

        Returns:
            ExecutionResult from file execution

        Raises:
            FileNotFoundError: If file doesn't exist
            ExecutionError: If file execution fails

        Example:
            >>> result = engine.load_metta_file("programs/reasoning.metta")
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"MeTTa file not found: {filepath}")

        if self.enable_logging:
            logger.info(f"Loading MeTTa file: {filepath}")

        try:
            code = filepath.read_text()
            return self.execute_program(code)
        except Exception as e:
            raise ExecutionError(f"Failed to load MeTTa file {filepath}: {e}") from e

    def register_atom(
        self,
        atom_name: str,
        atom_value: Any,
        atom_type: Optional[str] = None
    ) -> HyperonAtom:
        """
        Register a custom atom in the Atomspace.

        Args:
            atom_name: Name/symbol for the atom
            atom_value: Value to bind to the atom
            atom_type: Optional type annotation

        Returns:
            Created Hyperon atom

        Example:
            >>> engine.register_atom("my_concept", {"property": "value"})
        """
        if self.enable_logging:
            logger.debug(f"Registering atom: {atom_name} = {atom_value}")

        try:
            # Create appropriate Hyperon atom based on value type
            if isinstance(atom_value, str):
                atom = S(atom_value)
            elif isinstance(atom_value, (int, float)):
                atom = E(atom_name, atom_value)
            elif isinstance(atom_value, dict):
                # Convert dict to MeTTa expression
                atom = self._dict_to_metta_atom(atom_name, atom_value)
            else:
                # Generic expression atom
                atom = E(atom_name, str(atom_value))

            # Add to atomspace
            self.atomspace.add_atom(atom)
            self._registered_atoms[atom_name] = atom

            return atom

        except Exception as e:
            raise MeTTaEngineError(f"Failed to register atom {atom_name}: {e}") from e

    def query_atomspace(self, pattern: str) -> List[Dict[str, Any]]:
        """
        Query the Atomspace using a MeTTa pattern.

        Args:
            pattern: MeTTa pattern to match against

        Returns:
            List of matched results as dictionaries

        Example:
            >>> results = engine.query_atomspace("(coordinate ?x ?y ?strength)")
            >>> # Returns all coordination frames with their bindings
        """
        if self.enable_logging:
            logger.debug(f"Querying atomspace with pattern: {pattern}")

        try:
            # Execute query as MeTTa program
            query_code = f"!(match &self {pattern} $result)"
            result = self.execute_program(query_code)

            if result.success:
                return [
                    self._result_to_dict(r) for r in result.results
                ]
            else:
                logger.warning(f"Query failed: {result.error}")
                return []

        except Exception as e:
            logger.error(f"Atomspace query error: {e}")
            return []

    def compile_dsl_to_metta(self, dsl_operation: Dict[str, Any]) -> str:
        """
        Convert PUMA DSL operation to MeTTa code.

        The PUMA DSL represents operations in a structured format that gets
        compiled to executable MeTTa expressions for symbolic reasoning.

        Args:
            dsl_operation: PUMA DSL operation dictionary with keys:
                - operation: Operation type (e.g., "pattern_match", "transform")
                - params: Operation parameters
                - context: Optional execution context

        Returns:
            MeTTa code string

        Raises:
            CompilationError: If DSL operation cannot be compiled

        Example:
            >>> dsl = {
            ...     "operation": "pattern_match",
            ...     "params": {"pattern": "(+ ?x ?y)", "target": "(+ 2 3)"}
            ... }
            >>> metta_code = engine.compile_dsl_to_metta(dsl)
            >>> # Returns: "(match (+ ?x ?y) (+ 2 3))"
        """
        if self.enable_logging:
            logger.debug(f"Compiling DSL to MeTTa: {dsl_operation}")

        try:
            operation = dsl_operation.get("operation")
            params = dsl_operation.get("params", {})

            if operation == "pattern_match":
                return self._compile_pattern_match(params)
            elif operation == "transform":
                return self._compile_transform(params)
            elif operation == "frequency_analysis":
                return self._compile_frequency_analysis(params)
            elif operation == "relational_query":
                return self._compile_relational_query(params)
            elif operation == "custom":
                return params.get("metta_code", "")
            else:
                raise CompilationError(f"Unknown DSL operation: {operation}")

        except Exception as e:
            raise CompilationError(f"DSL compilation failed: {e}") from e

    def _compile_pattern_match(self, params: Dict[str, Any]) -> str:
        """Compile pattern matching operation to MeTTa"""
        pattern = params.get("pattern", "")
        target = params.get("target", "")
        return f"!(match &self {pattern} {target})"

    def _compile_transform(self, params: Dict[str, Any]) -> str:
        """Compile transformation operation to MeTTa"""
        input_pattern = params.get("input_pattern", "")
        output_pattern = params.get("output_pattern", "")
        target = params.get("target", "")
        return f"!(transform-by-pattern {input_pattern} {output_pattern} {target})"

    def _compile_frequency_analysis(self, params: Dict[str, Any]) -> str:
        """Compile frequency analysis (PUMA's core innovation) to MeTTa"""
        items = params.get("items", [])
        # Convert to MeTTa list
        items_str = " ".join(str(item) for item in items)
        return f"!(group-by-frequency ({items_str}))"

    def _compile_relational_query(self, params: Dict[str, Any]) -> str:
        """Compile relational frame query to MeTTa"""
        relation_type = params.get("relation_type", "")
        source = params.get("source", "?source")
        target = params.get("target", "?target")
        return f"!(match &self (RelFrame {relation_type} {source} {target} ?strength) $result)"

    def rft_to_metta(self, frame: RelationalFrame) -> str:
        """
        Convert PUMA RelationalFrame to MeTTa expression.

        Integrates RFT reasoning with symbolic MeTTa execution by translating
        relational frames into MeTTa atoms that can be queried and reasoned about.

        Args:
            frame: PUMA RelationalFrame instance

        Returns:
            MeTTa expression representing the relational frame

        Example:
            >>> frame = RelationalFrame(
            ...     relation_type=RelationType.COORDINATION,
            ...     source="concept_a",
            ...     target="concept_b",
            ...     strength=0.8
            ... )
            >>> metta_expr = engine.rft_to_metta(frame)
            >>> # Returns: "(RelFrame coordination concept_a concept_b 0.8)"
        """
        if self.enable_logging:
            logger.debug(f"Converting RFT frame to MeTTa: {frame}")

        relation_name = frame.relation_type.value
        source = frame.source
        target = frame.target
        strength = frame.strength

        # Build MeTTa expression
        metta_expr = f"(RelFrame {relation_name} {source} {target} {strength})"

        # Store frame for later retrieval
        frame_id = f"{source}_{relation_name}_{target}"
        self._rft_frames[frame_id] = frame

        return metta_expr

    def context_to_metta(self, context: Context) -> str:
        """
        Convert PUMA RFT Context to MeTTa knowledge base.

        Extracts state information, constraints, and goals from RFT Context
        and represents them as MeTTa atoms for reasoning.

        Args:
            context: PUMA RFT Context instance

        Returns:
            MeTTa program representing the context

        Example:
            >>> metta_kb = engine.context_to_metta(context)
            >>> engine.execute_program(metta_kb)
        """
        metta_lines = ["; PUMA Context Knowledge Base"]

        # Add state information
        if hasattr(context.state, "__dict__"):
            state_dict = context.state.__dict__
        elif isinstance(context.state, dict):
            state_dict = context.state
        else:
            state_dict = {"value": str(context.state)}

        for key, value in state_dict.items():
            metta_lines.append(f"(state-property {key} {self._python_to_metta(value)})")

        # Add constraints
        for key, value in context.constraints.items():
            metta_lines.append(f"(constraint {key} {self._python_to_metta(value)})")

        # Add goal test (if inspectable)
        if hasattr(context.goal_test, "__name__"):
            metta_lines.append(f"; Goal: {context.goal_test.__name__}")

        # Add metrics
        for key, value in context.metrics.items():
            metta_lines.append(f"(metric {key} {value})")

        return "\n".join(metta_lines)

    def entity_to_metta(self, entity: Entity) -> str:
        """
        Convert PUMA Entity to MeTTa atom.

        Args:
            entity: PUMA Entity instance

        Returns:
            MeTTa expression representing the entity
        """
        features_str = " ".join(
            f"({k} {self._python_to_metta(v)})"
            for k, v in entity.features.items()
        )
        return f"(Entity {entity.id} {entity.type} ({features_str}))"

    def get_sample_programs(self) -> Dict[str, str]:
        """
        Get sample MeTTa programs for common PUMA operations.

        Returns:
            Dictionary mapping operation names to MeTTa code examples
        """
        return {
            "pattern_matching": """
; Pattern matching for ARC-AGI grid analysis
; Find all cells matching a color pattern

!(match &self
    (cell ?x ?y ?color)
    (= ?color blue))
            """.strip(),

            "transformation": """
; Grid transformation using pattern-based rewriting
; Transform all blue cells to red

!(transform-by-pattern
    (cell ?x ?y blue)
    (cell ?x ?y red)
    $grid)
            """.strip(),

            "relational_reasoning": """
; Use coordination frames for analogical reasoning
; Find concepts similar to "square"

!(match &self
    (RelFrame coordination square ?target ?strength)
    (> ?strength 0.7))
            """.strip(),

            "frequency_analysis": """
; PUMA's core innovation: frequency-based grouping
; Group grid objects by occurrence count

!(group-by-frequency
    (cell 0 0 blue)
    (cell 1 0 blue)
    (cell 2 0 red)
    (cell 3 0 blue))

; Expected output: Groups by frequency
; High frequency: blue (3 occurrences)
; Low frequency: red (1 occurrence)
            """.strip(),

            "hierarchical_query": """
; Query hierarchical relations for categorization
; Find all instances of a category

!(match &self
    (RelFrame hierarchy ?instance category:shape ?strength)
    $result)
            """.strip(),

            "causal_reasoning": """
; Derive causal chains through frame transitivity
; If A causes B and B causes C, then A causes C

!(match &self
    (and
        (RelFrame causal ?a ?b ?s1)
        (RelFrame causal ?b ?c ?s2))
    (RelFrame causal ?a ?c (* ?s1 ?s2)))
            """.strip(),

            "temporal_sequence": """
; Analyze temporal sequences of events
; Find events that happened before a target event

!(match &self
    (RelFrame temporal ?before target_event ?strength)
    $result)
            """.strip(),
        }

    def _parse_expressions(self, code: str) -> List[str]:
        """
        Parse MeTTa code into individual expressions.
        Simple implementation that splits on balanced parentheses.
        """
        expressions = []
        current = []
        depth = 0
        in_string = False

        for char in code:
            if char == '"':
                in_string = not in_string
            elif not in_string:
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1

            current.append(char)

            if depth == 0 and current and not in_string:
                expr = ''.join(current).strip()
                if expr and not expr.startswith(';'):  # Skip comments
                    expressions.append(expr)
                current = []

        return expressions

    def _hyperon_to_python(self, atom: Any) -> Any:
        """Convert Hyperon atom to Python type"""
        if atom is None:
            return None

        # Handle Hyperon atom types
        if hasattr(atom, 'get_type'):
            atom_type = atom.get_type()
            if atom_type == 'Symbol':
                return str(atom)
            elif atom_type == 'Number':
                return float(atom)
            elif atom_type == 'Expression':
                return [self._hyperon_to_python(child) for child in atom.get_children()]

        # Fallback: return string representation
        return str(atom)

    def _python_to_metta(self, value: Any) -> str:
        """Convert Python value to MeTTa representation"""
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return "True" if value else "False"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, (list, tuple)):
            items = " ".join(self._python_to_metta(v) for v in value)
            return f"({items})"
        elif isinstance(value, dict):
            items = " ".join(
                f"({k} {self._python_to_metta(v)})"
                for k, v in value.items()
            )
            return f"({items})"
        else:
            return f'"{str(value)}"'

    def _dict_to_metta_atom(self, name: str, data: Dict) -> HyperonAtom:
        """Convert Python dict to MeTTa expression atom"""
        items = [S(name)]
        for key, value in data.items():
            items.append(E(S(key), S(str(value))))
        return E(*items)

    def _result_to_dict(self, result: Any) -> Dict[str, Any]:
        """Convert execution result to dictionary"""
        if isinstance(result, dict):
            return result
        elif isinstance(result, (list, tuple)):
            return {"result": result}
        else:
            return {"value": str(result)}

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics and engine state.

        Returns:
            Dictionary containing execution metrics, atomspace info, etc.
        """
        total_executions = len(self.execution_history)
        successful = sum(1 for r in self.execution_history if r.success)
        failed = total_executions - successful

        total_time = sum(r.execution_time for r in self.execution_history)
        avg_time = total_time / total_executions if total_executions > 0 else 0

        return {
            "total_executions": total_executions,
            "successful_executions": successful,
            "failed_executions": failed,
            "success_rate": successful / total_executions if total_executions > 0 else 0,
            "total_execution_time": total_time,
            "average_execution_time": avg_time,
            "registered_atoms": len(self._registered_atoms),
            "rft_frames_stored": len(self._rft_frames),
            "execution_mode": self.execution_mode.value,
        }

    def reset(self):
        """Reset engine state (clear history and registered atoms)"""
        if self.enable_logging:
            logger.info("Resetting MeTTa engine")

        self.execution_history.clear()
        self._rft_frames.clear()
        self._registered_atoms.clear()

        # Reinitialize MeTTa interpreter
        self.metta = MeTTa()
        self.atomspace = self.metta.space()
        self._initialize_puma_functions()

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"MeTTaExecutionEngine("
            f"mode={self.execution_mode.value}, "
            f"executions={stats['total_executions']}, "
            f"atoms={stats['registered_atoms']})"
        )

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
