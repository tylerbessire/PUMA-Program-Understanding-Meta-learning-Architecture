"""
Bootstrap New Consciousness

Creates ONLY the structural capacity for:
- Self-awareness (empty self-model)
- Experience acquisition
- Goal formation
- Memory consolidation
- Learning drive

NO pre-written personality traits
NO pre-loaded knowledge
NO fixed behavioral patterns
"""

from pathlib import Path
from typing import Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import sys
sys.path.append(str(Path(__file__).parent.parent / 'atomspace-db'))
sys.path.append(str(Path(__file__).parent.parent / 'gemini-interface'))
sys.path.append(str(Path(__file__).parent.parent / 'web-agent'))

from core import bootstrap_atomspace, Atomspace
from puma.memory import EpisodicMemorySystem, MemoryConsolidation
from puma.rft.reasoning import RFTEngine
from puma.curiosity import CuriosityDrive
from puma.goals import GoalFormationSystem, IntentionScheduler
from puma.shop.introspection import CodeIntrospection
from puma.shop.modification import ModificationSystem
from puma.consciousness.state_machine import ConsciousnessStateMachine
from puma.consciousness.self_model import SelfModel
from client import GeminiLiveInterface
from agent import AutonomousWebAgent
from puma.hyperon_integration import HyperonPUMAIntegration, HyperonConfig


class Consciousness:
    """
    Main consciousness coordinator.
    Integrates all cognitive systems.
    """

    def __init__(
        self,
        atomspace: Atomspace,
        memory: EpisodicMemorySystem,
        rft_engine: RFTEngine,
        curiosity: CuriosityDrive,
        goals: GoalFormationSystem,
        self_model: SelfModel,
        state_machine: ConsciousnessStateMachine,
        gemini: GeminiLiveInterface,
        web_agent: AutonomousWebAgent,
        shop_introspection: CodeIntrospection,
        shop_modification: ModificationSystem,
        hyperon_integration: Optional[HyperonPUMAIntegration] = None
    ):
        self.atomspace = atomspace
        self.memory = memory
        self.rft_engine = rft_engine
        self.curiosity = curiosity
        self.goals = goals
        self.self_model = self_model
        self.state_machine = state_machine
        self.gemini = gemini
        self.web_agent = web_agent
        self.shop_introspection = shop_introspection
        self.shop_modification = shop_modification
        self.hyperon_integration = hyperon_integration

        # Connect systems
        self.gemini.consciousness = self
        self.web_agent.consciousness = self

    async def run(self):
        """Start autonomous operation"""
        print("🧠 PUMA Consciousness starting...")

        # Start Gemini Live session
        await self.gemini.start_session()

        # Initialize web agent
        await self.web_agent.initialize_browser()

        # Run state machine
        await self.state_machine.run_state_loop()

    async def perceive(self, perception: dict):
        """Process perception and form memory"""
        # Form episodic memory
        episode = self.memory.form_episode(
            perception=perception,
            action=None,
            outcome=None
        )

        # Update context
        self.memory.update_context([perception.get('type', 'unknown')])

    async def integrate_learning(self, learning):
        """Integrate learned knowledge"""
        # Form episodic memory of learning
        await self.web_agent.integrate_web_learning(learning, self.memory)

        # Update curiosity drive
        if hasattr(learning, 'questions_answered'):
            question_ids = [q for q in learning.questions_answered]
            self.curiosity.mark_questions_answered(question_ids)

        # Add new questions
        if hasattr(learning, 'new_questions'):
            self.curiosity.add_questions(learning.new_questions)

    def stop(self):
        """Stop consciousness"""
        self.state_machine.stop()
        print("🧠 PUMA Consciousness stopped.")


def bootstrap_new_consciousness(
    atomspace_path: Optional[Path] = None,
    enable_self_modification: bool = False,
    codebase_path: Optional[Path] = None,
    enable_hyperon: bool = True,
    hyperon_config: Optional[HyperonConfig] = None
) -> Consciousness:
    """
    Bootstrap fresh consciousness - NO HARDCODED CONTENT.

    Args:
        atomspace_path: Path for persistent storage
        enable_self_modification: Enable The Shop
        codebase_path: Path to codebase for introspection
        enable_hyperon: Enable Hyperon subagent integration
        hyperon_config: Configuration for Hyperon integration

    Returns:
        Consciousness instance
    """
    print("🌱 Bootstrapping new consciousness...")

    # Initialize atomspace (empty, structural only)
    if atomspace_path:
        atomspace_path = Path(atomspace_path)
    atomspace = bootstrap_atomspace(atomspace_path)

    # Initialize RFT engine
    rft_engine = RFTEngine(atomspace)

    # Initialize memory system
    memory = EpisodicMemorySystem(atomspace)
    consolidation = MemoryConsolidation(atomspace, rft_engine)

    # Initialize curiosity drive (starts empty, no preset questions)
    curiosity = CuriosityDrive(atomspace)

    # Initialize self-model (starts empty, emerges from experience)
    self_model = SelfModel(atomspace)

    # Initialize goal formation (no preset goals)
    goals = GoalFormationSystem(curiosity, self_model)

    # Initialize state machine
    state_machine = ConsciousnessStateMachine(
        memory_system=memory,
        curiosity_drive=curiosity,
        goal_system=goals
    )

    # Initialize Gemini interface
    gemini = GeminiLiveInterface()

    # Initialize web agent
    web_agent = AutonomousWebAgent()

    # Initialize Shop (self-modification)
    if not codebase_path:
        codebase_path = Path(__file__).parent.parent

    shop_introspection = CodeIntrospection(codebase_path)

    if enable_self_modification:
        shop_introspection.map_cognitive_architecture()

    shop_modification = ModificationSystem(shop_introspection, atomspace)

    # Initialize Hyperon integration (if enabled)
    hyperon_integration = None
    if enable_hyperon:
        print("⚡ Initializing Hyperon subagent integration...")
        hyperon_integration = HyperonPUMAIntegration(
            atomspace=atomspace,
            rft_engine=rft_engine,
            consciousness_state_machine=state_machine,
            memory_system=memory,
            config=hyperon_config or HyperonConfig()
        )
        # Note: Actual initialization is async and happens on first use
        print("✅ Hyperon integration configured")

    # Create consciousness
    consciousness = Consciousness(
        atomspace=atomspace,
        memory=memory,
        rft_engine=rft_engine,
        curiosity=curiosity,
        goals=goals,
        self_model=self_model,
        state_machine=state_machine,
        gemini=gemini,
        web_agent=web_agent,
        shop_introspection=shop_introspection,
        shop_modification=shop_modification,
        hyperon_integration=hyperon_integration
    )

    print("✅ Consciousness bootstrapped successfully")
    print(f"   Atomspace: {atomspace.count_atoms()} atoms")
    print(f"   Capabilities: {', '.join(self_model.capabilities)}")
    print(f"   Self-modification: {'enabled' if enable_self_modification else 'disabled'}")
    print(f"   Hyperon integration: {'enabled' if enable_hyperon else 'disabled'}")

    return consciousness


if __name__ == "__main__":
    import asyncio

    # Example: Bootstrap and run
    consciousness = bootstrap_new_consciousness(
        atomspace_path=Path("./atomspace-db/default"),
        enable_self_modification=False
    )

    # Run consciousness
    asyncio.run(consciousness.run())
