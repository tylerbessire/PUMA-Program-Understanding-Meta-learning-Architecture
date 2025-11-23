"""
Hyperon SubAgent Manager Demo

Demonstrates the usage of the Hyperon subagent management system
for parallel reasoning, pattern matching, memory retrieval, and goal planning.

This example shows:
1. Creating and managing a pool of specialized subagents
2. Executing tasks with capability-based routing
3. Parallel task execution
4. Map-reduce distributed reasoning
5. Inter-agent communication
6. Integration with PUMA's cognitive architecture
"""

import asyncio
import sys
from pathlib import Path

# Add puma to path if needed
puma_root = Path(__file__).parent.parent
sys.path.insert(0, str(puma_root))

try:
    from puma.hyperon_subagents.manager import (
        HyperonSubAgent,
        SubAgentManager,
        SubAgentTask,
        SubAgentResult,
        SubAgentState,
        AgentCapability,
        HYPERON_AVAILABLE
    )
except ImportError as e:
    print(f"Note: Some dependencies may be missing: {e}")
    print("This is expected if running without full environment setup.")
    print("\nThe manager.py module is installed at:")
    print("  /home/user/PUMA-Program-Understanding-Meta-learning-Architecture/puma/hyperon_subagents/manager.py")
    sys.exit(0)


async def demo_basic_setup():
    """Demo 1: Basic setup and agent pool creation"""
    print("=" * 60)
    print("Demo 1: Basic Setup and Agent Pool Creation")
    print("=" * 60)

    # Create manager
    manager = SubAgentManager(max_agents=10)
    print(f"\n✓ Created SubAgentManager (max {manager.max_agents} agents)")

    # Create specialized agent pool
    manager.create_specialized_agents()
    print(f"✓ Created specialized agent pool: {len(manager.agents)} agents")

    # Show pool status
    status = manager.get_pool_status()
    print(f"\nPool Status:")
    print(f"  Total agents: {status['total_agents']}")
    print(f"  State distribution: {status['state_distribution']}")
    print(f"  Capability distribution: {status['capability_distribution']}")

    return manager


async def demo_single_task(manager):
    """Demo 2: Single task execution"""
    print("\n" + "=" * 60)
    print("Demo 2: Single Task Execution")
    print("=" * 60)

    # Create a reasoning task
    task = SubAgentTask(
        task_type="reasoning",
        metta_program="""
        ; Simple forward chaining
        (= (premise) A)
        (= (rule) (implies A B))
        (infer (premise) (rule))
        """,
        context={'domain': 'logic'},
        priority=0.8
    )

    print(f"\n✓ Created task: {task.task_type} (ID: {task.id})")
    print(f"  Priority: {task.priority}")

    # Execute task
    result = await manager.execute_task(
        task,
        required_capability=AgentCapability.REASONING
    )

    print(f"\n✓ Task executed")
    print(f"  Agent: {result.agent_id}")
    print(f"  Success: {result.success}")
    print(f"  Execution time: {result.execution_time:.4f}s")
    if result.success:
        print(f"  Output atoms: {len(result.output_atoms)}")
    else:
        print(f"  Error: {result.error}")


async def demo_parallel_execution(manager):
    """Demo 3: Parallel task execution"""
    print("\n" + "=" * 60)
    print("Demo 3: Parallel Task Execution")
    print("=" * 60)

    # Create multiple pattern matching tasks
    patterns = ["(shape square)", "(color red)", "(size large)", "(texture smooth)"]
    tasks = []

    for pattern in patterns:
        task = SubAgentTask(
            task_type="pattern_matching",
            metta_program=f"(find-pattern {pattern})",
            context={'search_space': 'visual_objects'},
            priority=0.7
        )
        tasks.append(task)

    print(f"\n✓ Created {len(tasks)} pattern matching tasks")

    # Execute all tasks in parallel
    import time
    start_time = time.time()
    results = await manager.execute_parallel(tasks)
    elapsed = time.time() - start_time

    print(f"\n✓ Parallel execution completed in {elapsed:.4f}s")

    # Process results
    successful_results = [r for r in results if r.success]
    print(f"  Successful: {len(successful_results)}/{len(tasks)}")
    print(f"  Average execution time: {sum(r.execution_time for r in results)/len(results):.4f}s")


async def demo_map_reduce(manager):
    """Demo 4: Map-reduce distributed reasoning"""
    print("\n" + "=" * 60)
    print("Demo 4: Map-Reduce Distributed Reasoning")
    print("=" * 60)

    # Define map programs (execute in parallel)
    map_programs = [
        "(match &self (pattern1 $x) $x)",
        "(match &self (pattern2 $y) $y)",
        "(match &self (pattern3 $z) $z)",
    ]

    # Define reduce program (combine results)
    reduce_program = """
    (= (combine-results $results)
       (synthesize-concept $results))
    """

    print(f"\n✓ Map phase: {len(map_programs)} programs")
    print(f"✓ Reduce phase: result synthesis")

    # Execute map-reduce
    result = await manager.map_reduce_reasoning(
        map_programs,
        reduce_program,
        context={'operation': 'pattern_synthesis'}
    )

    print(f"\n✓ Map-reduce completed")
    print(f"  Success: {result.success}")
    print(f"  Combined output: {len(result.output_atoms)} atoms")


async def demo_communication(manager):
    """Demo 5: Inter-agent communication"""
    print("\n" + "=" * 60)
    print("Demo 5: Inter-Agent Communication")
    print("=" * 60)

    # Broadcast message to all agents
    manager.broadcast_message(
        message={'type': 'update', 'data': 'new_knowledge_available'},
        sender_id='control_system'
    )
    print("\n✓ Broadcast message to all agents")

    # Get an agent
    agents = list(manager.agents.values())
    if agents:
        agent = agents[0]

        # Send direct message
        manager.send_message(
            recipient_id=agent.id,
            message={'type': 'task_hint', 'hint': 'try_backward_chaining'},
            sender_id='planner'
        )
        print(f"✓ Sent direct message to {agent.name}")

        # Retrieve messages
        messages = manager.get_messages(agent.id, clear=True)
        print(f"✓ Agent received {len(messages)} messages")
        for msg in messages:
            print(f"  - From {msg['sender']}: {msg['type']}")


async def demo_agent_metrics(manager):
    """Demo 6: Performance monitoring"""
    print("\n" + "=" * 60)
    print("Demo 6: Performance Monitoring")
    print("=" * 60)

    # Get pool status
    status = manager.get_pool_status()
    print(f"\nPool Statistics:")
    print(f"  Average success rate: {status['average_success_rate']:.2%}")
    print(f"  Pending tasks: {status['pending_tasks']}")
    print(f"  Completed tasks: {status['completed_tasks']}")

    # Get individual agent metrics
    metrics = manager.get_agent_metrics()
    print(f"\nTop 3 Most Active Agents:")
    sorted_metrics = sorted(metrics, key=lambda m: m['execution_count'], reverse=True)[:3]

    for i, agent_metrics in enumerate(sorted_metrics, 1):
        print(f"\n  {i}. {agent_metrics['name']}")
        print(f"     State: {agent_metrics['state']}")
        print(f"     Capabilities: {', '.join(agent_metrics['capabilities'])}")
        print(f"     Executions: {agent_metrics['execution_count']}")
        if agent_metrics['execution_count'] > 0:
            print(f"     Success rate: {agent_metrics['success_rate']:.2%}")
            print(f"     Avg time: {agent_metrics['average_execution_time']:.4f}s")


async def demo_capability_based_routing(manager):
    """Demo 7: Capability-based task routing"""
    print("\n" + "=" * 60)
    print("Demo 7: Capability-Based Task Routing")
    print("=" * 60)

    # Find agents by capability
    capabilities_to_test = [
        AgentCapability.REASONING,
        AgentCapability.PATTERN_MATCHING,
        AgentCapability.MEMORY_RETRIEVAL,
        AgentCapability.GOAL_PLANNING
    ]

    print("\nAgent Pool Capabilities:")
    for capability in capabilities_to_test:
        agents = manager.find_agents_with_capability(capability)
        print(f"  {capability.value}: {len(agents)} agents")

        # Find best agent for this capability
        best_agent = manager.find_capable_agent(capability, prefer_idle=True)
        if best_agent:
            print(f"    → Selected: {best_agent.name} (state: {best_agent.state.value})")


async def main():
    """Run all demos"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Hyperon SubAgent Manager - Demonstration  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")

    if not HYPERON_AVAILABLE:
        print("\nNote: Hyperon not installed - running in simulation mode")
        print("For full functionality, install: pip install hyperon")

    try:
        # Run demos
        manager = await demo_basic_setup()
        await demo_single_task(manager)
        await demo_parallel_execution(manager)
        await demo_map_reduce(manager)
        await demo_communication(manager)
        await demo_capability_based_routing(manager)
        await demo_agent_metrics(manager)

        # Cleanup
        print("\n" + "=" * 60)
        print("Cleanup")
        print("=" * 60)
        manager.shutdown()
        print("\n✓ Manager shutdown complete")

    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nFor more information, see:")
    print("  /home/user/PUMA-Program-Understanding-Meta-learning-Architecture/puma/hyperon_subagents/README_MANAGER.md")
    print()


if __name__ == "__main__":
    asyncio.run(main())
