"""
SubAgentCoordinator Usage Examples

Demonstrates the various coordination strategies, communication patterns,
and capabilities of the SubAgentCoordinator system.
"""

import asyncio
import time
from pathlib import Path

# Import coordinator components
from coordinator import (
    SubAgentCoordinator,
    CoordinationStrategy,
    CommunicationPattern,
    TaskPriority,
    SubAgentTask,
    ConsciousnessState,
)

# Import Atomspace if available
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from atomspace_db.core import Atomspace
except ImportError:
    Atomspace = None


# ============================================================================
# Example Task Functions
# ============================================================================


async def process_data(data: str, delay: float = 0.5) -> str:
    """Simulate data processing"""
    await asyncio.sleep(delay)
    return f"Processed: {data}"


async def analyze_pattern(pattern: str) -> dict:
    """Simulate pattern analysis"""
    await asyncio.sleep(0.3)
    return {
        'pattern': pattern,
        'complexity': len(pattern),
        'unique_chars': len(set(pattern)),
    }


def compute_sum(numbers: list) -> int:
    """Synchronous computation"""
    time.sleep(0.2)
    return sum(numbers)


async def solve_problem(problem: str, difficulty: int = 5) -> str:
    """Simulate problem solving with variable difficulty"""
    await asyncio.sleep(difficulty * 0.1)
    return f"Solution to '{problem}': {42 * difficulty}"


async def aggregate_results(input: list) -> dict:
    """Pipeline aggregation function"""
    return {
        'count': len(input),
        'summary': f"Aggregated {len(input)} items",
    }


# ============================================================================
# Example 1: Basic Parallel Execution
# ============================================================================


async def example_parallel_execution():
    """Demonstrate parallel task execution with load balancing"""
    print("\n" + "=" * 60)
    print("Example 1: Parallel Execution with Load Balancing")
    print("=" * 60)

    # Create coordinator
    coordinator = SubAgentCoordinator(
        max_agents=5,
        default_strategy=CoordinationStrategy.PARALLEL,
    )

    # Register agents with different capabilities
    coordinator.register_agent(
        agent_id="agent_1",
        name="Data Processor 1",
        capabilities={"data_processing"},
    )
    coordinator.register_agent(
        agent_id="agent_2",
        name="Data Processor 2",
        capabilities={"data_processing"},
    )
    coordinator.register_agent(
        agent_id="agent_3",
        name="Pattern Analyzer",
        capabilities={"pattern_analysis"},
    )

    # Submit multiple tasks
    task_ids = []
    for i in range(5):
        task_id = await coordinator.submit_task(
            process_data,
            f"data_{i}",
            delay=0.5,
            name=f"Process Data {i}",
            priority=TaskPriority.NORMAL,
        )
        task_ids.append(task_id)

    # Wait for all tasks to complete
    print(f"\nSubmitted {len(task_ids)} tasks, waiting for completion...")

    results = []
    for task_id in task_ids:
        result = await coordinator.wait_for_task(task_id, timeout=10.0)
        results.append(result)
        print(f"  Task {result.task_id[:8]}... completed in {result.execution_time:.2f}s")

    # Show metrics
    metrics = coordinator.get_metrics()
    print(f"\nMetrics:")
    print(f"  Total tasks: {metrics.total_tasks}")
    print(f"  Completed: {metrics.completed_tasks}")
    print(f"  Failed: {metrics.failed_tasks}")
    print(f"  Avg execution time: {metrics.average_execution_time:.2f}s")


# ============================================================================
# Example 2: Sequential Execution with Dependencies
# ============================================================================


async def example_sequential_dependencies():
    """Demonstrate sequential execution with task dependencies"""
    print("\n" + "=" * 60)
    print("Example 2: Sequential Execution with Dependencies")
    print("=" * 60)

    coordinator = SubAgentCoordinator(
        max_agents=3,
        default_strategy=CoordinationStrategy.SEQUENTIAL,
    )

    # Register agents
    coordinator.register_agent("seq_agent_1", "Sequential Agent 1")
    coordinator.register_agent("seq_agent_2", "Sequential Agent 2")

    # Create task chain: A -> B -> C
    task_a = await coordinator.submit_task(
        process_data,
        "step_A",
        name="Task A",
    )

    task_b = await coordinator.submit_task(
        process_data,
        "step_B",
        name="Task B",
        dependencies=[task_a],  # B depends on A
    )

    task_c = await coordinator.submit_task(
        process_data,
        "step_C",
        name="Task C",
        dependencies=[task_b],  # C depends on B
    )

    print(f"\nCreated dependency chain: {task_a[:8]} -> {task_b[:8]} -> {task_c[:8]}")

    # Wait for final task
    final_result = await coordinator.wait_for_task(task_c, timeout=20.0)
    print(f"\nFinal task completed: {final_result.result}")

    # Show execution order
    print("\nExecution order:")
    for task_id in [task_a, task_b, task_c]:
        result = coordinator.task_results[task_id]
        print(f"  {result.task_id[:8]}... executed by {result.agent_id}")


# ============================================================================
# Example 3: Competitive Execution
# ============================================================================


async def example_competitive_execution():
    """Demonstrate competitive execution (multiple agents, best result wins)"""
    print("\n" + "=" * 60)
    print("Example 3: Competitive Execution")
    print("=" * 60)

    coordinator = SubAgentCoordinator(max_agents=5)

    # Register multiple agents
    for i in range(5):
        coordinator.register_agent(
            f"comp_agent_{i}",
            f"Competitive Agent {i}",
        )

    # Create a task
    task = SubAgentTask(
        task_id="",
        name="Solve Complex Problem",
        function=solve_problem,
        args=("NP-hard problem",),
        kwargs={'difficulty': 5},
    )

    print("\nRunning competitive execution with 5 agents...")

    # Execute competitively
    best_result = await coordinator.execute_competitive(
        task,
        num_agents=5,
        selection_strategy='fastest',
    )

    print(f"\nBest result from agent: {best_result.agent_id}")
    print(f"Result: {best_result.result}")
    print(f"Execution time: {best_result.execution_time:.3f}s")


# ============================================================================
# Example 4: Pipeline Execution
# ============================================================================


async def example_pipeline_execution():
    """Demonstrate pipeline execution with output passing"""
    print("\n" + "=" * 60)
    print("Example 4: Pipeline Execution")
    print("=" * 60)

    coordinator = SubAgentCoordinator(max_agents=3)

    # Register pipeline agents
    coordinator.register_agent("pipe_1", "Pipeline Stage 1")
    coordinator.register_agent("pipe_2", "Pipeline Stage 2")
    coordinator.register_agent("pipe_3", "Pipeline Stage 3")

    # Create pipeline tasks
    tasks = [
        SubAgentTask(
            task_id="",
            name="Generate Data",
            function=lambda: ["item1", "item2", "item3"],
        ),
        SubAgentTask(
            task_id="",
            name="Process Data",
            function=lambda input: [f"processed_{item}" for item in input],
        ),
        SubAgentTask(
            task_id="",
            name="Aggregate Results",
            function=aggregate_results,
        ),
    ]

    print("\nExecuting 3-stage pipeline...")

    # Execute pipeline
    final_result = await coordinator.execute_pipeline(tasks)

    print(f"\nPipeline completed!")
    print(f"Final result: {final_result.result}")


# ============================================================================
# Example 5: Consensus Execution
# ============================================================================


async def example_consensus_execution():
    """Demonstrate consensus-based execution"""
    print("\n" + "=" * 60)
    print("Example 5: Consensus Execution")
    print("=" * 60)

    coordinator = SubAgentCoordinator(max_agents=5)

    # Register voting agents
    for i in range(5):
        coordinator.register_agent(
            f"voter_{i}",
            f"Voting Agent {i}",
        )

    # Create task that should produce consistent results
    task = SubAgentTask(
        task_id="",
        name="Compute Sum",
        function=compute_sum,
        args=([1, 2, 3, 4, 5],),
    )

    print("\nRunning consensus execution with 5 agents...")
    print("Threshold: 66% agreement required")

    # Execute with consensus
    consensus_result = await coordinator.execute_with_consensus(
        task,
        num_agents=5,
        consensus_threshold=0.66,
    )

    print(f"\nConsensus result: {consensus_result.result}")
    print(f"Votes: {consensus_result.metadata.get('consensus_votes', 0)}/5")
    print(f"Consensus {'achieved' if consensus_result.metadata.get('consensus_votes', 0) >= 3 else 'failed'}")


# ============================================================================
# Example 6: Communication Patterns
# ============================================================================


async def example_communication_patterns():
    """Demonstrate different communication patterns"""
    print("\n" + "=" * 60)
    print("Example 6: Communication Patterns")
    print("=" * 60)

    coordinator = SubAgentCoordinator(max_agents=5)

    # Register agents
    agents = []
    for i in range(5):
        agent_id = f"comm_agent_{i}"
        coordinator.register_agent(agent_id, f"Comm Agent {i}")
        agents.append(agent_id)

    # 1. Broadcast
    print("\n1. Broadcasting message to all agents...")
    count = await coordinator.broadcast(
        sender_id=agents[0],
        topic="system_update",
        content={"version": "2.0", "status": "active"},
    )
    print(f"   Message sent to {count} agents")

    # 2. Point-to-point
    print("\n2. Sending point-to-point message...")
    success = await coordinator.send_message(
        sender_id=agents[0],
        receiver_id=agents[1],
        topic="task_assignment",
        content={"task": "process_data", "priority": "high"},
    )
    print(f"   Message sent: {success}")

    # 3. Publish-Subscribe
    print("\n3. Using publish-subscribe pattern...")

    # Subscribe agents to topics
    coordinator.subscribe(agents[1], "data_updates")
    coordinator.subscribe(agents[2], "data_updates")
    coordinator.subscribe(agents[3], "alerts")

    # Publish to topic
    subs = await coordinator.publish(
        sender_id=agents[0],
        topic="data_updates",
        content={"new_data": [1, 2, 3, 4, 5]},
    )
    print(f"   Published to {subs} subscribers on 'data_updates'")

    # 4. Request-Reply
    print("\n4. Request-Reply pattern...")

    # Simulate agent that can reply
    async def handle_request():
        messages = await coordinator.receive_messages(agents[1], timeout=1.0)
        if messages:
            msg = messages[0]
            if msg.metadata.get('expects_reply'):
                await coordinator.send_reply(
                    sender_id=agents[1],
                    request_message=msg,
                    content={"status": "completed", "result": 42},
                )

    # Start handler in background
    handler_task = asyncio.create_task(handle_request())

    # Send request
    reply = await coordinator.request_reply(
        sender_id=agents[0],
        receiver_id=agents[1],
        topic="compute_request",
        content={"operation": "sum", "values": [1, 2, 3]},
        timeout=5.0,
    )

    await handler_task
    print(f"   Received reply: {reply}")


# ============================================================================
# Example 7: Consciousness State Integration
# ============================================================================


async def example_consciousness_integration():
    """Demonstrate integration with PUMA consciousness states"""
    print("\n" + "=" * 60)
    print("Example 7: Consciousness State Integration")
    print("=" * 60)

    coordinator = SubAgentCoordinator(
        max_agents=3,
        consciousness_integration=True,
    )

    # Register agents
    coordinator.register_agent("aware_1", "Aware Agent 1")
    coordinator.register_agent("aware_2", "Aware Agent 2")

    # Show state transitions and strategy changes
    states = [
        ConsciousnessState.IDLE,
        ConsciousnessState.EXPLORING,
        ConsciousnessState.CONVERSING,
        ConsciousnessState.SLEEPING,
    ]

    for state in states:
        coordinator.set_consciousness_state(state)
        print(f"\nState: {state.value}")
        print(f"  Default Strategy: {coordinator.default_strategy.value}")

        # Submit task appropriate for state
        if state == ConsciousnessState.EXPLORING:
            await coordinator.submit_task(
                analyze_pattern,
                "complex_pattern_xyz",
                name="Explore Pattern",
            )
        elif state == ConsciousnessState.CONVERSING:
            await coordinator.submit_task(
                solve_problem,
                "user_question",
                name="Answer Question",
            )

    print("\nConsciousness integration demonstrated!")


# ============================================================================
# Example 8: Fault Tolerance and Retry
# ============================================================================


async def example_fault_tolerance():
    """Demonstrate fault tolerance and retry logic"""
    print("\n" + "=" * 60)
    print("Example 8: Fault Tolerance and Retry")
    print("=" * 60)

    coordinator = SubAgentCoordinator(max_agents=3)

    coordinator.register_agent("fault_agent_1", "Fault-Tolerant Agent")

    # Create a task that fails initially
    attempt_count = 0

    async def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise Exception(f"Temporary failure (attempt {attempt_count})")
        return "Success after retries!"

    print("\nSubmitting task that fails initially...")

    task_id = await coordinator.submit_task(
        flaky_function,
        name="Flaky Task",
        max_retries=3,
    )

    # Start coordinator worker to process task
    coordinator_task = asyncio.create_task(coordinator._worker_loop())

    # Wait for completion
    result = await coordinator.wait_for_task(task_id, timeout=10.0)

    # Stop worker
    coordinator.running = False
    await asyncio.sleep(0.1)

    print(f"\nTask result: {result.result}")
    print(f"Retry count: {result.retry_count}")
    print(f"Status: {result.status.value}")


# ============================================================================
# Example 9: Monitoring and Debugging
# ============================================================================


async def example_monitoring():
    """Demonstrate monitoring and debugging capabilities"""
    print("\n" + "=" * 60)
    print("Example 9: Monitoring and Debugging")
    print("=" * 60)

    coordinator = SubAgentCoordinator(max_agents=3)

    # Register event handlers
    events_received = []

    def on_task_submitted(task):
        events_received.append(('task_submitted', task.name))

    def on_task_completed(task, result):
        events_received.append(('task_completed', task.name, result.status.value))

    coordinator.on('task_submitted', on_task_submitted)
    coordinator.on('task_completed', on_task_completed)

    # Register agents
    coordinator.register_agent("mon_1", "Monitored Agent 1")
    coordinator.register_agent("mon_2", "Monitored Agent 2")

    # Submit tasks
    for i in range(3):
        await coordinator.submit_task(
            process_data,
            f"monitored_data_{i}",
            name=f"Monitored Task {i}",
        )

    # Start processing
    coordinator.running = True
    worker = asyncio.create_task(coordinator._worker_loop())

    # Wait a bit for processing
    await asyncio.sleep(2.0)

    # Stop
    coordinator.running = False
    await asyncio.sleep(0.1)

    # Show events
    print("\nEvents received:")
    for event in events_received:
        print(f"  {event}")

    # Show debug info
    print("\n" + coordinator.debug_info())


# ============================================================================
# Example 10: Atomspace Integration
# ============================================================================


async def example_atomspace_integration():
    """Demonstrate Atomspace integration for shared memory"""
    print("\n" + "=" * 60)
    print("Example 10: Atomspace Integration")
    print("=" * 60)

    if not Atomspace:
        print("\nAtomspace not available, skipping example")
        return

    # Create atomspace
    atomspace = Atomspace()

    # Create coordinator with atomspace
    coordinator = SubAgentCoordinator(
        atomspace=atomspace,
        max_agents=3,
        enable_atomspace_pubsub=True,
    )

    coordinator.register_agent("atom_1", "Atomspace Agent 1")
    coordinator.register_agent("atom_2", "Atomspace Agent 2")

    # Send messages (will be stored in atomspace)
    print("\nSending messages via Atomspace pub-sub...")

    await coordinator.publish(
        sender_id="atom_1",
        topic="atomspace_topic",
        content={"data": "shared_knowledge"},
    )

    # Check atomspace
    print(f"Atoms in atomspace: {atomspace.count_atoms()}")

    print("\nAtomspace integration demonstrated!")


# ============================================================================
# Main Entry Point
# ============================================================================


async def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("SubAgentCoordinator - Comprehensive Examples")
    print("=" * 60)

    examples = [
        ("Parallel Execution", example_parallel_execution),
        ("Sequential Dependencies", example_sequential_dependencies),
        ("Competitive Execution", example_competitive_execution),
        ("Pipeline Execution", example_pipeline_execution),
        ("Consensus Execution", example_consensus_execution),
        ("Communication Patterns", example_communication_patterns),
        ("Consciousness Integration", example_consciousness_integration),
        ("Fault Tolerance", example_fault_tolerance),
        ("Monitoring", example_monitoring),
        ("Atomspace Integration", example_atomspace_integration),
    ]

    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
