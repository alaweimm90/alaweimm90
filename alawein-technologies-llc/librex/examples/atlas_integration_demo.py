#!/usr/bin/env python3
"""
ORCHEX-Librex Integration Demo

Demonstrates how to use Librex with MEZAN/ORCHEX for multi-agent
optimization workflows.
"""

import json
import logging
import time
from typing import Dict, Any

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_standalone_agent():
    """
    Demo: Deploy Librex as a standalone ORCHEX agent
    """
    from Librex.integrations.ORCHEX import LibrexAgent, ATLASConfig

    print("\n" + "="*60)
    print("DEMO: Standalone Librex Agent")
    print("="*60)

    # Configure agent
    config = ATLASConfig(
        agent_id="Librex-demo-agent",
        max_concurrent_tasks=5,
        default_method="auto",
        enable_gpu=False  # Set to True if GPU available
    )

    # Initialize and register agent
    agent = LibrexAgent(config)
    agent.register_with_atlas()

    # Start processing tasks
    agent.start_background_processor()

    # Get agent status
    status = agent.get_agent_status()
    print(f"\nAgent Status:")
    print(json.dumps(status, indent=2))

    # Simulate running for a bit
    print("\nAgent is running and ready to process tasks...")
    time.sleep(5)

    # Cleanup
    agent.stop_background_processor()
    agent.deregister_from_atlas()

    print("\nAgent shutdown complete")


def demo_task_submission():
    """
    Demo: Submit optimization tasks to the queue
    """
    from Librex.integrations.ORCHEX import ATLASOptimizationAdapter, ATLASConfig

    print("\n" + "="*60)
    print("DEMO: Task Submission")
    print("="*60)

    # Initialize adapter
    config = ATLASConfig.from_env()
    adapter = ATLASOptimizationAdapter(config)

    # Example 1: Submit QAP task
    print("\n1. Submitting QAP optimization task...")

    n = 10  # Problem size
    flow_matrix = np.random.randint(1, 100, size=(n, n))
    distance_matrix = np.random.randint(1, 100, size=(n, n))

    qap_result = adapter.submit_optimization_request(
        agent_id="demo-agent",
        problem_type="qap",
        problem_data={
            "flow_matrix": flow_matrix.tolist(),
            "distance_matrix": distance_matrix.tolist()
        },
        method="auto",  # Let AI select the method
        priority=1,
        metadata={"demo": "qap_example"}
    )

    print(f"   Task ID: {qap_result['task_id']}")
    print(f"   Method: {qap_result['method']}")
    print(f"   Est. time: {qap_result['estimated_completion_seconds']}s")
    print(f"   Blackboard: {qap_result['redis_blackboard_key']}")

    # Example 2: Submit TSP task
    print("\n2. Submitting TSP optimization task...")

    num_cities = 15
    coordinates = np.random.rand(num_cities, 2) * 100

    tsp_result = adapter.submit_optimization_request(
        agent_id="demo-agent",
        problem_type="tsp",
        problem_data={
            "coordinates": coordinates.tolist()
        },
        method="simulated_annealing",
        config={
            "initial_temperature": 1000,
            "cooling_rate": 0.95,
            "max_iterations": 5000
        },
        priority=2,
        metadata={"demo": "tsp_example"}
    )

    print(f"   Task ID: {tsp_result['task_id']}")
    print(f"   Method: {tsp_result['method']}")

    return qap_result['task_id'], tsp_result['task_id']


def demo_method_recommendation():
    """
    Demo: Get AI-powered method recommendations
    """
    from Librex.integrations.ORCHEX import ATLASOptimizationAdapter, ATLASConfig

    print("\n" + "="*60)
    print("DEMO: AI Method Recommendation")
    print("="*60)

    adapter = ATLASOptimizationAdapter(ATLASConfig.from_env())

    # Get recommendation for a large QAP problem
    n = 100
    problem_data = {
        "flow_matrix": np.random.rand(n, n).tolist(),
        "distance_matrix": np.random.rand(n, n).tolist()
    }

    recommendation = adapter.get_method_recommendation(
        agent_id="demo-agent",
        problem_type="qap",
        problem_data=problem_data
    )

    print(f"\nProblem: QAP with size {n}x{n}")
    print(f"Recommended method: {recommendation['recommendation']['method']}")
    print(f"Confidence: {recommendation['recommendation']['confidence']:.1%}")
    print(f"Explanation: {recommendation['recommendation']['explanation']}")
    print(f"\nSupporting evidence:")
    for evidence in recommendation['recommendation']['supporting_evidence']:
        print(f"  - {evidence}")


def demo_batch_operations():
    """
    Demo: Submit multiple tasks in batch
    """
    from Librex.integrations.ORCHEX import ATLASOptimizationAdapter, ATLASConfig

    print("\n" + "="*60)
    print("DEMO: Batch Task Submission")
    print("="*60)

    adapter = ATLASOptimizationAdapter(ATLASConfig.from_env())

    # Prepare batch of tasks
    tasks = []

    # Add QAP tasks of different sizes
    for size in [5, 10, 15]:
        tasks.append({
            "problem_type": "qap",
            "problem_data": {
                "flow_matrix": np.random.rand(size, size).tolist(),
                "distance_matrix": np.random.rand(size, size).tolist()
            },
            "method": "auto",
            "priority": size // 5,  # Smaller problems have higher priority
            "metadata": {"batch": "qap", "size": size}
        })

    # Add TSP tasks
    for num_cities in [10, 20]:
        tasks.append({
            "problem_type": "tsp",
            "problem_data": {
                "coordinates": (np.random.rand(num_cities, 2) * 100).tolist()
            },
            "method": "simulated_annealing",
            "priority": 1,
            "metadata": {"batch": "tsp", "cities": num_cities}
        })

    # Submit batch
    print(f"\nSubmitting batch of {len(tasks)} tasks...")
    results = adapter.batch_submit("demo-agent", tasks)

    print("\nBatch submission results:")
    for i, result in enumerate(results):
        if result.get("status") == "submitted":
            print(f"  Task {i+1}: {result['task_id']} - {result['method']}")
        else:
            print(f"  Task {i+1}: ERROR - {result.get('error')}")

    return [r['task_id'] for r in results if r.get('status') == 'submitted']


def demo_status_monitoring(task_ids):
    """
    Demo: Monitor task status and retrieve results
    """
    from Librex.integrations.ORCHEX import ATLASOptimizationAdapter, ATLASConfig

    print("\n" + "="*60)
    print("DEMO: Status Monitoring")
    print("="*60)

    adapter = ATLASOptimizationAdapter(ATLASConfig.from_env())

    print(f"\nMonitoring {len(task_ids)} tasks...")

    # Check status multiple times
    for iteration in range(3):
        print(f"\n--- Check {iteration + 1} ---")

        for task_id in task_ids[:3]:  # Check first 3 tasks
            status = adapter.get_task_status(task_id)
            print(f"Task {task_id[:8]}...: {status['status']}")

            if status['status'] == 'completed':
                result = status.get('result', {})
                print(f"  Objective: {result.get('objective', 'N/A')}")
                print(f"  Valid: {result.get('is_valid', 'N/A')}")
                print(f"  Time: {result.get('execution_time_seconds', 'N/A'):.2f}s")

        time.sleep(2)


def demo_queue_management():
    """
    Demo: Direct queue management
    """
    from Librex.integrations.ORCHEX import OptimizationTaskQueue, ATLASConfig

    print("\n" + "="*60)
    print("DEMO: Queue Management")
    print("="*60)

    queue = OptimizationTaskQueue(ATLASConfig.from_env())

    # Get queue statistics
    stats = queue.get_queue_stats()
    print("\nQueue Statistics:")
    print(f"  Pending tasks: {stats['pending_tasks']}")
    print(f"  Results available: {stats['results_available']}")
    print(f"  Redis connected: {stats['redis_connected']}")
    print(f"  Task status breakdown:")
    for status, count in stats['status_breakdown'].items():
        if count > 0:
            print(f"    {status}: {count}")

    # Clean up old tasks (older than 1 day)
    deleted = queue.cleanup_old_tasks(days=1)
    print(f"\nCleaned up {deleted} old tasks")


def demo_atlas_integration():
    """
    Demo: Full ORCHEX integration example
    """
    print("\n" + "="*60)
    print("DEMO: Complete ORCHEX Integration")
    print("="*60)

    # This would be called from within an ORCHEX agent
    class OptimizationResearchAgent:
        """Example ORCHEX agent that uses Librex"""

        def __init__(self):
            from Librex.integrations.ORCHEX import ATLASOptimizationAdapter
            self.optimizer = ATLASOptimizationAdapter()
            self.agent_id = "research-agent-1"

        def execute_optimization_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
            """
            Execute optimization as part of research workflow
            """
            # Extract problem from research task
            problem_type = task.get("problem_type", "qap")
            problem_data = task.get("problem_data")

            # Submit to Librex
            result = self.optimizer.submit_optimization_request(
                agent_id=self.agent_id,
                problem_type=problem_type,
                problem_data=problem_data,
                method="auto"  # Let AI decide
            )

            # Return task info for ORCHEX orchestration
            return {
                "optimization_task_id": result["task_id"],
                "blackboard_key": result["redis_blackboard_key"],
                "estimated_completion": result["estimated_completion_seconds"]
            }

    # Example usage
    agent = OptimizationResearchAgent()
    research_task = {
        "problem_type": "qap",
        "problem_data": {
            "flow_matrix": np.random.rand(5, 5).tolist(),
            "distance_matrix": np.random.rand(5, 5).tolist()
        }
    }

    result = agent.execute_optimization_task(research_task)
    print("\nATLAS Agent Result:")
    print(json.dumps(result, indent=2))


def main():
    """
    Run all demos
    """
    print("\n" + "="*70)
    print(" ORCHEX-Librex INTEGRATION DEMONSTRATION")
    print("="*70)

    try:
        # Check Redis connection
        import redis
        r = redis.Redis.from_url("redis://localhost:6379/0", decode_responses=True)
        r.ping()
        print("\n✓ Redis connection successful")
    except Exception as e:
        print(f"\n✗ Redis connection failed: {e}")
        print("  Please ensure Redis is running: redis-server")
        return

    # Run demos
    try:
        # 1. Method recommendation
        demo_method_recommendation()

        # 2. Task submission
        qap_id, tsp_id = demo_task_submission()

        # 3. Batch operations
        batch_ids = demo_batch_operations()

        # 4. Status monitoring
        demo_status_monitoring([qap_id, tsp_id] + batch_ids[:2])

        # 5. Queue management
        demo_queue_management()

        # 6. ORCHEX integration
        demo_atlas_integration()

        # 7. Standalone agent (comment out if you want it to run continuously)
        # demo_standalone_agent()

    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)

    print("\n" + "="*70)
    print(" DEMONSTRATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()