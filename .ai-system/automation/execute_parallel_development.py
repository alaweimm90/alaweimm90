#!/usr/bin/env python3
"""
Execute Parallel Development Workflow

Direct execution of parallel_development.yaml workflow
bypassing the main workflows.yaml registry issues.
"""

import yaml
import asyncio
import time
import uuid
from parallel_executor import ParallelWorkflowExecutor
from workflow_types import WorkflowContext, ParallelTask, TaskPriority


def _execute_parallel_stage(executor, stage, context):
    """Execute a single parallel stage and return the result."""
    task = ParallelTask(
        task_id=f"task_{uuid.uuid4().hex[:8]}",
        stage=stage,
        context=context,
        priority=TaskPriority.MEDIUM
    )
    return executor._handle_shell_command(task)


def _log_stage_result(stage_name, result):
    """Log the result of a stage execution."""
    if result.status.value == 'completed':
        print(f"   ✅ {stage_name}: {result.status.value} ({result.duration_ms}ms)")
        if result.output:
            if result.output.get('mode') == 'background_watch':
                print(f"   🔄 Background process started: {result.output.get('process_id')}")
            else:
                print(f"   📄 Output: {str(result.output)[:100]}...")
    else:
        print(f"   ❌ {stage_name}: {result.status.value}")
        if result.error:
            print(f"   🚨 Error: {result.error}")


def load_standalone_workflow(yaml_path: str) -> dict:
    """Load workflow definition from standalone YAML file."""
    try:
        with open(yaml_path, 'r') as f:
            workflow_data = yaml.safe_load(f)
        return workflow_data
    except Exception as e:
        print(f"❌ Failed to load workflow: {e}")
        return None


def execute_standalone_workflow():
    """Execute the parallel development workflow directly."""
    print("🚀 Executing Parallel Development Workflow")
    print("=" * 60)

    # Load the standalone workflow
    workflow_path = "workflows/config/parallel_development.yaml"
    workflow_data = load_standalone_workflow(workflow_path)

    if not workflow_data:
        print("❌ Failed to load workflow definition")
        return

    print(f"✅ Loaded workflow: {workflow_data.get('name', 'Unknown')}")
    print(f"📋 Description: {workflow_data.get('description', 'No description')}")

    # Initialize parallel executor
    executor = ParallelWorkflowExecutor(max_workers=6)
    executor.start_parallel_services()

    try:
        # Extract stages from workflow
        stages = workflow_data.get('stages', [])
        print(f"\n🔄 Executing {len(stages)} parallel stages...")

        # Create workflow context
        context = WorkflowContext(
            workflow_id="parallel_dev_direct",
            inputs={"project_path": ".", "environment": "development"}
        )

        # Execute each stage using the parallel executor
        for i, stage in enumerate(stages, 1):
            stage_name = stage.get('name', f'stage_{i}')
            print(f"\n📝 Stage {i}/{len(stages)}: {stage_name}")
            print(f"   Description: {stage.get('description', 'No description')}")
            print(f"   Priority: {stage.get('priority', 'medium')}")
            print(f"   Tools: {stage.get('tools', [])}")

            # Check if stage is parallel
            if not stage.get('parallel', False):
                print(f"   ⏭️  Skipping non-parallel stage")
                continue

            print(f"   ⚡ Executing in parallel mode...")
            result = _execute_parallel_stage(executor, stage, context)
            _log_stage_result(stage_name, result)
            context.stage_results[stage_name] = result
            context.checkpoint(stage_name)

        # Display execution summary
        print(f"\n📊 EXECUTION SUMMARY")
        print(f"=" * 60)
        print(f"Workflow ID: {context.workflow_id}")
        print(f"Stages completed: {len(context.stage_results)}")
        print(f"Checkpoints: {len(context.checkpoints)}")

        # Display stage results
        completed = sum(1 for r in context.stage_results.values() if r.status.value == 'completed')
        failed = sum(1 for r in context.stage_results.values() if r.status.value == 'failed')

        print(f"\n📈 Results:")
        print(f"   ✅ Completed: {completed}")
        print(f"   ❌ Failed: {failed}")

        # Display background processes
        bg_processes = len(executor.background_manager.processes)
        if bg_processes > 0:
            print(f"\n🔄 Background Processes: {bg_processes}")
            health = executor.background_manager.health_check()
            print(f"   Healthy: {health['healthy']}")
            print(f"   Unhealthy: {health['unhealthy']}")

        print(f"\n🎯 Parallel Development Workflow Execution Complete!")

    except Exception as e:
        print(f"❌ Execution error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up services
        executor.stop_parallel_services()
        print(f"\n🧹 All services stopped cleanly")


if __name__ == "__main__":
    execute_standalone_workflow()
