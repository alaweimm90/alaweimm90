#!/usr/bin/env python3
"""Workflow orchestration engine."""

import subprocess
import yaml
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dag import DAG, Step, StepStatus

class WorkflowEngine:
    """Execute workflows with dependency management."""
    
    def __init__(self, max_parallel: int = 4):
        self.max_parallel = max_parallel
        self.dag = DAG()
    
    def load_workflow(self, workflow_file: Path) -> None:
        """Load workflow from YAML file."""
        with open(workflow_file) as f:
            config = yaml.safe_load(f)
        
        for step_config in config['steps']:
            step = Step(
                id=step_config['id'],
                command=step_config['run'],
                depends_on=step_config.get('depends_on', [])
            )
            self.dag.add_step(step)
        
        if not self.dag.validate():
            raise ValueError("Workflow contains cycles")
    
    def execute(self, context: Dict = None) -> bool:
        """Execute workflow with parallel execution where possible."""
        context = context or {}
        
        print(f"[START] Executing workflow with {len(self.dag.steps)} steps")
        
        while not self.dag.is_complete():
            ready_steps = self.dag.get_ready_steps()
            
            if not ready_steps:
                if self.dag.has_failures():
                    print("[FAIL] Workflow failed")
                    return False
                break
            
            # Execute ready steps in parallel
            self._execute_parallel(ready_steps, context)
        
        success = not self.dag.has_failures()
        status = "SUCCESS" if success else "FAILED"
        print(f"[{status}] Workflow complete")
        
        return success
    
    def _execute_parallel(self, steps: List[Step], context: Dict):
        """Execute steps in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = {
                executor.submit(self._execute_step, step, context): step
                for step in steps
            }
            
            for future in as_completed(futures):
                step = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] Step {step.id} failed: {e}")
                    step.status = StepStatus.FAILED
    
    def _execute_step(self, step: Step, context: Dict):
        """Execute single step."""
        step.status = StepStatus.RUNNING
        print(f"[RUN] {step.id}")
        
        try:
            # Replace context variables in command
            command = self._replace_context(step.command, context)
            
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 min timeout
            )
            
            if result.returncode == 0:
                step.status = StepStatus.SUCCESS
                step.result = {
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': 0
                }
                print(f"[OK] {step.id}")
            else:
                step.status = StepStatus.FAILED
                step.result = {
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }
                print(f"[FAIL] {step.id}: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            step.status = StepStatus.FAILED
            print(f"[TIMEOUT] {step.id}")
        except Exception as e:
            step.status = StepStatus.FAILED
            print(f"[ERROR] {step.id}: {e}")
    
    def _replace_context(self, command: str, context: Dict) -> str:
        """Replace context variables in command."""
        for key, value in context.items():
            command = command.replace(f"${{{key}}}", str(value))
        return command
    
    def get_results(self) -> Dict:
        """Get execution results."""
        return {
            step_id: {
                'status': step.status.value,
                'result': step.result
            }
            for step_id, step in self.dag.steps.items()
        }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Execute workflow")
    parser.add_argument('workflow', help='Workflow YAML file')
    parser.add_argument('--context', '-c', action='append', help='Context variables (key=value)')
    parser.add_argument('--parallel', '-p', type=int, default=4, help='Max parallel steps')
    
    args = parser.parse_args()
    
    # Parse context
    context = {}
    if args.context:
        for item in args.context:
            key, value = item.split('=', 1)
            context[key] = value
    
    # Execute workflow
    engine = WorkflowEngine(max_parallel=args.parallel)
    engine.load_workflow(Path(args.workflow))
    success = engine.execute(context)
    
    # Print results
    results = engine.get_results()
    print("\n=== Results ===")
    for step_id, result in results.items():
        print(f"{step_id}: {result['status']}")
    
    exit(0 if success else 1)

if __name__ == '__main__':
    main()
