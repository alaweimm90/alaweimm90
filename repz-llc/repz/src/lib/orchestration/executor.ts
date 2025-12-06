/**
 * Workflow Executor Module
 * Handles execution of DAG-based workflows
 */

import { DAG, DAGNode, getTopologicalOrder } from './dag';

export interface ExecutionContext {
  dagId: string;
  startTime: Date;
  endTime?: Date;
  status: 'pending' | 'running' | 'completed' | 'failed';
  results: Map<string, unknown>;
  errors: Map<string, Error>;
}

export interface ExecutorOptions {
  maxConcurrency?: number;
  timeout?: number;
  retryCount?: number;
  onNodeStart?: (node: DAGNode) => void;
  onNodeComplete?: (node: DAGNode, result: unknown) => void;
  onNodeError?: (node: DAGNode, error: Error) => void;
}

export async function executeDAG(
  dag: DAG,
  handlers: Map<string, (node: DAGNode, context: ExecutionContext) => Promise<unknown>>,
  options: ExecutorOptions = {}
): Promise<ExecutionContext> {
  const context: ExecutionContext = {
    dagId: dag.id,
    startTime: new Date(),
    status: 'running',
    results: new Map(),
    errors: new Map(),
  };

  try {
    const order = getTopologicalOrder(dag);

    for (const nodeId of order) {
      const node = dag.nodes.find(n => n.id === nodeId);
      if (!node) continue;

      const handler = handlers.get(node.id) || handlers.get('default');
      if (!handler) {
        console.warn(`No handler for node ${node.id}`);
        continue;
      }

      try {
        options.onNodeStart?.(node);
        node.status = 'running';

        const result = await handler(node, context);
        context.results.set(node.id, result);
        node.status = 'completed';

        options.onNodeComplete?.(node, result);
      } catch (error) {
        node.status = 'failed';
        const err = error instanceof Error ? error : new Error(String(error));
        context.errors.set(node.id, err);
        options.onNodeError?.(node, err);

        // Continue or fail based on node criticality
        if (node.data && (node.data as { critical?: boolean }).critical) {
          throw err;
        }
      }
    }

    context.status = context.errors.size > 0 ? 'failed' : 'completed';
  } catch (error) {
    context.status = 'failed';
  } finally {
    context.endTime = new Date();
  }

  return context;
}

export function createExecutor(options: ExecutorOptions = {}) {
  return {
    execute: (
      dag: DAG,
      handlers: Map<string, (node: DAGNode, context: ExecutionContext) => Promise<unknown>>
    ) => executeDAG(dag, handlers, options),
  };
}

/**
 * Run an analytics plan (convenience wrapper for analytics DAGs)
 */
export async function runAnalyticsPlan(
  dag: DAG,
  dataFetcher: (metric: string, timeRange: { start: Date; end: Date }) => Promise<unknown>
): Promise<ExecutionContext> {
  const handlers = new Map<string, (node: DAGNode, context: ExecutionContext) => Promise<unknown>>();

  // Handler for collection nodes
  handlers.set('default', async (node, context) => {
    if (node.id.startsWith('collect-')) {
      const data = node.data as { metric: string; timeRange: { start: Date; end: Date } };
      return dataFetcher(data.metric, data.timeRange);
    }

    if (node.id === 'aggregate') {
      // Aggregate all collected data
      const collectedData: Record<string, unknown> = {};
      context.results.forEach((value, key) => {
        if (key.startsWith('collect-')) {
          const metric = key.replace('collect-', '');
          collectedData[metric] = value;
        }
      });
      return collectedData;
    }

    if (node.id === 'visualize') {
      // Return aggregated data for visualization
      return context.results.get('aggregate');
    }

    return null;
  });

  return executeDAG(dag, handlers);
}

export default {
  executeDAG,
  createExecutor,
  runAnalyticsPlan,
};
