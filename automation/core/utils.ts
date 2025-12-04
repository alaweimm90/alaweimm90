import { RouteResult } from '../types';

/**
 * Route patterns for task classification
 */
interface RoutePattern {
  keywords: string[];
  type: string;
  tools: string[];
  agents: string[];
}

/**
 * Get predefined route patterns for task classification
 */
function getRoutePatterns(): RoutePattern[] {
  return [
    {
      keywords: ['debug', 'fix', 'error', 'bug', 'issue'],
      type: 'debugging',
      tools: ['cline', 'cursor', 'claude_code'],
      agents: ['debugger_agent', 'coder_agent'],
    },
    {
      keywords: ['implement', 'create', 'build', 'develop', 'feature'],
      type: 'development',
      tools: ['cursor', 'claude_code', 'copilot'],
      agents: ['coder_agent', 'architect_agent'],
    },
    {
      keywords: ['refactor', 'clean', 'optimize', 'improve'],
      type: 'refactoring',
      tools: ['kilo_code', 'cursor'],
      agents: ['coder_agent', 'reviewer_agent'],
    },
    {
      keywords: ['review', 'check', 'audit', 'analyze'],
      type: 'review',
      tools: ['claude_code', 'cline'],
      agents: ['reviewer_agent', 'critic_agent'],
    },
    {
      keywords: ['test', 'spec', 'coverage'],
      type: 'testing',
      tools: ['cursor', 'copilot'],
      agents: ['qa_engineer_agent', 'coder_agent'],
    },
    {
      keywords: ['document', 'readme', 'docs', 'explain'],
      type: 'documentation',
      tools: ['claude_code', 'cursor'],
      agents: ['writer_agent', 'technical_writer_agent'],
    },
    {
      keywords: ['deploy', 'release', 'ci', 'cd', 'pipeline'],
      type: 'devops',
      tools: ['cline', 'claude_code'],
      agents: ['devops_agent', 'mlops_agent'],
    },
    {
      keywords: ['research', 'investigate', 'explore', 'study'],
      type: 'research',
      tools: ['claude_code', 'perplexity'],
      agents: ['scientist_agent', 'scout_agent'],
    },
  ];
}

/**
 * Calculate confidence score for pattern matching
 */
function patternConfidence(pattern: RoutePattern, query: string): number {
  const matches = pattern.keywords.filter((keyword) => query.includes(keyword)).length;
  return matches > 0 ? Math.min(0.5 + matches * 0.2, 1.0) : 0;
}

/**
 * Choose the best matching pattern for a query
 */
function chooseBest(
  query: string,
  patterns: RoutePattern[]
): RoutePattern & { confidence: number } {
  let best: RoutePattern & { confidence: number } = {
    type: 'general',
    confidence: 0.3,
    tools: ['cursor', 'copilot'],
    agents: ['coder_agent'],
    keywords: [],
  };

  for (const pattern of patterns) {
    const confidence = patternConfidence(pattern, query);
    if (confidence > best.confidence) {
      best = { ...pattern, confidence };
    }
  }

  return best;
}

/**
 * Route a task based on natural language description
 */
export function routeTask(description: string): RouteResult {
  const query = description.toLowerCase();
  const patterns = getRoutePatterns();
  const best = chooseBest(query, patterns);

  return {
    task_type: best.type,
    confidence: best.confidence,
    recommended_tools: best.tools,
    suggested_agents: best.agents,
  };
}
