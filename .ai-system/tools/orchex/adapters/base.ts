/**
 * ORCHEX Base Adapter
 * Common interface for all LLM provider adapters
 */

import { Agent, Task, TaskResult } from '@ORCHEX/types/index.js';

// ============================================================================
// Message Types
// ============================================================================

export type MessageRole = 'system' | 'user' | 'assistant';

export interface Message {
  role: MessageRole;
  content: string;
}

export interface CompletionRequest {
  messages: Message[];
  maxTokens?: number;
  temperature?: number;
  stopSequences?: string[];
  stream?: boolean;
}

export interface CompletionResponse {
  content: string;
  tokensUsed: {
    input: number;
    output: number;
    total: number;
  };
  finishReason: 'end_turn' | 'max_tokens' | 'stop_sequence' | 'error';
  model: string;
  latency: number;
}

// ============================================================================
// Adapter Interface
// ============================================================================

export interface LLMAdapter {
  /**
   * Provider identifier
   */
  readonly provider: string;

  /**
   * Check if the adapter is configured (API key present)
   */
  isConfigured(): boolean;

  /**
   * Send a completion request
   */
  complete(request: CompletionRequest): Promise<CompletionResponse>;

  /**
   * Execute a task using this adapter
   */
  executeTask(task: Task): Promise<TaskResult>;

  /**
   * Get current rate limit status
   */
  getRateLimitStatus(): RateLimitStatus;
}

export interface RateLimitStatus {
  remaining: number;
  limit: number;
  resetAt?: Date;
  isLimited: boolean;
}

// ============================================================================
// Base Adapter Implementation
// ============================================================================

export abstract class BaseAdapter implements LLMAdapter {
  protected agent: Agent;
  protected rateLimitStatus: RateLimitStatus = {
    remaining: 1000,
    limit: 1000,
    isLimited: false,
  };

  abstract readonly provider: string;

  constructor(agent: Agent) {
    this.agent = agent;
  }

  abstract isConfigured(): boolean;
  abstract complete(request: CompletionRequest): Promise<CompletionResponse>;

  /**
   * Default task execution - converts task to messages and calls complete
   */
  async executeTask(task: Task): Promise<TaskResult> {
    const startTime = Date.now();

    try {
      const messages = this.taskToMessages(task);
      const response = await this.complete({
        messages,
        maxTokens: this.agent.config.maxTokens,
        temperature: this.agent.config.temperature,
      });

      return {
        success: true,
        output: response.content,
        tokensUsed: response.tokensUsed.total,
        latency: response.latency,
        agentId: this.agent.id,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        latency: Date.now() - startTime,
        agentId: this.agent.id,
      };
    }
  }

  getRateLimitStatus(): RateLimitStatus {
    return { ...this.rateLimitStatus };
  }

  /**
   * Convert a task to messages for the LLM
   */
  protected taskToMessages(task: Task): Message[] {
    const messages: Message[] = [];

    // System message based on task type
    const systemPrompt = this.getSystemPromptForTaskType(task.type);
    if (systemPrompt) {
      messages.push({ role: 'system', content: systemPrompt });
    }

    // Build user message from task
    let userContent = task.description;

    if (task.context.codeSnippet) {
      userContent += `\n\nCode:\n\`\`\`${task.context.language || ''}\n${task.context.codeSnippet}\n\`\`\``;
    }

    if (task.context.files?.length) {
      userContent += `\n\nRelevant files: ${task.context.files.join(', ')}`;
    }

    if (task.context.additionalContext) {
      userContent += `\n\nAdditional context: ${task.context.additionalContext}`;
    }

    messages.push({ role: 'user', content: userContent });

    return messages;
  }

  /**
   * Get system prompt for task type
   */
  protected getSystemPromptForTaskType(taskType: string): string {
    const prompts: Record<string, string> = {
      code_generation:
        'You are an expert software engineer. Generate clean, well-documented code that follows best practices.',
      code_review:
        'You are a senior code reviewer. Analyze the code for bugs, security issues, performance problems, and style violations. Be thorough but constructive.',
      refactoring:
        'You are a refactoring specialist. Improve code structure, readability, and maintainability while preserving functionality.',
      documentation:
        'You are a technical writer. Create clear, comprehensive documentation that helps developers understand and use the code.',
      testing:
        'You are a QA engineer. Write comprehensive tests that cover edge cases and ensure code reliability.',
      debugging:
        'You are a debugging expert. Analyze the problem, identify root causes, and provide clear solutions.',
      analysis:
        'You are a code analyst. Provide detailed analysis of code structure, complexity, and potential improvements.',
      explanation:
        'You are a patient teacher. Explain code concepts clearly, breaking down complex ideas into understandable parts.',
      chat: 'You are a helpful AI assistant for software development tasks.',
    };

    return prompts[taskType] || prompts.chat;
  }

  /**
   * Update rate limit status from response headers
   */
  protected updateRateLimitStatus(_headers: Record<string, string>): void {
    // Override in provider-specific adapters
  }
}
