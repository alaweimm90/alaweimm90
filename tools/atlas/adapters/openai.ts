/**
 * ATLAS OpenAI Adapter
 * Adapter for GPT models via the OpenAI API
 */

import { Agent } from '../types/index.js';
import { BaseAdapter, CompletionRequest, CompletionResponse, Message } from './base.js';

// ============================================================================
// OpenAI API Types
// ============================================================================

interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface OpenAIRequest {
  model: string;
  messages: OpenAIMessage[];
  max_tokens?: number;
  temperature?: number;
  stop?: string[];
}

interface OpenAIResponse {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: 'assistant';
      content: string;
    };
    finish_reason: 'stop' | 'length' | 'content_filter';
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface OpenAIError {
  error: {
    message: string;
    type: string;
    code: string;
  };
}

// ============================================================================
// OpenAI Adapter Implementation
// ============================================================================

export class OpenAIAdapter extends BaseAdapter {
  readonly provider = 'openai';
  private apiKey: string | null = null;
  private baseUrl = 'https://api.openai.com/v1';

  constructor(agent: Agent) {
    super(agent);
    this.apiKey = process.env[agent.config.apiKeyEnv || 'OPENAI_API_KEY'] || null;
  }

  isConfigured(): boolean {
    return this.apiKey !== null && this.apiKey.length > 0;
  }

  async complete(request: CompletionRequest): Promise<CompletionResponse> {
    if (!this.isConfigured()) {
      throw new Error('OpenAI API key not configured');
    }

    const startTime = Date.now();

    const openaiRequest: OpenAIRequest = {
      model: this.agent.model,
      messages: this.convertMessages(request.messages),
      max_tokens: request.maxTokens || this.agent.config.maxTokens,
      temperature: request.temperature ?? this.agent.config.temperature,
    };

    if (request.stopSequences?.length) {
      openaiRequest.stop = request.stopSequences;
    }

    const response = await this.makeRequest(openaiRequest);
    const latency = Date.now() - startTime;

    const choice = response.choices[0];

    return {
      content: choice?.message?.content || '',
      tokensUsed: {
        input: response.usage.prompt_tokens,
        output: response.usage.completion_tokens,
        total: response.usage.total_tokens,
      },
      finishReason: this.mapFinishReason(choice?.finish_reason),
      model: response.model,
      latency,
    };
  }

  /**
   * Convert generic messages to OpenAI format
   */
  private convertMessages(messages: Message[]): OpenAIMessage[] {
    return messages.map((msg) => ({
      role: msg.role,
      content: msg.content,
    }));
  }

  /**
   * Make HTTP request to OpenAI API
   */
  private async makeRequest(request: OpenAIRequest): Promise<OpenAIResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.agent.config.timeout);

    try {
      const response = await fetch(`${this.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // Update rate limit status from headers
      this.updateRateLimitFromHeaders(response.headers);

      if (!response.ok) {
        const errorBody = (await response.json()) as OpenAIError;
        throw new Error(`OpenAI API error: ${errorBody.error?.message || response.statusText}`);
      }

      return (await response.json()) as OpenAIResponse;
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Request timed out after ${this.agent.config.timeout}ms`);
      }

      throw error;
    }
  }

  /**
   * Update rate limit status from response headers
   */
  private updateRateLimitFromHeaders(headers: Headers): void {
    const remaining = headers.get('x-ratelimit-remaining-requests');
    const limit = headers.get('x-ratelimit-limit-requests');
    const reset = headers.get('x-ratelimit-reset-requests');

    if (remaining !== null) {
      this.rateLimitStatus.remaining = parseInt(remaining, 10);
    }
    if (limit !== null) {
      this.rateLimitStatus.limit = parseInt(limit, 10);
    }
    if (reset !== null) {
      // OpenAI uses relative time format like "1s" or "1m"
      const seconds = this.parseResetTime(reset);
      this.rateLimitStatus.resetAt = new Date(Date.now() + seconds * 1000);
    }

    this.rateLimitStatus.isLimited = this.rateLimitStatus.remaining === 0;
  }

  /**
   * Parse OpenAI reset time format
   */
  private parseResetTime(reset: string): number {
    const match = reset.match(/(\d+)(ms|s|m|h)/);
    if (!match) return 60;

    const value = parseInt(match[1], 10);
    const unit = match[2];

    switch (unit) {
      case 'ms':
        return value / 1000;
      case 's':
        return value;
      case 'm':
        return value * 60;
      case 'h':
        return value * 3600;
      default:
        return 60;
    }
  }

  /**
   * Map OpenAI finish reason to generic format
   */
  private mapFinishReason(
    reason?: 'stop' | 'length' | 'content_filter'
  ): CompletionResponse['finishReason'] {
    switch (reason) {
      case 'stop':
        return 'end_turn';
      case 'length':
        return 'max_tokens';
      case 'content_filter':
        return 'error';
      default:
        return 'end_turn';
    }
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export function createOpenAIAdapter(agent: Agent): OpenAIAdapter {
  return new OpenAIAdapter(agent);
}
