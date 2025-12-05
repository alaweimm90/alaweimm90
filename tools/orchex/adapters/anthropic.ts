/**
 * ORCHEX Anthropic Adapter
 * Adapter for Claude models via the Anthropic API
 */

import { Agent } from '@ORCHEX/types/index.js';
import { BaseAdapter, CompletionRequest, CompletionResponse, Message } from './base.js';

// ============================================================================
// Anthropic API Types
// ============================================================================

interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface AnthropicRequest {
  model: string;
  max_tokens: number;
  messages: AnthropicMessage[];
  system?: string;
  temperature?: number;
  stop_sequences?: string[];
}

interface AnthropicResponse {
  id: string;
  type: 'message';
  role: 'assistant';
  content: Array<{ type: 'text'; text: string }>;
  model: string;
  stop_reason: 'end_turn' | 'max_tokens' | 'stop_sequence';
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
}

interface AnthropicError {
  type: 'error';
  error: {
    type: string;
    message: string;
  };
}

// ============================================================================
// Anthropic Adapter Implementation
// ============================================================================

export class AnthropicAdapter extends BaseAdapter {
  readonly provider = 'anthropic';
  private apiKey: string | null = null;
  private baseUrl = 'https://api.anthropic.com/v1';

  constructor(agent: Agent) {
    super(agent);
    this.apiKey = process.env[agent.config.apiKeyEnv || 'ANTHROPIC_API_KEY'] || null;
  }

  isConfigured(): boolean {
    return this.apiKey !== null && this.apiKey.length > 0;
  }

  async complete(request: CompletionRequest): Promise<CompletionResponse> {
    if (!this.isConfigured()) {
      throw new Error('Anthropic API key not configured');
    }

    const startTime = Date.now();

    // Extract system message and convert messages
    const { systemMessage, messages } = this.convertMessages(request.messages);

    const anthropicRequest: AnthropicRequest = {
      model: this.agent.model,
      max_tokens: request.maxTokens || this.agent.config.maxTokens,
      messages,
      temperature: request.temperature ?? this.agent.config.temperature,
    };

    if (systemMessage) {
      anthropicRequest.system = systemMessage;
    }

    if (request.stopSequences?.length) {
      anthropicRequest.stop_sequences = request.stopSequences;
    }

    const response = await this.makeRequest(anthropicRequest);
    const latency = Date.now() - startTime;

    return {
      content: response.content.map((c) => c.text).join(''),
      tokensUsed: {
        input: response.usage.input_tokens,
        output: response.usage.output_tokens,
        total: response.usage.input_tokens + response.usage.output_tokens,
      },
      finishReason: this.mapStopReason(response.stop_reason),
      model: response.model,
      latency,
    };
  }

  /**
   * Convert generic messages to Anthropic format
   */
  private convertMessages(messages: Message[]): {
    systemMessage: string | null;
    messages: AnthropicMessage[];
  } {
    let systemMessage: string | null = null;
    const anthropicMessages: AnthropicMessage[] = [];

    for (const msg of messages) {
      if (msg.role === 'system') {
        systemMessage = msg.content;
      } else {
        anthropicMessages.push({
          role: msg.role as 'user' | 'assistant',
          content: msg.content,
        });
      }
    }

    return { systemMessage, messages: anthropicMessages };
  }

  /**
   * Make HTTP request to Anthropic API
   */
  private async makeRequest(request: AnthropicRequest): Promise<AnthropicResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.agent.config.timeout);

    try {
      const response = await fetch(`${this.baseUrl}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': this.apiKey!,
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // Update rate limit status from headers
      this.updateRateLimitFromHeaders(response.headers);

      if (!response.ok) {
        const errorBody = (await response.json()) as AnthropicError;
        throw new Error(`Anthropic API error: ${errorBody.error?.message || response.statusText}`);
      }

      return (await response.json()) as AnthropicResponse;
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
    const remaining = headers.get('anthropic-ratelimit-requests-remaining');
    const limit = headers.get('anthropic-ratelimit-requests-limit');
    const reset = headers.get('anthropic-ratelimit-requests-reset');

    if (remaining !== null) {
      this.rateLimitStatus.remaining = parseInt(remaining, 10);
    }
    if (limit !== null) {
      this.rateLimitStatus.limit = parseInt(limit, 10);
    }
    if (reset !== null) {
      this.rateLimitStatus.resetAt = new Date(reset);
    }

    this.rateLimitStatus.isLimited = this.rateLimitStatus.remaining === 0;
  }

  /**
   * Map Anthropic stop reason to generic format
   */
  private mapStopReason(
    reason: 'end_turn' | 'max_tokens' | 'stop_sequence'
  ): CompletionResponse['finishReason'] {
    return reason;
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export function createAnthropicAdapter(agent: Agent): AnthropicAdapter {
  return new AnthropicAdapter(agent);
}
