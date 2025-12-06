/**
 * Claude API Client for Attributa AI Assistant
 * Handles communication with Anthropic's Claude API
 */

export interface ClaudeMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ClaudeStreamChunk {
  type: 'content_block_delta' | 'message_start' | 'message_stop';
  delta?: {
    type: 'text_delta';
    text: string;
  };
}

export interface ClaudeClientConfig {
  apiKey: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
}

export class ClaudeClient {
  private apiKey: string;
  private model: string;
  private maxTokens: number;
  private temperature: number;
  private baseUrl = 'https://api.anthropic.com/v1';

  constructor(config: ClaudeClientConfig) {
    this.apiKey = config.apiKey;
    this.model = config.model || 'claude-3-5-sonnet-20241022';
    this.maxTokens = config.maxTokens || 4096;
    this.temperature = config.temperature || 0.7;
  }

  /**
   * Send a message to Claude and get a complete response
   */
  async sendMessage(
    messages: ClaudeMessage[],
    systemPrompt?: string
  ): Promise<string> {
    try {
      const response = await fetch(`${this.baseUrl}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': this.apiKey,
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify({
          model: this.model,
          max_tokens: this.maxTokens,
          temperature: this.temperature,
          system: systemPrompt,
          messages,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(
          `Claude API error: ${error.error?.message || response.statusText}`
        );
      }

      const data = await response.json();
      return data.content[0]?.text || '';
    } catch (error) {
      console.error('Claude API error:', error);
      throw error;
    }
  }

  /**
   * Send a message to Claude and get a streaming response
   */
  async *streamMessage(
    messages: ClaudeMessage[],
    systemPrompt?: string
  ): AsyncGenerator<string, void, unknown> {
    try {
      const response = await fetch(`${this.baseUrl}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': this.apiKey,
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify({
          model: this.model,
          max_tokens: this.maxTokens,
          temperature: this.temperature,
          system: systemPrompt,
          messages,
          stream: true,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(
          `Claude API error: ${error.error?.message || response.statusText}`
        );
      }

      if (!response.body) {
        throw new Error('No response body for streaming');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;

            try {
              const parsed: ClaudeStreamChunk = JSON.parse(data);
              if (
                parsed.type === 'content_block_delta' &&
                parsed.delta?.type === 'text_delta'
              ) {
                yield parsed.delta.text;
              }
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }
    } catch (error) {
      console.error('Claude streaming error:', error);
      throw error;
    }
  }

  /**
   * Validate API key by making a test request
   */
  async validateApiKey(): Promise<boolean> {
    try {
      await this.sendMessage(
        [{ role: 'user', content: 'Hi' }],
        'You are a helpful assistant. Respond with just "Hello".'
      );
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Update client configuration
   */
  updateConfig(config: Partial<ClaudeClientConfig>) {
    if (config.apiKey) this.apiKey = config.apiKey;
    if (config.model) this.model = config.model;
    if (config.maxTokens) this.maxTokens = config.maxTokens;
    if (config.temperature !== undefined) this.temperature = config.temperature;
  }
}

/**
 * Singleton instance management
 */
let clientInstance: ClaudeClient | null = null;

export function getClaudeClient(config?: ClaudeClientConfig): ClaudeClient {
  if (!clientInstance && !config) {
    throw new Error('ClaudeClient not initialized. Provide config on first call.');
  }
  if (config) {
    clientInstance = new ClaudeClient(config);
  }
  return clientInstance!;
}

export function resetClaudeClient() {
  clientInstance = null;
}
