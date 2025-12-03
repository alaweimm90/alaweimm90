/**
 * ATLAS Google Adapter
 * Adapter for Gemini models via the Google AI API
 */

import { Agent } from '@atlas/types/index.js';
import { BaseAdapter, CompletionRequest, CompletionResponse, Message } from './base.js';

// ============================================================================
// Google AI API Types
// ============================================================================

interface GeminiContent {
  role: 'user' | 'model';
  parts: Array<{ text: string }>;
}

interface GeminiRequest {
  contents: GeminiContent[];
  systemInstruction?: {
    parts: Array<{ text: string }>;
  };
  generationConfig?: {
    maxOutputTokens?: number;
    temperature?: number;
    stopSequences?: string[];
  };
}

interface GeminiResponse {
  candidates: Array<{
    content: {
      parts: Array<{ text: string }>;
      role: 'model';
    };
    finishReason: 'STOP' | 'MAX_TOKENS' | 'SAFETY' | 'RECITATION' | 'OTHER';
  }>;
  usageMetadata: {
    promptTokenCount: number;
    candidatesTokenCount: number;
    totalTokenCount: number;
  };
}

interface GeminiError {
  error: {
    code: number;
    message: string;
    status: string;
  };
}

// ============================================================================
// Google Adapter Implementation
// ============================================================================

export class GoogleAdapter extends BaseAdapter {
  readonly provider = 'google';
  private apiKey: string | null = null;
  private baseUrl = 'https://generativelanguage.googleapis.com/v1beta';

  constructor(agent: Agent) {
    super(agent);
    this.apiKey = process.env[agent.config.apiKeyEnv || 'GOOGLE_API_KEY'] || null;
  }

  isConfigured(): boolean {
    return this.apiKey !== null && this.apiKey.length > 0;
  }

  async complete(request: CompletionRequest): Promise<CompletionResponse> {
    if (!this.isConfigured()) {
      throw new Error('Google API key not configured');
    }

    const startTime = Date.now();

    const geminiRequest = this.buildRequest(request);
    const response = await this.makeRequest(geminiRequest);
    const latency = Date.now() - startTime;

    const candidate = response.candidates[0];
    const content = candidate?.content?.parts?.map((p) => p.text).join('') || '';

    return {
      content,
      tokensUsed: {
        input: response.usageMetadata.promptTokenCount,
        output: response.usageMetadata.candidatesTokenCount,
        total: response.usageMetadata.totalTokenCount,
      },
      finishReason: this.mapFinishReason(candidate?.finishReason),
      model: this.agent.model,
      latency,
    };
  }

  /**
   * Build Gemini API request
   */
  private buildRequest(request: CompletionRequest): GeminiRequest {
    const { systemMessage, contents } = this.convertMessages(request.messages);

    const geminiRequest: GeminiRequest = {
      contents,
      generationConfig: {
        maxOutputTokens: request.maxTokens || this.agent.config.maxTokens,
        temperature: request.temperature ?? this.agent.config.temperature,
      },
    };

    if (systemMessage) {
      geminiRequest.systemInstruction = {
        parts: [{ text: systemMessage }],
      };
    }

    if (request.stopSequences?.length) {
      geminiRequest.generationConfig!.stopSequences = request.stopSequences;
    }

    return geminiRequest;
  }

  /**
   * Convert generic messages to Gemini format
   */
  private convertMessages(messages: Message[]): {
    systemMessage: string | null;
    contents: GeminiContent[];
  } {
    let systemMessage: string | null = null;
    const contents: GeminiContent[] = [];

    for (const msg of messages) {
      if (msg.role === 'system') {
        systemMessage = msg.content;
      } else {
        contents.push({
          role: msg.role === 'assistant' ? 'model' : 'user',
          parts: [{ text: msg.content }],
        });
      }
    }

    return { systemMessage, contents };
  }

  /**
   * Make HTTP request to Google AI API
   */
  private async makeRequest(request: GeminiRequest): Promise<GeminiResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.agent.config.timeout);

    // Gemini uses model name in URL
    const modelName = this.agent.model.includes('/')
      ? this.agent.model
      : `models/${this.agent.model}`;
    const url = `${this.baseUrl}/${modelName}:generateContent?key=${this.apiKey}`;

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorBody = (await response.json()) as GeminiError;
        throw new Error(`Google AI API error: ${errorBody.error?.message || response.statusText}`);
      }

      return (await response.json()) as GeminiResponse;
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Request timed out after ${this.agent.config.timeout}ms`);
      }

      throw error;
    }
  }

  /**
   * Map Gemini finish reason to generic format
   */
  private mapFinishReason(
    reason?: 'STOP' | 'MAX_TOKENS' | 'SAFETY' | 'RECITATION' | 'OTHER'
  ): CompletionResponse['finishReason'] {
    switch (reason) {
      case 'STOP':
        return 'end_turn';
      case 'MAX_TOKENS':
        return 'max_tokens';
      case 'SAFETY':
      case 'RECITATION':
      case 'OTHER':
        return 'error';
      default:
        return 'end_turn';
    }
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export function createGoogleAdapter(agent: Agent): GoogleAdapter {
  return new GoogleAdapter(agent);
}
