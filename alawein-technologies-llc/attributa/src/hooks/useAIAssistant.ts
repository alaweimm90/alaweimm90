/**
 * Custom hook for AI Assistant functionality
 */

import { useCallback } from 'react';
import { useAIAssistantStore, getConversationHistory } from '@/store/aiAssistantStore';
import { getClaudeClient, resetClaudeClient } from '@/services/ai/claudeClient';
import { buildContextualPrompt } from '@/services/ai/prompts';
import { useToast } from '@/hooks/use-toast';

export function useAIAssistant() {
  const store = useAIAssistantStore();
  const { toast } = useToast();

  /**
   * Initialize Claude client with current settings
   */
  const initializeClient = useCallback(() => {
    const { settings } = store;
    if (!settings.apiKey) {
      throw new Error('API key not configured');
    }

    resetClaudeClient();
    return getClaudeClient({
      apiKey: settings.apiKey,
      model: settings.model,
      maxTokens: settings.maxTokens,
      temperature: settings.temperature,
    });
  }, [store.settings]);

  /**
   * Send a message and get a response (non-streaming)
   */
  const sendMessage = useCallback(
    async (content: string) => {
      try {
        store.setLoading(true);
        store.setError(null);

        // Add user message
        store.addMessage({
          role: 'user',
          content,
        });

        // Initialize client
        const client = initializeClient();

        // Build system prompt with context
        const systemPrompt = buildContextualPrompt({
          currentPage: store.currentContext.page,
        });

        // Get conversation history
        const history = getConversationHistory(store);

        // Send to Claude
        const response = await client.sendMessage(history, systemPrompt);

        // Add assistant response
        store.addMessage({
          role: 'assistant',
          content: response,
        });

        return response;
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : 'Failed to send message';
        store.setError(errorMessage);
        store.addMessage({
          role: 'assistant',
          content: '',
          error: errorMessage,
        });
        toast({
          variant: 'destructive',
          title: 'AI Assistant Error',
          description: errorMessage,
        });
        throw error;
      } finally {
        store.setLoading(false);
      }
    },
    [store, initializeClient, toast]
  );

  /**
   * Send a message and get a streaming response
   */
  const sendStreamingMessage = useCallback(
    async (content: string) => {
      try {
        store.setLoading(true);
        store.setError(null);

        // Add user message
        store.addMessage({
          role: 'user',
          content,
        });

        // Initialize client
        const client = initializeClient();

        // Build system prompt with context
        const systemPrompt = buildContextualPrompt({
          currentPage: store.currentContext.page,
        });

        // Get conversation history
        const history = getConversationHistory(store);

        // Create placeholder for streaming message
        const streamingId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        store.addMessage({
          role: 'assistant',
          content: '',
          id: streamingId,
          timestamp: Date.now(),
          isStreaming: true,
        });

        // Stream response
        let fullResponse = '';
        for await (const chunk of client.streamMessage(history, systemPrompt)) {
          fullResponse += chunk;
          store.updateMessage(streamingId, {
            content: fullResponse,
          });
        }

        // Mark streaming complete
        store.updateMessage(streamingId, {
          isStreaming: false,
        });

        return fullResponse;
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : 'Failed to send message';
        store.setError(errorMessage);
        toast({
          variant: 'destructive',
          title: 'AI Assistant Error',
          description: errorMessage,
        });
        throw error;
      } finally {
        store.setLoading(false);
      }
    },
    [store, initializeClient, toast]
  );

  /**
   * Validate API key
   */
  const validateApiKey = useCallback(
    async (apiKey: string): Promise<boolean> => {
      try {
        resetClaudeClient();
        const client = getClaudeClient({
          apiKey,
          model: store.settings.model,
        });
        const isValid = await client.validateApiKey();
        if (isValid) {
          toast({
            title: 'API Key Valid',
            description: 'Successfully connected to Claude API',
          });
        } else {
          toast({
            variant: 'destructive',
            title: 'Invalid API Key',
            description: 'Could not authenticate with Claude API',
          });
        }
        return isValid;
      } catch (error) {
        toast({
          variant: 'destructive',
          title: 'Validation Failed',
          description: error instanceof Error ? error.message : 'Unknown error',
        });
        return false;
      }
    },
    [store.settings.model, toast]
  );

  /**
   * Quick action helpers
   */
  const explainScore = useCallback(
    async (score: number) => {
      const message = `Can you explain what my attribution score of ${(score * 100).toFixed(1)}% means?`;
      return sendStreamingMessage(message);
    },
    [sendStreamingMessage]
  );

  const explainMethod = useCallback(
    async (method: 'gltr' | 'detectgpt' | 'watermark') => {
      const methods = {
        gltr: 'Can you explain how GLTR analysis works?',
        detectgpt: 'Can you explain how DetectGPT analysis works?',
        watermark: 'Can you explain how watermark detection works?',
      };
      return sendStreamingMessage(methods[method]);
    },
    [sendStreamingMessage]
  );

  const getSuggestions = useCallback(async () => {
    const message = 'Based on my current analysis results, what should I do next?';
    return sendStreamingMessage(message);
  }, [sendStreamingMessage]);

  return {
    // State
    messages: store.messages,
    isLoading: store.isLoading,
    error: store.error,
    isPanelOpen: store.isPanelOpen,
    settings: store.settings,
    isReady: store.settings.enabled && store.settings.apiKey.length > 0,

    // Actions
    sendMessage,
    sendStreamingMessage,
    clearMessages: store.clearMessages,
    togglePanel: store.togglePanel,
    setPanelOpen: store.setPanelOpen,
    updateSettings: store.updateSettings,
    setContext: store.setContext,
    validateApiKey,

    // Quick actions
    explainScore,
    explainMethod,
    getSuggestions,
  };
}
