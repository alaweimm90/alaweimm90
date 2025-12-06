/**
 * Zustand store for AI Assistant state management
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { ClaudeMessage } from '@/services/ai/claudeClient';

export interface AIMessage extends ClaudeMessage {
  id: string;
  timestamp: number;
  isStreaming?: boolean;
  error?: string;
}

export interface AISettings {
  apiKey: string;
  model: string;
  temperature: number;
  maxTokens: number;
  enabled: boolean;
}

interface AIAssistantStore {
  // Settings
  settings: AISettings;
  updateSettings: (settings: Partial<AISettings>) => void;

  // Chat state
  messages: AIMessage[];
  isLoading: boolean;
  error: string | null;

  // Actions
  addMessage: (message: Omit<AIMessage, 'id' | 'timestamp'>) => void;
  updateMessage: (id: string, update: Partial<AIMessage>) => void;
  clearMessages: () => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;

  // Panel state
  isPanelOpen: boolean;
  setPanelOpen: (open: boolean) => void;
  togglePanel: () => void;

  // Context
  currentContext: {
    page?: string;
    documentId?: string;
    selectedSegment?: string;
  };
  setContext: (context: AIAssistantStore['currentContext']) => void;
}

const defaultSettings: AISettings = {
  apiKey: '',
  model: 'claude-3-5-sonnet-20241022',
  temperature: 0.7,
  maxTokens: 4096,
  enabled: false,
};

export const useAIAssistantStore = create<AIAssistantStore>()(
  persist(
    (set) => ({
      // Settings
      settings: defaultSettings,
      updateSettings: (newSettings) =>
        set((state) => ({
          settings: { ...state.settings, ...newSettings },
        })),

      // Chat state
      messages: [],
      isLoading: false,
      error: null,

      // Actions
      addMessage: (message) =>
        set((state) => ({
          messages: [
            ...state.messages,
            {
              ...message,
              id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
              timestamp: Date.now(),
            },
          ],
        })),

      updateMessage: (id, update) =>
        set((state) => ({
          messages: state.messages.map((msg) =>
            msg.id === id ? { ...msg, ...update } : msg
          ),
        })),

      clearMessages: () => set({ messages: [], error: null }),

      setLoading: (loading) => set({ isLoading: loading }),

      setError: (error) => set({ error }),

      // Panel state
      isPanelOpen: false,
      setPanelOpen: (open) => set({ isPanelOpen: open }),
      togglePanel: () => set((state) => ({ isPanelOpen: !state.isPanelOpen })),

      // Context
      currentContext: {},
      setContext: (context) => set({ currentContext: context }),
    }),
    {
      name: 'attributa-ai-assistant',
      partialize: (state) => ({
        settings: {
          ...state.settings,
          apiKey: state.settings.apiKey, // Persist API key
        },
        messages: state.messages.slice(-50), // Keep only last 50 messages
      }),
    }
  )
);

/**
 * Helper to check if AI assistant is configured and ready
 */
export function useAIAssistantReady(): boolean {
  const { settings } = useAIAssistantStore();
  return settings.enabled && settings.apiKey.length > 0;
}

/**
 * Helper to get conversation history in Claude API format
 */
export function getConversationHistory(store: AIAssistantStore): ClaudeMessage[] {
  return store.messages
    .filter((msg) => !msg.error && !msg.isStreaming)
    .map((msg) => ({
      role: msg.role,
      content: msg.content,
    }));
}
