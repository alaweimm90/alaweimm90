/**
 * AI Assistant Chat Interface
 */

import { useState, useRef, useEffect } from 'react';
import { Bot, Send, Trash2, X, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { AIMessage } from './AIMessage';
import { useAIAssistant } from '@/hooks/useAIAssistant';
import { cn } from '@/lib/utils';

interface AIAssistantChatProps {
  className?: string;
}

const QUICK_PROMPTS = [
  { label: 'Explain my score', prompt: 'Can you explain what my attribution score means?' },
  { label: 'How does GLTR work?', prompt: 'Can you explain how GLTR analysis works in detail?' },
  { label: 'What should I do next?', prompt: 'Based on my results, what should I do next?' },
  { label: 'False positives?', prompt: 'What causes false positives in AI detection?' },
];

export default function AIAssistantChat({ className }: AIAssistantChatProps) {
  const {
    messages,
    isLoading,
    isReady,
    sendStreamingMessage,
    clearMessages,
    setPanelOpen,
  } = useAIAssistant();

  const [input, setInput] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isLoading || !isReady) return;

    const message = input.trim();
    setInput('');
    textareaRef.current?.focus();

    try {
      await sendStreamingMessage(message);
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  const handleQuickPrompt = (prompt: string) => {
    setInput(prompt);
    textareaRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  if (!isReady) {
    return (
      <div className={cn('flex flex-col h-full', className)}>
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <div className="flex items-center gap-2">
            <Bot className="h-5 w-5" />
            <h3 className="font-semibold">AI Assistant</h3>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setPanelOpen(false)}
            aria-label="Close assistant"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Not Configured Message */}
        <div className="flex-1 flex items-center justify-center p-6">
          <div className="text-center space-y-4 max-w-sm">
            <div className="mx-auto h-12 w-12 rounded-full bg-muted flex items-center justify-center">
              <Sparkles className="h-6 w-6 text-muted-foreground" />
            </div>
            <div>
              <h4 className="font-semibold mb-2">AI Assistant Not Configured</h4>
              <p className="text-sm text-muted-foreground mb-4">
                Configure your Claude API key in Settings to enable the AI assistant.
              </p>
              <Button
                onClick={() => {
                  setPanelOpen(false);
                  // Navigate to settings - you can use router here
                  window.location.hash = '#/settings';
                }}
              >
                Go to Settings
              </Button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
            <Bot className="h-4 w-4 text-primary" />
          </div>
          <div>
            <h3 className="font-semibold">AI Assistant</h3>
            <p className="text-xs text-muted-foreground">Powered by Claude</p>
          </div>
        </div>
        <div className="flex items-center gap-1">
          {messages.length > 0 && (
            <Button
              variant="ghost"
              size="icon"
              onClick={clearMessages}
              aria-label="Clear chat"
              title="Clear chat history"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setPanelOpen(false)}
            aria-label="Close assistant"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 p-4" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center space-y-4 max-w-sm">
              <div className="mx-auto h-12 w-12 rounded-full bg-muted flex items-center justify-center">
                <Bot className="h-6 w-6 text-muted-foreground" />
              </div>
              <div>
                <h4 className="font-semibold mb-2">How can I help?</h4>
                <p className="text-sm text-muted-foreground mb-4">
                  I can explain analysis results, answer questions about methodology, and
                  provide guidance on using Attributa effectively.
                </p>
              </div>
              {/* Quick Prompts */}
              <div className="flex flex-wrap gap-2 justify-center">
                {QUICK_PROMPTS.map((prompt) => (
                  <Button
                    key={prompt.label}
                    variant="outline"
                    size="sm"
                    onClick={() => handleQuickPrompt(prompt.prompt)}
                    className="text-xs"
                  >
                    {prompt.label}
                  </Button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((message) => (
              <AIMessage key={message.id} message={message} />
            ))}
          </div>
        )}
      </ScrollArea>

      {/* Input */}
      <div className="p-4 border-t">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask me anything..."
            className="min-h-[60px] resize-none"
            disabled={isLoading}
            rows={2}
          />
          <Button
            type="submit"
            size="icon"
            disabled={!input.trim() || isLoading}
            className="h-[60px] w-[60px]"
          >
            <Send className="h-4 w-4" />
          </Button>
        </form>
        <p className="text-xs text-muted-foreground mt-2">
          Press <kbd className="px-1 py-0.5 bg-muted rounded text-xs">Enter</kbd> to send,{' '}
          <kbd className="px-1 py-0.5 bg-muted rounded text-xs">Shift+Enter</kbd> for new
          line
        </p>
      </div>
    </div>
  );
}
