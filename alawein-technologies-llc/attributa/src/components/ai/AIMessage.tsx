/**
 * AI Message Bubble Component
 */

import { memo } from 'react';
import { Bot, User, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { AIMessage as AIMessageType } from '@/store/aiAssistantStore';
import ReactMarkdown from 'react-markdown';

interface AIMessageProps {
  message: AIMessageType;
  className?: string;
}

export const AIMessage = memo(function AIMessage({ message, className }: AIMessageProps) {
  const isUser = message.role === 'user';
  const hasError = !!message.error;

  return (
    <div
      className={cn(
        'flex gap-3 animate-in fade-in slide-in-from-bottom-2 duration-300',
        isUser ? 'flex-row-reverse' : 'flex-row',
        className
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          'flex-shrink-0 h-8 w-8 rounded-full flex items-center justify-center',
          isUser
            ? 'bg-primary text-primary-foreground'
            : hasError
            ? 'bg-destructive text-destructive-foreground'
            : 'bg-muted text-muted-foreground'
        )}
      >
        {isUser ? (
          <User className="h-4 w-4" />
        ) : hasError ? (
          <AlertCircle className="h-4 w-4" />
        ) : (
          <Bot className="h-4 w-4" />
        )}
      </div>

      {/* Message Content */}
      <div
        className={cn(
          'flex-1 max-w-[80%] space-y-2',
          isUser ? 'text-right' : 'text-left'
        )}
      >
        {/* Message Bubble */}
        <div
          className={cn(
            'inline-block px-4 py-2 rounded-lg',
            isUser
              ? 'bg-primary text-primary-foreground'
              : hasError
              ? 'bg-destructive/10 text-destructive border border-destructive/20'
              : 'bg-muted text-foreground'
          )}
        >
          {hasError ? (
            <div className="text-sm">
              <strong>Error:</strong> {message.error}
            </div>
          ) : message.isStreaming && message.content === '' ? (
            <div className="flex items-center gap-2 text-sm">
              <div className="flex gap-1">
                <span className="animate-pulse">●</span>
                <span className="animate-pulse delay-75">●</span>
                <span className="animate-pulse delay-150">●</span>
              </div>
              <span className="text-muted-foreground">Thinking...</span>
            </div>
          ) : (
            <div className="prose prose-sm dark:prose-invert max-w-none">
              {isUser ? (
                <p className="whitespace-pre-wrap">{message.content}</p>
              ) : (
                <ReactMarkdown
                  components={{
                    // Custom rendering for code blocks
                    code: ({ inline, ...props }) =>
                      inline ? (
                        <code
                          className="px-1 py-0.5 rounded bg-muted text-sm font-mono"
                          {...props}
                        />
                      ) : (
                        <pre className="bg-muted p-3 rounded-md overflow-x-auto">
                          <code className="text-sm font-mono" {...props} />
                        </pre>
                      ),
                    // Style links
                    a: (props) => (
                      <a
                        className="text-primary hover:underline"
                        target="_blank"
                        rel="noopener noreferrer"
                        {...props}
                      />
                    ),
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              )}
            </div>
          )}
        </div>

        {/* Timestamp */}
        <div
          className={cn(
            'text-xs text-muted-foreground px-1',
            isUser ? 'text-right' : 'text-left'
          )}
        >
          {new Date(message.timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          })}
          {message.isStreaming && ' • Streaming...'}
        </div>
      </div>
    </div>
  );
});
