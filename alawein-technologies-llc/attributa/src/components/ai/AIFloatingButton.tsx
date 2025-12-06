/**
 * Floating AI Assistant Button
 */

import { Bot } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useAIAssistant } from '@/hooks/useAIAssistant';
import { cn } from '@/lib/utils';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';

interface AIFloatingButtonProps {
  className?: string;
}

export default function AIFloatingButton({ className }: AIFloatingButtonProps) {
  const { togglePanel, isPanelOpen, isReady } = useAIAssistant();

  if (!isReady) {
    // Don't show button if not configured
    return null;
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          onClick={togglePanel}
          className={cn(
            'fixed bottom-6 right-6 h-14 w-14 rounded-full shadow-lg',
            'hover:scale-110 transition-transform duration-200',
            'z-50',
            isPanelOpen && 'scale-95 opacity-80',
            className
          )}
          size="icon"
          aria-label="Toggle AI Assistant"
        >
          <Bot className="h-6 w-6" />
        </Button>
      </TooltipTrigger>
      <TooltipContent side="left">
        <p>AI Assistant (Cmd+K)</p>
      </TooltipContent>
    </Tooltip>
  );
}
