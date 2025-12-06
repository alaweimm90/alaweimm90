/**
 * AI Assistant Side Panel with Slide Animation
 */

import { useEffect } from 'react';
import { useAIAssistant } from '@/hooks/useAIAssistant';
import AIAssistantChat from './AIAssistantChat';
import { cn } from '@/lib/utils';

export default function AIAssistantPanel() {
  const { isPanelOpen, setPanelOpen } = useAIAssistant();

  // Handle escape key to close panel
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isPanelOpen) {
        setPanelOpen(false);
      }
    };

    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [isPanelOpen, setPanelOpen]);

  // Handle Cmd+K / Ctrl+K to toggle panel
  useEffect(() => {
    const handleShortcut = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setPanelOpen(!isPanelOpen);
      }
    };

    window.addEventListener('keydown', handleShortcut);
    return () => window.removeEventListener('keydown', handleShortcut);
  }, [isPanelOpen, setPanelOpen]);

  return (
    <>
      {/* Backdrop */}
      {isPanelOpen && (
        <div
          className="fixed inset-0 bg-background/80 backdrop-blur-sm z-40 animate-in fade-in duration-200"
          onClick={() => setPanelOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Panel */}
      <div
        className={cn(
          'fixed top-0 right-0 h-full w-full md:w-[500px] bg-background border-l shadow-2xl z-50',
          'transition-transform duration-300 ease-in-out',
          isPanelOpen ? 'translate-x-0' : 'translate-x-full'
        )}
      >
        <AIAssistantChat className="h-full" />
      </div>
    </>
  );
}
