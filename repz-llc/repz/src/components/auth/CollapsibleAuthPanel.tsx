import React from 'react';
import { ChevronDown } from 'lucide-react';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';

interface CollapsibleAuthPanelProps {
  isOpen: boolean;
  onToggle: () => void;
  title: string;
  children: React.ReactNode;
}

export const CollapsibleAuthPanel: React.FC<CollapsibleAuthPanelProps> = ({
  isOpen,
  onToggle,
  title,
  children
}) => {
  return (
    <Collapsible open={isOpen} onOpenChange={onToggle}>
      <div className="bg-card rounded-lg shadow-lg border border-border mb-4">
        <CollapsibleTrigger asChild>
          <button className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-muted/50 rounded-t-lg transition-colors" onClick={() => console.log("CollapsibleAuthPanel button clicked")}>
            <span className="font-medium text-foreground">{title}</span>
            <ChevronDown 
              className={`w-5 h-5 transform transition-transform duration-200 ${
                isOpen ? 'rotate-180' : ''
              }`}
            />
          </button>
        </CollapsibleTrigger>
        
        <CollapsibleContent>
          <div className="p-4 border-t border-border">
            {children}
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
};