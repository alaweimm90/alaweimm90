import React from 'react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { HelpCircle } from 'lucide-react';
import { InlineMath } from '@/components/ui/Math';

interface ParameterTooltipProps {
  name: string;
  description: string;
  formula?: string;
  units?: string;
  physicsContext?: string;
  range?: { min: number; max: number };
  defaultValue?: number;
  children: React.ReactNode;
}

export const ParameterTooltip: React.FC<ParameterTooltipProps> = ({
  name,
  description,
  formula,
  units,
  physicsContext,
  range,
  defaultValue,
  children
}) => {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="flex items-center gap-2 cursor-help">
            {children}
            <HelpCircle className="w-4 h-4 text-muted-foreground hover:text-foreground transition-colors" />
          </div>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs p-4 space-y-2">
          <div className="font-semibold text-foreground">{name}</div>
          <div className="text-sm text-muted-foreground">{description}</div>
          
          {formula && (
            <div className="py-2 px-3 bg-muted/50 rounded border">
              <InlineMath math={formula} />
            </div>
          )}
          
          {units && (
            <div className="text-xs">
              <span className="font-medium">Units:</span> {units}
            </div>
          )}
          
          {range && (
            <div className="text-xs">
              <span className="font-medium">Range:</span> {range.min} - {range.max}
              {defaultValue !== undefined && (
                <span className="ml-2 text-muted-foreground">(default: {defaultValue})</span>
              )}
            </div>
          )}
          
          {physicsContext && (
            <div className="text-xs p-2 bg-accent/20 rounded border border-accent/30">
              <span className="font-medium">Physics:</span> {physicsContext}
            </div>
          )}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};