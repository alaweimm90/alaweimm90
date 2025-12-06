import React from 'react';

interface EnhancedPlotFooterProps {
  traceCount: number;
}

export const EnhancedPlotFooter: React.FC<EnhancedPlotFooterProps> = ({ traceCount }) => {
  return (
    <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t border-border/30">
      <div className="flex items-center gap-4">
        <span>SimCore Scientific Plot v2.0</span>
        <span>•</span>
        <span>{traceCount} traces displayed</span>
      </div>
      <div className="flex items-center gap-2">
        <span>Theme-aware</span>
        <span>•</span>
        <span>Publication ready</span>
      </div>
    </div>
  );
};

export default EnhancedPlotFooter;
