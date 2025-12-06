import React from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Download, Settings, Info } from 'lucide-react';

interface EnhancedPlotHeaderProps {
  title: string;
  badgeLabel: string;
  badgeVariant: React.ComponentProps<typeof Badge>['variant'];
  validationInfo?: { isValid: boolean; message: string; details?: string[] };
  showControls: boolean;
  showExportOptions: boolean;
  onExport: () => void;
  onToggleControls: () => void;
}

export const EnhancedPlotHeader: React.FC<EnhancedPlotHeaderProps> = ({
  title,
  badgeLabel,
  badgeVariant,
  validationInfo,
  showControls,
  showExportOptions,
  onExport,
  onToggleControls,
}) => {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-3">
        <h3 className="text-lg font-semibold font-display text-foreground tracking-[0.005em]">{title}</h3>
        <Badge variant={badgeVariant}>{badgeLabel}</Badge>
        {validationInfo && (
          <Badge variant={validationInfo.isValid ? 'default' : 'destructive'}>
            <Info className="w-3 h-3 mr-1" />
            {validationInfo.isValid ? 'Validated' : 'Warning'}
          </Badge>
        )}
      </div>

      {showControls && (
        <div className="flex items-center gap-2">
          {showExportOptions && (
            <>
              <Button variant="outline" size="sm" onClick={onExport} className="h-8">
                <Download className="w-3 h-3 mr-1" />
                Data
              </Button>
              <Separator orientation="vertical" className="h-6" />
            </>
          )}
          <Button variant="outline" size="sm" onClick={onToggleControls} className="h-8">
            <Settings className="w-3 h-3 mr-1" />
            Controls
          </Button>
        </div>
      )}
    </div>
  );
};

export default EnhancedPlotHeader;
