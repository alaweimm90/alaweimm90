/**
 * Quick Export Toolbar Component
 * Provides quick access to export functionality
 */

import React from 'react';
import { Button } from '@/components/ui/button';
import { Download, FileText, Image, Database } from 'lucide-react';

interface QuickExportToolbarProps {
  onExport?: (format: 'png' | 'svg' | 'json' | 'csv') => void;
  className?: string;
}

export function QuickExportToolbar({ onExport, className }: QuickExportToolbarProps) {
  return (
    <div className={`flex items-center gap-spacing-xs ${className}`}>
      <Button
        variant="pill"
        size="sm"
        onClick={() => onExport?.('png')}
        className="flex items-center gap-1 min-h-touch"
      >
        <Image className="h-3 w-3" />
        PNG
      </Button>
      <Button
        variant="pill"
        size="sm"
        onClick={() => onExport?.('svg')}
        className="flex items-center gap-1 min-h-touch"
      >
        <FileText className="h-3 w-3" />
        SVG
      </Button>
      <Button
        variant="pill"
        size="sm"
        onClick={() => onExport?.('json')}
        className="flex items-center gap-1 min-h-touch"
      >
        <Database className="h-3 w-3" />
        JSON
      </Button>
      <Button
        variant="pill"
        size="sm"
        onClick={() => onExport?.('csv')}
        className="flex items-center gap-1 min-h-touch"
      >
        <Download className="h-3 w-3" />
        CSV
      </Button>
    </div>
  );
}

export default QuickExportToolbar;