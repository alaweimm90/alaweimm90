import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { RefreshCw } from 'lucide-react';

interface AnalyzerRetryChipProps {
  onRetry: () => void;
  className?: string;
}

export const AnalyzerRetryChip = ({ onRetry, className }: AnalyzerRetryChipProps) => {
  return (
    <Badge 
      variant="destructive" 
      className={`inline-flex items-center gap-1 ${className}`}
      role="alert"
    >
      Analyzer retry
      <Button
        variant="ghost"
        size="sm"
        onClick={onRetry}
        className="h-4 w-4 p-0 hover:bg-transparent"
        aria-label="Retry analyzer"
      >
        <RefreshCw className="h-3 w-3" />
      </Button>
    </Badge>
  );
};