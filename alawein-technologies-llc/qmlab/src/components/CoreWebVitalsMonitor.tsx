import { useEffect } from 'react';
import { useCoreWebVitals } from '@/hooks/usePerformance';

interface CoreWebVitalsMonitorProps {
  onMetricsUpdate?: (metrics: any) => void;
}

export const CoreWebVitalsMonitor = ({ onMetricsUpdate }: CoreWebVitalsMonitorProps) => {
  const coreWebVitals = useCoreWebVitals();

  useEffect(() => {
    if (onMetricsUpdate) {
      onMetricsUpdate(coreWebVitals);
    }
  }, [coreWebVitals, onMetricsUpdate]);

  return null; // This component doesn't render anything
};