import { useToast } from '@/hooks/use-toast';
import { CheckCircle, XCircle, Clock, Play, Pause, BarChart3, Zap } from 'lucide-react';
import { trackQuantumEvents } from '@/lib/analytics';

export interface ToastOptions {
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export const useStatusToasts = () => {
  const { toast } = useToast();

  const showTrainingStart = (dataset: string, epochs: number, options?: ToastOptions) => {
    toast({
      title: "Training Started! 🚀",
      description: `Training on ${dataset} for ${epochs} epochs`,
      duration: options?.duration || 3000,
      action: options?.action ? (
          <button
            onClick={options.action.onClick}
            className="inline-flex h-8 shrink-0 items-center justify-center rounded-md bg-transparent px-3 text-sm font-medium transition-colors hover:bg-secondary focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
          >
            {options.action.label}
          </button>
      ) : undefined,
    });
    
    // Track the toast display
    trackQuantumEvents.featureDiscovery('training_start_toast', 'status_feedback');
  };

  const showTrainingComplete = (dataset: string, finalLoss: number, epochs: number, options?: ToastOptions) => {
    toast({
      title: "Training Complete! ✅",
      description: `Finished training on ${dataset}. Final loss: ${finalLoss.toFixed(4)}`,
      duration: options?.duration || 5000,
      action: options?.action ? (
          <button
            onClick={options.action.onClick}
            className="inline-flex h-8 shrink-0 items-center justify-center rounded-md bg-transparent px-3 text-sm font-medium transition-colors hover:bg-secondary focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
          >
            <CheckCircle className="mr-1 h-4 w-4" />
            {options.action.label}
          </button>
      ) : undefined,
    });
    
    trackQuantumEvents.featureDiscovery('training_complete_toast', 'status_feedback');
  };

  const showTrainingPaused = (currentEpoch: number, totalEpochs: number, options?: ToastOptions) => {
    toast({
      title: "Training Paused ⏸️",
      description: `Paused at epoch ${currentEpoch} of ${totalEpochs}`,
      duration: options?.duration || 3000,
      action: options?.action ? (
          <button
            onClick={options.action.onClick}
            className="inline-flex h-8 shrink-0 items-center justify-center rounded-md bg-transparent px-3 text-sm font-medium transition-colors hover:bg-secondary focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
          >
            <Play className="mr-1 h-4 w-4" />
            {options.action.label}
          </button>
      ) : undefined,
    });
    
    trackQuantumEvents.featureDiscovery('training_paused_toast', 'status_feedback');
  };

  const showTrainingResumed = (currentEpoch: number, totalEpochs: number, options?: ToastOptions) => {
    toast({
      title: "Training Resumed ▶️",
      description: `Continuing from epoch ${currentEpoch} of ${totalEpochs}`,
      duration: options?.duration || 2000,
    });
    
    trackQuantumEvents.featureDiscovery('training_resumed_toast', 'status_feedback');
  };

  const showTrainingError = (errorMessage: string, options?: ToastOptions) => {
    toast({
      variant: "destructive",
      title: "Training Error ❌",
      description: errorMessage,
      duration: options?.duration || 6000,
      action: options?.action ? (
          <button
            onClick={options.action.onClick}
            className="inline-flex h-8 shrink-0 items-center justify-center rounded-md bg-transparent px-3 text-sm font-medium text-destructive-foreground transition-colors hover:bg-destructive/10 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
          >
            {options.action.label}
          </button>
      ) : undefined,
    });
    
    trackQuantumEvents.featureDiscovery('training_error_toast', 'error_feedback');
  };

  const showCircuitChanged = (gateCount: number, qubitCount: number, options?: ToastOptions) => {
    if (gateCount === 0) {
      toast({
        title: "Circuit Cleared 🧹",
        description: "All gates removed from the circuit",
        duration: options?.duration || 2000,
      });
    } else {
      toast({
        title: "Circuit Updated ⚡",
        description: `${gateCount} gates on ${qubitCount} qubits`,
        duration: options?.duration || 2000,
      });
    }
    
    trackQuantumEvents.featureDiscovery('circuit_change_toast', 'status_feedback');
  };

  const showQuantumStateChange = (message: string, options?: ToastOptions) => {
    toast({
      title: "Quantum State Updated 🌀",
      description: message,
      duration: options?.duration || 2000,
    });
    
    trackQuantumEvents.featureDiscovery('quantum_state_toast', 'visualization_feedback');
  };

  const showFeatureComingSoon = (featureName: string, options?: ToastOptions) => {
    toast({
      title: `${featureName} Coming Soon! 🚧`,
      description: "This feature is under development. Stay tuned!",
      duration: options?.duration || 3000,
    });
    
    trackQuantumEvents.featureDiscovery('coming_soon_toast', 'feature_request');
  };

  const showSuccessMessage = (title: string, description?: string, options?: ToastOptions) => {
    toast({
      title: `${title} ✅`,
      description: description,
      duration: options?.duration || 3000,
      action: options?.action ? (
          <button
            onClick={options.action.onClick}
            className="inline-flex h-8 shrink-0 items-center justify-center rounded-md bg-transparent px-3 text-sm font-medium transition-colors hover:bg-secondary focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
          >
            {options.action.label}
          </button>
      ) : undefined,
    });
    
    trackQuantumEvents.featureDiscovery('success_toast', 'positive_feedback');
  };

  const showInfoMessage = (title: string, description?: string, options?: ToastOptions) => {
    toast({
      title: `${title} ℹ️`,
      description: description,
      duration: options?.duration || 4000,
      action: options?.action ? (
          <button
            onClick={options.action.onClick}
            className="inline-flex h-8 shrink-0 items-center justify-center rounded-md bg-transparent px-3 text-sm font-medium transition-colors hover:bg-secondary focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
          >
            {options.action.label}
          </button>
      ) : undefined,
    });
    
    trackQuantumEvents.featureDiscovery('info_toast', 'informational_feedback');
  };

  return {
    showTrainingStart,
    showTrainingComplete,
    showTrainingPaused,
    showTrainingResumed,
    showTrainingError,
    showCircuitChanged,
    showQuantumStateChange,
    showFeatureComingSoon,
    showSuccessMessage,
    showInfoMessage
  };
};