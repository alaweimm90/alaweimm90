import { useState, useEffect, useCallback } from 'react';
import { notificationManager, NotificationManager } from '@/lib/notification-manager';
import { trackQuantumEvents } from '@/lib/analytics';
import { logger } from '@/lib/logger';

interface CircuitConfig {
  gates?: Array<Record<string, unknown>>;
  qubits?: number;
  [key: string]: unknown;
}

interface NotificationState {
  permission: NotificationPermission;
  isSupported: boolean;
  isReady: boolean;
  queueSize: number;
}

interface NotificationActions {
  requestPermission: () => Promise<boolean>;
  showTrainingComplete: (accuracy: number, duration: number, trainingId: string) => Promise<boolean>;
  showSimulationReady: (circuitId: string) => Promise<boolean>;
  showAchievement: (title: string, description: string) => Promise<boolean>;
  queueTraining: (circuitConfig: any, trainingParams: any) => Promise<string>;
  queueSimulation: (circuitConfig: any, simulationParams: any) => Promise<string>;
  queueSharing: (shareData: any) => Promise<string>;
}

export const useNotifications = (): NotificationState & NotificationActions => {
  const [state, setState] = useState<NotificationState>({
    permission: 'default',
    isSupported: 'Notification' in window,
    isReady: false,
    queueSize: 0
  });

  // Initialize notification manager
  useEffect(() => {
    const initNotifications = async () => {
      if (!state.isSupported) return;

      try {
        // Get service worker registration
        const registration = await navigator.serviceWorker.ready;
        const success = await notificationManager.init(registration);

        if (success) {
          updateState();
          trackQuantumEvents.featureDiscovery('notifications_initialized');
        }
      } catch (error) {
        logger.error('Failed to initialize notifications', { error });
        trackQuantumEvents.errorBoundary(
          'Notification initialization failed',
          (error as Error).stack || 'No stack trace',
          'notifications'
        );
      }
    };

    initNotifications();
  }, [state.isSupported]);

  // Update state from notification manager
  const updateState = useCallback(() => {
    setState(prev => ({
      ...prev,
      permission: Notification.permission,
      isReady: notificationManager.isReady,
      queueSize: notificationManager.queueSize
    }));
  }, []);

  // Request notification permission
  const requestPermission = useCallback(async (): Promise<boolean> => {
    try {
      const granted = await notificationManager.requestPermission();
      updateState();
      
      if (granted) {
        trackQuantumEvents.featureDiscovery('notification_permission_requested_granted');
      } else {
        trackQuantumEvents.featureDiscovery('notification_permission_requested_denied');
      }
      
      return granted;
    } catch (error) {
      logger.error('Failed to request notification permission', { error });
      return false;
    }
  }, [updateState]);

  // Show training completion notification
  const showTrainingComplete = useCallback(async (
    accuracy: number,
    duration: number,
    trainingId: string
  ): Promise<boolean> => {
    const success = await notificationManager.showNotification({
      type: 'training_complete',
      title: '🎯 Quantum Training Complete!',
      body: `Your VQC achieved ${accuracy.toFixed(1)}% accuracy in ${(duration / 1000).toFixed(1)}s`,
      requireInteraction: true,
      quantum_data: {
        trainingId,
        accuracy,
        duration
      }
    });

    if (success) {
      trackQuantumEvents.featureDiscovery('training_notification_shown', JSON.stringify({
        accuracy,
        duration
      }));
    }

    return success;
  }, []);

  // Show simulation ready notification
  const showSimulationReady = useCallback(async (circuitId: string): Promise<boolean> => {
    const success = await notificationManager.showNotification({
      type: 'simulation_ready',
      title: '⚡ Quantum Simulation Ready',
      body: 'Your quantum circuit simulation has completed successfully',
      quantum_data: {
        circuitId
      }
    });

    if (success) {
      trackQuantumEvents.featureDiscovery('simulation_notification_shown', JSON.stringify({
        circuitId
      }));
    }

    return success;
  }, []);

  // Show achievement notification
  const showAchievement = useCallback(async (
    title: string,
    description: string
  ): Promise<boolean> => {
    const success = await notificationManager.showNotification({
      type: 'achievement',
      title: `🏆 ${title}`,
      body: description,
      requireInteraction: true
    });

    if (success) {
      trackQuantumEvents.featureDiscovery('achievement_notification_shown', JSON.stringify({
        achievement: title
      }));
    }

    return success;
  }, []);

  // Queue training for background processing
  const queueTraining = useCallback(async (
    circuitConfig: CircuitConfig,
    trainingParams: Record<string, unknown>
  ): Promise<string> => {
    const operationId = await notificationManager.queueOperation({
      type: 'training',
      data: { circuitConfig, trainingParams },
      priority: 'high'
    });

    updateState();
    
    trackQuantumEvents.featureDiscovery('training_queued_background', JSON.stringify({
      operationId,
      gates: circuitConfig.gates?.length || 0
    }));

    return operationId;
  }, [updateState]);

  // Queue simulation for background processing
  const queueSimulation = useCallback(async (
    circuitConfig: CircuitConfig,
    simulationParams: Record<string, unknown>
  ): Promise<string> => {
    const operationId = await notificationManager.queueOperation({
      type: 'simulation',
      data: { circuitConfig, simulationParams },
      priority: 'medium'
    });

    updateState();
    
    trackQuantumEvents.featureDiscovery('simulation_queued_background', JSON.stringify({
      operationId,
      qubits: circuitConfig.qubits || 0
    }));

    return operationId;
  }, [updateState]);

  // Queue sharing operation
  const queueSharing = useCallback(async (shareData: Record<string, unknown>): Promise<string> => {
    const operationId = await notificationManager.queueOperation({
      type: 'sharing',
      data: { shareData },
      priority: 'low'
    });

    updateState();
    
    trackQuantumEvents.featureDiscovery('sharing_queued_background', JSON.stringify({
      operationId,
      shareType: shareData.type
    }));

    return operationId;
  }, [updateState]);

  // Listen for permission changes
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        updateState();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [updateState]);

  // Periodic state updates
  useEffect(() => {
    const interval = setInterval(updateState, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, [updateState]);

  return {
    ...state,
    requestPermission,
    showTrainingComplete,
    showSimulationReady,
    showAchievement,
    queueTraining,
    queueSimulation,
    queueSharing
  };
};

// Hook for automatic achievement notifications
export const useAchievements = () => {
  const { showAchievement, isReady } = useNotifications();

  const checkAndNotifyAchievements = useCallback(async (metrics: {
    circuitsBuilt?: number;
    trainingsCompleted?: number;
    simulationsRun?: number;
    bestAccuracy?: number;
  }) => {
    if (!isReady) return;

    const { circuitsBuilt = 0, trainingsCompleted = 0, simulationsRun = 0, bestAccuracy = 0 } = metrics;

    // First circuit achievement
    if (circuitsBuilt === 1) {
      await showAchievement(
        'First Steps in Quantum',
        'You built your first quantum circuit!'
      );
    }

    // Circuit builder milestones
    if (circuitsBuilt === 10) {
      await showAchievement(
        'Circuit Architect',
        'You\'ve built 10 quantum circuits!'
      );
    }

    if (circuitsBuilt === 50) {
      await showAchievement(
        'Quantum Engineer',
        'Amazing! 50 quantum circuits created!'
      );
    }

    // Training achievements
    if (trainingsCompleted === 1) {
      await showAchievement(
        'First Training Complete',
        'You trained your first quantum model!'
      );
    }

    if (bestAccuracy >= 90) {
      await showAchievement(
        'High Accuracy Master',
        `Excellent! Your model achieved ${bestAccuracy.toFixed(1)}% accuracy!`
      );
    }

    if (bestAccuracy >= 95) {
      await showAchievement(
        'Quantum ML Expert',
        'Outstanding! 95%+ accuracy achieved!'
      );
    }

    // Simulation milestones
    if (simulationsRun === 25) {
      await showAchievement(
        'Simulation Specialist',
        'You\'ve run 25 quantum simulations!'
      );
    }

    if (simulationsRun === 100) {
      await showAchievement(
        'Quantum Researcher',
        'Incredible! 100 simulations completed!'
      );
    }

  }, [showAchievement, isReady]);

  return { checkAndNotifyAchievements };
};

// Hook for training progress notifications
export const useTrainingNotifications = () => {
  const { showTrainingComplete, queueTraining, isReady } = useNotifications();

  const notifyTrainingProgress = useCallback(async (
    epoch: number,
    totalEpochs: number,
    currentLoss: number,
    accuracy: number
  ) => {
    // Only show notifications for significant milestones
    const progressPercent = (epoch / totalEpochs) * 100;
    
    if (progressPercent === 25 || progressPercent === 50 || progressPercent === 75) {
      // These would be shown as in-app notifications, not system notifications
      trackQuantumEvents.featureDiscovery('training_milestone_reached', JSON.stringify({
        epoch,
        totalEpochs,
        progressPercent,
        accuracy
      }));
    }
  }, []);

  const startBackgroundTraining = useCallback(async (
    circuitConfig: CircuitConfig,
    trainingParams: Record<string, unknown>
  ): Promise<string> => {
    if (!isReady) {
      throw new Error('Notifications not ready for background training');
    }

    // Queue the training operation
    const operationId = await queueTraining(circuitConfig, trainingParams);

    // Show immediate feedback
    trackQuantumEvents.featureDiscovery('background_training_started', JSON.stringify({
      operationId,
      estimatedDuration: (trainingParams.epochs as number || 100) * 100 // Rough estimate
    }));

    return operationId;
  }, [queueTraining, isReady]);

  return {
    notifyTrainingProgress,
    startBackgroundTraining,
    showTrainingComplete
  };
};