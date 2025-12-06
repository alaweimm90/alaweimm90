import { useState, useEffect, useCallback } from 'react';
import { useMobileCapacitor } from './useMobileCapacitor';

interface OfflineData {
  workouts: Record<string, unknown>[];
  progress: Record<string, unknown>[];
  notes: Record<string, unknown>[];
  lastSync: Date | null;
}

export const useOfflineSupport = () => {
  const { networkStatus } = useMobileCapacitor();
  const [isOnline, setIsOnline] = useState(networkStatus.connected);
  const [offlineData, setOfflineData] = useState<OfflineData>({
    workouts: [],
    progress: [],
    notes: [],
    lastSync: null
  });
  const [pendingSync, setPendingSync] = useState<Array<Record<string, unknown>>>([]);

  useEffect(() => {
    setIsOnline(networkStatus.connected);

    // Auto-sync when coming back online
    if (networkStatus.connected && pendingSync.length > 0) {
      syncPendingData();
    }
  }, [networkStatus.connected, pendingSync.length, syncPendingData]);

  useEffect(() => {
    // Load offline data from localStorage on mount
    loadOfflineData();
  }, []);

  const loadOfflineData = () => {
    try {
      const stored = localStorage.getItem('repzcoach_offline_data');
      if (stored) {
        const data = JSON.parse(stored);
        setOfflineData({
          ...data,
          lastSync: data.lastSync ? new Date(data.lastSync) : null
        });
      }
      
      const pending = localStorage.getItem('repzcoach_pending_sync');
      if (pending) {
        setPendingSync(JSON.parse(pending));
      }
    } catch (error) {
      console.error('Error loading offline data:', error);
    }
  };

  const saveOfflineData = (data: Partial<OfflineData>) => {
    try {
      const updated = { ...offlineData, ...data, lastSync: new Date() };
      setOfflineData(updated);
      localStorage.setItem('repzcoach_offline_data', JSON.stringify(updated));
    } catch (error) {
      console.error('Error saving offline data:', error);
    }
  };

  const addToPendingSync = (action: Record<string, unknown>) => {
    const updated = [...pendingSync, { ...action, timestamp: Date.now() }];
    setPendingSync(updated);
    localStorage.setItem('repzcoach_pending_sync', JSON.stringify(updated));
  };

  const syncPendingData = useCallback(async () => {
    if (!isOnline || pendingSync.length === 0) return;

    try {
      // Process pending sync items
      const syncPromises = pendingSync.map(async (item) => {
        switch (item.type) {
          case 'workout_log':
            // Sync workout data
            break;
          case 'progress_update':
            // Sync progress data
            break;
          case 'note_create':
            // Sync notes
            break;
          default:
            break;
        }
      });

      await Promise.all(syncPromises);
      
      // Clear pending sync
      setPendingSync([]);
      localStorage.removeItem('repzcoach_pending_sync');
      
      // Update last sync time
      saveOfflineData({ lastSync: new Date() });
      
    } catch (error) {
      console.error('Error syncing data:', error);
    }
  }, [isOnline, pendingSync, saveOfflineData]);

  const cacheWorkout = (workout: Record<string, unknown>) => {
    const workouts = [...offlineData.workouts, workout];
    saveOfflineData({ workouts });
  };

  const cacheProgress = (progress: Record<string, unknown>) => {
    const progressData = [...offlineData.progress, progress];
    saveOfflineData({ progress: progressData });
  };

  const getOfflineWorkouts = () => offlineData.workouts;
  const getOfflineProgress = () => offlineData.progress;

  return {
    isOnline,
    offlineData,
    pendingSync: pendingSync.length,
    saveOfflineData,
    addToPendingSync,
    syncPendingData,
    cacheWorkout,
    cacheProgress,
    getOfflineWorkouts,
    getOfflineProgress,
    lastSync: offlineData.lastSync
  };
};