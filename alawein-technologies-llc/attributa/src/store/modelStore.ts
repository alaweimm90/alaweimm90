import { create } from 'zustand';

interface ModelState {
  isLoading: boolean;
  progress: number;
  modelName: string | null;
  error: string | null;
  setLoading: (loading: boolean) => void;
  setProgress: (progress: number) => void;
  setModelName: (name: string | null) => void;
  setError: (error: string | null) => void;
}

export const useModelStore = create<ModelState>((set) => ({
  isLoading: false,
  progress: 0,
  modelName: null,
  error: null,
  setLoading: (loading) => set({ isLoading: loading }),
  setProgress: (progress) => set({ progress }),
  setModelName: (modelName) => set({ modelName }),
  setError: (error) => set({ error }),
}));