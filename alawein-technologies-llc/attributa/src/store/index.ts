import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Report, AnalysisOptions } from '@/types';

interface AppState {
  // Current analysis state
  currentReport: Report | null;
  isAnalyzing: boolean;
  
  // History
  reports: Report[];
  
  // Settings
  apiKeys: {
    openai?: string;
    anthropic?: string;
  };
  
  analysisOptions: AnalysisOptions & { minChars: number; localModelName?: string };
  
  scoringWeights: {
    gltr: number;
    detectgpt: number;
    watermark: number;
    citations: number;
    cwe: number;
    shortPenalty: number;
  };
  
  // PDFs persisted by docId (as data URLs for reload safety)
  pdfsByDocId: Record<string, { name: string; dataUrl: string }[]>;
  
  // Actions
  setCurrentReport: (report: Report | null) => void;
  setAnalyzing: (analyzing: boolean) => void;
  addReport: (report: Report) => void;
  updateReport: (report: Report) => void;
  setPdfsForReport: (docId: string, pdfs: { name: string; dataUrl: string }[]) => void;
  updateApiKey: (provider: 'openai' | 'anthropic', key: string) => void;
  updateAnalysisOptions: (options: Partial<AnalysisOptions & { minChars: number; localModelName?: string }>) => void;
  updateScoringWeights: (weights: Partial<AppState['scoringWeights']>) => void;
  clearWorkspace: () => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      currentReport: null,
      isAnalyzing: false,
      reports: [],
      apiKeys: {},
      analysisOptions: {
        useLocalOnly: true,
        tryWatermark: false,
        useExternalApis: false,
        minChars: 1200,
        localModelName: 'onnx-community/gpt2',
      },
      
      scoringWeights: {
        gltr: 0.22,
        detectgpt: 0.22,
        watermark: 0.18,
        citations: 0.25,
        cwe: 0.10,
        shortPenalty: 0.03,
      },
      
      pdfsByDocId: {},
      
      setCurrentReport: (report) => set({ currentReport: report }),
      setAnalyzing: (analyzing) => set({ isAnalyzing: analyzing }),
      
      addReport: (report) => {
        const { reports } = get();
        set({ 
          reports: [report, ...reports].slice(0, 50), // Keep last 50 reports
          currentReport: report 
        });
      },

      updateReport: (report) => {
        const { reports } = get();
        const updated = reports.map(r => r.docId === report.docId ? report : r);
        set({ reports: updated, currentReport: report });
      },

      setPdfsForReport: (docId, pdfs) => {
        const { pdfsByDocId } = get();
        set({ pdfsByDocId: { ...pdfsByDocId, [docId]: pdfs } });
      },
      
      updateApiKey: (provider, key) => {
        const { apiKeys } = get();
        set({ 
          apiKeys: { ...apiKeys, [provider]: key }
        });
      },
      
      updateAnalysisOptions: (options) => {
        const { analysisOptions } = get();
        set({
          analysisOptions: { ...analysisOptions, ...options }
        });
      },
      
      updateScoringWeights: (weights) => {
        const { scoringWeights } = get();
        set({
          scoringWeights: { ...scoringWeights, ...weights }
        });
      },
      
      clearWorkspace: () => set({ 
        reports: [], 
        currentReport: null 
      }),
    }),
    {
      name: 'attributa-storage',
      partialize: (state) => ({
        reports: state.reports,
        apiKeys: state.apiKeys,
        analysisOptions: state.analysisOptions,
        scoringWeights: state.scoringWeights,
        pdfsByDocId: state.pdfsByDocId,
      }),
    }
  )
);