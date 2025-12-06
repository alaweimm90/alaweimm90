import { create } from "zustand";

export type EvaluatorKind = "auto" | "local" | "openai" | "gemini";
export type BackendKind = "auto" | "webgpu" | "wasm" | "cpu";

export interface RtWeights {
  semantic: number;
  coverage: number;
  ats: number;
  gap: number;
}

export interface RtOptions {
  evaluatorKind: EvaluatorKind;
  localOnly: boolean;
  backend: BackendKind;
  localEmbModel?: string;
  openaiApiKey?: string;
  openaiEmbModel?: string;
  geminiApiKey?: string;
  geminiEmbModel?: string;
}

export interface RtResults {
  semantic: number;
  coverage: number;
  ats: number;
  gap: number;
  final: number;
  issues: { id: string; text: string; severity: "low" | "medium" | "high" }[];
  suggestions: string[];
}

interface RtState {
  resumeText: string;
  jdText: string;
  weights: RtWeights;
  options: RtOptions;
  results?: RtResults;
  setResumeText: (t: string) => void;
  setJdText: (t: string) => void;
  setWeights: (w: Partial<RtWeights>) => void;
  setOptions: (o: Partial<RtOptions>) => void;
  setResults: (r?: RtResults) => void;
  clearKeys: () => void;
}

export const useRtStore = create<RtState>((set) => ({
  resumeText: "",
  jdText: "",
  weights: { semantic: 0.4, coverage: 0.35, ats: 0.2, gap: 0.05 },
  options: { evaluatorKind: "local", localOnly: true, backend: "auto", localEmbModel: "sentence-transformers/all-MiniLM-L6-v2" },
  results: undefined,
  setResumeText: (t) => set({ resumeText: t }),
  setJdText: (t) => set({ jdText: t }),
  setWeights: (w) => set((s) => ({ weights: { ...s.weights, ...w } })),
  setOptions: (o) => set((s) => ({ options: { ...s.options, ...o } })),
  setResults: (r) => set({ results: r }),
  clearKeys: () => set((s) => ({ options: { ...s.options, openaiApiKey: undefined, geminiApiKey: undefined } })),
}));
