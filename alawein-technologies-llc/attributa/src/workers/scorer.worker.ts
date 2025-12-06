// src/workers/scorer.worker.ts
/// <reference lib="webworker" />
import { ensureModel } from '@/lib/nlp/tokenizers';
import { gltrStats } from '@/lib/nlp/gltr';
import { computeRanks, avgLogProb } from '@/lib/nlp/stepwise';
import { detectGPTCurvature } from '@/lib/nlp/detectgpt';

self.onmessage = async (e: MessageEvent) => {
  const { text } = e.data as { text: string };
  const model = await ensureModel();

  const ranks: number[] = await computeRanks(model, text);
  const gltr = gltrStats(ranks);

  const detectgpt = await detectGPTCurvature(text, (t: string) => avgLogProb(model, t));

  (self as unknown as Worker).postMessage({ gltr, detectgpt });
};
