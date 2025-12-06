// src/services/api.ts
// Simple wrapper over mock APIs for consistency and future swapping
export { 
  mockIngest as ingest,
  mockAnalyzeText as analyzeText,
  mockAnalyzeWatermark as analyzeWatermark,
  mockAuditCitations as auditCitations,
  mockAnalyzeCode as analyzeCode,
  mockComputeScore as computeScore,
} from '@/services/mockApi';

// Optional cache clearer used by demos; no-op for now
export function clearSegmentCache() {
  try {
    // placeholder for any in-memory caches
  } catch (error) {
    // Intentionally empty - no caches to clear
    console.debug('Cache clear attempted:', error);
  }
}
