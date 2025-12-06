/**
 * LiveItIconic Launch Platform - Main Export
 *
 * AI-Powered Product Launch & Market Domination Platform
 */

// Core Infrastructure
export { BaseAgent } from './core/BaseAgent';
export { EventBus, eventBus } from './core/EventBus';
export { StateManager, stateManager } from './core/StateManager';
export { LaunchOrchestrator } from './core/LaunchOrchestrator';

// Learning System
export { LearningEngine, learningEngine } from './learning/LearningEngine';
export type { LearningConfig } from './learning/LearningEngine';

// Agents
export {
  CompetitorAnalystAgent,
  TrendDetectorAgent,
  BrandArchitectAgent,
  CopyWriterAgent,
  CampaignManagerAgent,
  AnalyticsInterpreterAgent,
  AgentFactory,
} from './agents';

// Types
export * from './types';

// Version
export const VERSION = '2.0.0';
export const PLATFORM_NAME = 'LiveItIconic Launch Platform';
