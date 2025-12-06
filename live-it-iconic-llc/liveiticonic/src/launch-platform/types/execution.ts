/**
 * Launch Platform - Execution Types
 * Type-safe execution parameters and results for all agents
 */

/**
 * Base execution parameters that all agents receive
 */
export interface BaseExecutionParams {
  action: string;
  [key: string]: unknown;
}

/**
 * Generic execution result
 */
export interface ExecutionResult<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  metrics?: Record<string, number>;
  timestamp: number;
}

/**
 * Brand Identity data structure
 */
export interface BrandIdentity {
  name: string;
  mission: string;
  vision: string;
  values: string[];
  personality: string[];
  voiceTone: string[];
}

/**
 * Market positioning data
 */
export interface MarketPositioning {
  targetMarket: string;
  uniqueValueProposition: string;
  differentiators: string[];
  competitiveAdvantages: string[];
}

/**
 * Visual system design
 */
export interface VisualSystem {
  colors: {
    primary: string[];
    secondary: string[];
    accent: string[];
  };
  typography: {
    headings: string;
    body: string;
    mono: string;
  };
  imagery: {
    style: string;
    guidelines: string[];
  };
}

/**
 * Content creation parameters
 */
export interface ContentParams extends BaseExecutionParams {
  topic: string;
  format: 'article' | 'social' | 'email' | 'video';
  tone: string;
  targetAudience: string;
  keywords?: string[];
}

/**
 * Market research parameters
 */
export interface MarketResearchParams extends BaseExecutionParams {
  product: string;
  industry: string;
  region?: string;
  competitors?: string[];
}

/**
 * Campaign management parameters
 */
export interface CampaignParams extends BaseExecutionParams {
  campaignType: 'launch' | 'awareness' | 'conversion' | 'retention';
  channels: string[];
  budget: number;
  duration: number;
  objectives: string[];
}

/**
 * Analytics parameters
 */
export interface AnalyticsParams extends BaseExecutionParams {
  metrics: string[];
  timeRange: {
    start: Date | string;
    end: Date | string;
  };
  dimensions?: string[];
}

/**
 * Type-safe execution parameter union
 */
export type AgentExecutionParams =
  | BaseExecutionParams
  | ContentParams
  | MarketResearchParams
  | CampaignParams
  | AnalyticsParams;
