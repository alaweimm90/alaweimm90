/**
 * LiveItIconic Launch Platform - Core Type Definitions
 *
 * Comprehensive type system for the AI-powered product launch orchestration platform
 */

// ============================================================================
// CORE AGENT TYPES
// ============================================================================

export enum AgentStatus {
  IDLE = 'idle',
  THINKING = 'thinking',
  EXECUTING = 'executing',
  WAITING = 'waiting',
  COMPLETED = 'completed',
  FAILED = 'failed',
  PAUSED = 'paused',
}

export enum AgentPriority {
  CRITICAL = 'critical',
  HIGH = 'high',
  NORMAL = 'normal',
  LOW = 'low',
}

export enum MessageType {
  REQUEST = 'request',
  RESPONSE = 'response',
  BROADCAST = 'broadcast',
  ERROR = 'error',
  STATUS_UPDATE = 'status_update',
}

export interface AgentCapability {
  name: string;
  description: string;
  inputs: Record<string, unknown>;
  outputs: Record<string, unknown>;
  constraints: Constraint[];
  dependencies: string[];
  estimatedDuration: number;
  successMetrics: Metric[];
}

export interface Constraint {
  type: 'budget' | 'time' | 'resource' | 'ethical' | 'legal';
  value: string | number | boolean | Record<string, unknown>;
  required: boolean;
}

export interface Metric {
  name: string;
  target: number;
  current?: number;
  unit: string;
}

export interface AgentMessage {
  id: string;
  from: string;
  to: string | string[]; // Can send to multiple agents
  type: MessageType;
  priority: AgentPriority;
  payload: Record<string, unknown>;
  timestamp: Date;
  requiresAck: boolean;
  metadata?: Record<string, unknown>;
}

export interface AgentConfig {
  id: string;
  name: string;
  type: AgentType;
  capabilities: AgentCapability[];
  maxConcurrentTasks: number;
  timeout: number;
  retryAttempts: number;
  learningEnabled: boolean;
}

export enum AgentType {
  // Market Intelligence
  COMPETITOR_ANALYST = 'competitor_analyst',
  TREND_DETECTOR = 'trend_detector',
  CUSTOMER_RESEARCHER = 'customer_researcher',
  PRICING_STRATEGIST = 'pricing_strategist',
  MARKET_SIZER = 'market_sizer',

  // Creative & Branding
  BRAND_ARCHITECT = 'brand_architect',
  COPYWRITER = 'copywriter',
  VISUAL_DESIGNER = 'visual_designer',
  VIDEO_PRODUCER = 'video_producer',
  STORYTELLER = 'storyteller',

  // Launch Execution
  CAMPAIGN_MANAGER = 'campaign_manager',
  SOCIAL_MEDIA_STRATEGIST = 'social_media_strategist',
  INFLUENCER_OUTREACH = 'influencer_outreach',
  PR_MANAGER = 'pr_manager',
  EMAIL_MARKETER = 'email_marketer',
  CONTENT_DISTRIBUTOR = 'content_distributor',

  // Optimization
  CONVERSION_OPTIMIZER = 'conversion_optimizer',
  SEO_SPECIALIST = 'seo_specialist',
  PAID_ADS_MANAGER = 'paid_ads_manager',
  ANALYTICS_INTERPRETER = 'analytics_interpreter',
  FEEDBACK_ANALYZER = 'feedback_analyzer',

  // Supporting Roles
  DATA_COLLECTOR = 'data_collector',
  QUALITY_CONTROLLER = 'quality_controller',
  COMPLIANCE_CHECKER = 'compliance_checker',
  BUDGET_MANAGER = 'budget_manager',
  RISK_ASSESSOR = 'risk_assessor',
}

// ============================================================================
// PRODUCT & MARKET TYPES
// ============================================================================

export interface Product {
  id: string;
  name: string;
  description: string;
  category: string;
  targetMarket: TargetMarket;
  pricing: PricingStrategy;
  features: Feature[];
  differentiators: string[];
  stage: ProductStage;
}

export enum ProductStage {
  CONCEPT = 'concept',
  DEVELOPMENT = 'development',
  READY_TO_LAUNCH = 'ready_to_launch',
  LAUNCHING = 'launching',
  LIVE = 'live',
  SCALING = 'scaling',
}

export interface TargetMarket {
  segments: MarketSegment[];
  primaryPersona: Persona;
  secondaryPersonas: Persona[];
  geography: string[];
  tam: number;
  sam: number;
  som: number;
}

export interface MarketSegment {
  name: string;
  size: number;
  growth: number;
  characteristics: string[];
  painPoints: string[];
}

export interface Persona {
  name: string;
  age: [number, number];
  income: [number, number];
  occupation: string[];
  psychographics: string[];
  behaviors: string[];
  painPoints: string[];
  goals: string[];
  channels: string[];
}

export interface Feature {
  name: string;
  description: string;
  benefit: string;
  priority: 'must-have' | 'nice-to-have' | 'future';
}

export interface PricingStrategy {
  model: 'freemium' | 'subscription' | 'one-time' | 'usage-based' | 'tiered';
  basePrice: number;
  currency: string;
  tiers?: PricingTier[];
  discounts?: Discount[];
}

export interface PricingTier {
  name: string;
  price: number;
  features: string[];
  limits: Record<string, number>;
}

export interface Discount {
  type: 'early-bird' | 'volume' | 'promotional' | 'seasonal';
  value: number;
  isPercentage: boolean;
  conditions: string[];
  validUntil?: Date;
}

// ============================================================================
// LAUNCH TYPES
// ============================================================================

export interface LaunchPlan {
  id: string;
  product: Product;
  strategy: LaunchStrategy;
  resources: ResourceAllocation;
  timeline: LaunchTimeline;
  metrics: LaunchMetrics;
  risks: Risk[];
  status: LaunchStatus;
  createdAt: Date;
  updatedAt: Date;
}

export enum LaunchStatus {
  PLANNING = 'planning',
  APPROVED = 'approved',
  IN_PROGRESS = 'in_progress',
  PAUSED = 'paused',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

export interface LaunchStrategy {
  type: 'stealth' | 'soft' | 'hard' | 'phased' | 'viral';
  channels: Channel[];
  messaging: MessagingStrategy;
  positioning: Positioning;
  budget: Budget;
}

export interface Channel {
  id: string;
  name: string;
  type: ChannelType;
  priority: number;
  budget: number;
  metrics: ChannelMetrics;
  content: Content[];
  schedule: ChannelSchedule;
}

export enum ChannelType {
  SOCIAL_MEDIA = 'social_media',
  EMAIL = 'email',
  PAID_ADS = 'paid_ads',
  INFLUENCER = 'influencer',
  PR = 'pr',
  CONTENT_MARKETING = 'content_marketing',
  SEO = 'seo',
  COMMUNITY = 'community',
  EVENTS = 'events',
  PARTNERSHIPS = 'partnerships',
}

export interface ChannelMetrics {
  reach: number;
  engagement: number;
  conversions: number;
  cac: number;
  roi: number;
}

export interface Content {
  id: string;
  type: ContentType;
  title: string;
  description: string;
  format: string;
  status: ContentStatus;
  assignedTo: string; // Agent ID
  dueDate: Date;
  assets: Asset[];
}

export enum ContentType {
  BLOG_POST = 'blog_post',
  SOCIAL_POST = 'social_post',
  VIDEO = 'video',
  IMAGE = 'image',
  INFOGRAPHIC = 'infographic',
  LANDING_PAGE = 'landing_page',
  EMAIL = 'email',
  PRESS_RELEASE = 'press_release',
  WHITEPAPER = 'whitepaper',
  CASE_STUDY = 'case_study',
}

export enum ContentStatus {
  PLANNING = 'planning',
  IN_PRODUCTION = 'in_production',
  REVIEW = 'review',
  APPROVED = 'approved',
  PUBLISHED = 'published',
  ARCHIVED = 'archived',
}

export interface Asset {
  id: string;
  type: 'image' | 'video' | 'audio' | 'document' | 'code';
  url: string;
  metadata: Record<string, unknown>;
}

export interface MessagingStrategy {
  coreMessage: string;
  valueProposition: string;
  tagline: string;
  keyMessages: string[];
  callToAction: string[];
  tone: string[];
}

export interface Positioning {
  category: string;
  differentiation: string;
  targetCompetitors: string[];
  uniqueValue: string;
}

export interface Budget {
  total: number;
  allocated: Record<string, number>;
  spent: number;
  remaining: number;
  breakdown: BudgetItem[];
}

export interface BudgetItem {
  category: string;
  amount: number;
  spent: number;
  remaining: number;
}

export interface ResourceAllocation {
  agents: AgentAssignment[];
  tools: Tool[];
  budget: Budget;
  timeline: number; // days
}

export interface AgentAssignment {
  agentId: string;
  agentType: AgentType;
  tasks: Task[];
  priority: AgentPriority;
  allocation: number; // percentage 0-100
}

export interface Task {
  id: string;
  name: string;
  description: string;
  assignedTo: string;
  status: TaskStatus;
  priority: AgentPriority;
  dependencies: string[];
  estimatedDuration: number;
  actualDuration?: number;
  deliverables: string[];
  createdAt: Date;
  dueDate: Date;
  completedAt?: Date;
}

export enum TaskStatus {
  PENDING = 'pending',
  IN_PROGRESS = 'in_progress',
  BLOCKED = 'blocked',
  REVIEW = 'review',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

export interface Tool {
  id: string;
  name: string;
  type: string;
  apiKey?: string;
  config: Record<string, unknown>;
}

export interface LaunchTimeline {
  startDate: Date;
  launchDate: Date;
  endDate: Date;
  phases: Phase[];
  milestones: Milestone[];
}

export interface Phase {
  id: string;
  name: string;
  description: string;
  startDate: Date;
  endDate: Date;
  tasks: Task[];
  status: PhaseStatus;
}

export enum PhaseStatus {
  NOT_STARTED = 'not_started',
  IN_PROGRESS = 'in_progress',
  COMPLETED = 'completed',
  DELAYED = 'delayed',
}

export interface Milestone {
  id: string;
  name: string;
  description: string;
  date: Date;
  completed: boolean;
  metrics: Record<string, number>;
}

export interface LaunchMetrics {
  targets: Record<string, Metric>;
  actual: Record<string, number>;
  timestamp: Date;
}

export interface Risk {
  id: string;
  category: RiskCategory;
  description: string;
  impact: 'low' | 'medium' | 'high' | 'critical';
  probability: number; // 0-1
  mitigation: string;
  owner: string;
  status: 'identified' | 'monitoring' | 'mitigated' | 'realized';
}

export enum RiskCategory {
  MARKET = 'market',
  COMPETITIVE = 'competitive',
  TECHNICAL = 'technical',
  FINANCIAL = 'financial',
  REGULATORY = 'regulatory',
  REPUTATIONAL = 'reputational',
  OPERATIONAL = 'operational',
}

export interface ChannelSchedule {
  frequency: string;
  timeSlots: TimeSlot[];
  blackoutDates: Date[];
}

export interface TimeSlot {
  day: string;
  time: string;
  timezone: string;
}

// ============================================================================
// LEARNING & OPTIMIZATION TYPES
// ============================================================================

export interface Experience {
  id: string;
  launchId: string;
  productType: string;
  strategy: LaunchStrategy;
  actions: Action[];
  outcomes: Outcome[];
  metrics: Record<string, number>;
  timestamp: Date;
}

export interface Action {
  id: string;
  agentId: string;
  type: string;
  parameters: Record<string, unknown>;
  timestamp: Date;
  result: ActionResult;
}

export interface ActionResult {
  success: boolean;
  output: unknown;
  metrics: Record<string, number>;
  duration: number;
  error?: string;
}

export interface Outcome {
  metric: string;
  target: number;
  actual: number;
  variance: number;
  success: boolean;
}

export interface Pattern {
  id: string;
  type: 'success' | 'failure' | 'optimization';
  description: string;
  conditions: Condition[];
  actions: string[];
  outcomes: Record<string, number>;
  confidence: number;
  frequency: number;
  domains: string[];
}

export interface Condition {
  field: string;
  operator: 'equals' | 'greater' | 'less' | 'contains' | 'between';
  value: string | number | boolean | unknown[];
}

export interface SuccessPrediction {
  probability: number;
  confidence: number;
  factors: Factor[];
  risks: Risk[];
  recommendations: Recommendation[];
}

export interface Factor {
  name: string;
  impact: number; // -1 to 1
  confidence: number;
  explanation: string;
}

export interface Recommendation {
  id: string;
  priority: AgentPriority;
  category: string;
  description: string;
  expectedImpact: number;
  effort: 'low' | 'medium' | 'high';
  confidence: number;
}

// ============================================================================
// MARKET INTELLIGENCE TYPES
// ============================================================================

export interface MarketData {
  trends: Trend[];
  competitors: Competitor[];
  opportunities: Opportunity[];
  threats: Threat[];
  insights: Insight[];
  timestamp: Date;
}

export interface Trend {
  id: string;
  name: string;
  category: string;
  direction: 'rising' | 'falling' | 'stable';
  magnitude: number;
  velocity: number;
  relevance: number;
  sources: string[];
  keywords: string[];
}

export interface Competitor {
  id: string;
  name: string;
  category: string;
  marketShare: number;
  strengths: string[];
  weaknesses: string[];
  products: CompetitorProduct[];
  activities: CompetitorActivity[];
  sentiment: number; // -1 to 1
}

export interface CompetitorProduct {
  name: string;
  description: string;
  price: number;
  features: string[];
  reviews: number;
  rating: number;
}

export interface CompetitorActivity {
  type: 'launch' | 'update' | 'campaign' | 'acquisition' | 'partnership';
  description: string;
  date: Date;
  impact: 'low' | 'medium' | 'high';
}

export interface Opportunity {
  id: string;
  type: 'market' | 'product' | 'channel' | 'partnership' | 'timing';
  description: string;
  potential: number;
  effort: 'low' | 'medium' | 'high';
  timeframe: string;
  confidence: number;
  sources: string[];
}

export interface Threat {
  id: string;
  type: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  probability: number;
  timeframe: string;
  mitigation: string;
}

export interface Insight {
  id: string;
  category: string;
  description: string;
  importance: number;
  actionable: boolean;
  recommendations: string[];
  sources: string[];
}

// ============================================================================
// GOVERNANCE & SAFETY TYPES
// ============================================================================

export interface EthicalConstraints {
  truthfulness: {
    noFalseAdvertising: boolean;
    accurateClaimsOnly: boolean;
    transparentPricing: boolean;
  };
  fairness: {
    noBiasAmplification: boolean;
    equalOpportunity: boolean;
    inclusiveMessaging: boolean;
  };
  privacy: {
    gdprCompliant: boolean;
    ccpaCompliant: boolean;
    userConsentRequired: boolean;
  };
  safety: {
    noBrandDamage: boolean;
    noReputationalRisk: boolean;
    noLegalViolations: boolean;
  };
}

export interface HumanOversight {
  approvalRequired: {
    campaignLaunch: boolean;
    majorSpend: boolean;
    brandChanges: boolean;
    controversialContent: boolean;
  };
  reviewTriggers: {
    confidenceThreshold: number;
    spendThreshold: number;
    riskScore: number;
  };
  killSwitches: {
    emergencyStop: boolean;
    rollback: boolean;
    pauseCampaign: boolean;
  };
}

export interface ComplianceCheck {
  id: string;
  type: string;
  status: 'pending' | 'approved' | 'rejected' | 'needs_review';
  reviewer?: string;
  notes: string;
  timestamp: Date;
}

// ============================================================================
// EXECUTION RESULT TYPES
// ============================================================================

export interface LaunchResult {
  launchId: string;
  success: boolean;
  metrics: LaunchMetrics;
  outcomes: Outcome[];
  insights: Insight[];
  nextSteps: Recommendation[];
  timestamp: Date;
}

export interface ExecutionContext {
  launchPlan: LaunchPlan;
  currentPhase: Phase;
  availableResources: ResourceAllocation;
  constraints: EthicalConstraints;
  oversight: HumanOversight;
}
export * from './execution';
