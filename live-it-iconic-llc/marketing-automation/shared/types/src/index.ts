// Core Types for Marketing Automation Platform

// ==================== USER & AUTHENTICATION ====================

export interface User {
  id: string;
  email: string;
  name: string;
  role: UserRole;
  organizationId: string;
  preferences: UserPreferences;
  apiKeys: ApiKey[];
  createdAt: Date;
  updatedAt: Date;
}

export enum UserRole {
  ADMIN = 'admin',
  MANAGER = 'manager',
  CREATOR = 'creator',
  VIEWER = 'viewer'
}

export interface UserPreferences {
  timezone: string;
  language: string;
  notifications: NotificationSettings;
  defaultBrand?: string;
}

export interface NotificationSettings {
  email: boolean;
  push: boolean;
  sms: boolean;
  slack: boolean;
}

export interface ApiKey {
  id: string;
  name: string;
  key: string;
  permissions: string[];
  lastUsed?: Date;
  expiresAt?: Date;
}

// ==================== ORGANIZATION ====================

export interface Organization {
  id: string;
  name: string;
  plan: PricingPlan;
  industry: Industry;
  settings: OrganizationSettings;
  brands: Brand[];
  users: User[];
  createdAt: Date;
  updatedAt: Date;
}

export enum PricingPlan {
  STARTER = 'starter',
  PROFESSIONAL = 'professional',
  ENTERPRISE = 'enterprise',
  AGENCY = 'agency'
}

export enum Industry {
  ECOMMERCE = 'ecommerce',
  B2B_SAAS = 'b2b_saas',
  SERVICES = 'services',
  CONTENT_CREATOR = 'content_creator',
  RETAIL = 'retail',
  HEALTHCARE = 'healthcare',
  FINANCE = 'finance',
  EDUCATION = 'education',
  REAL_ESTATE = 'real_estate',
  HOSPITALITY = 'hospitality',
  OTHER = 'other'
}

export interface OrganizationSettings {
  usage: UsageLimits;
  integrations: IntegrationConfig[];
  compliance: ComplianceSettings;
  branding: BrandingSettings;
}

export interface UsageLimits {
  postsPerMonth: number;
  emailsPerMonth: number;
  aiTokensPerMonth: number;
  storageGB: number;
  teamMembers: number;
}

// ==================== BRAND ====================

export interface Brand {
  id: string;
  name: string;
  organizationId: string;
  industry: Industry;
  voiceAndTone: VoiceAndTone;
  visualIdentity: VisualIdentity;
  targetAudience: TargetAudience;
  competitors: Competitor[];
  keywords: string[];
  createdAt: Date;
  updatedAt: Date;
}

export interface VoiceAndTone {
  personality: string[];
  style: ContentStyle;
  prohibitedWords: string[];
  preferredPhrases: string[];
  emojiUsage: 'none' | 'minimal' | 'moderate' | 'frequent';
  formality: 'casual' | 'professional' | 'formal';
}

export enum ContentStyle {
  CONVERSATIONAL = 'conversational',
  PROFESSIONAL = 'professional',
  EDUCATIONAL = 'educational',
  ENTERTAINING = 'entertaining',
  INSPIRATIONAL = 'inspirational',
  AUTHORITATIVE = 'authoritative'
}

export interface VisualIdentity {
  primaryColor: string;
  secondaryColors: string[];
  fonts: FontConfig[];
  logoUrl?: string;
  brandGuidelinesUrl?: string;
}

export interface FontConfig {
  name: string;
  family: string;
  weights: number[];
  usage: 'heading' | 'body' | 'accent';
}

export interface TargetAudience {
  demographics: Demographics;
  psychographics: Psychographics;
  painPoints: string[];
  goals: string[];
  platforms: PlatformType[];
}

export interface Demographics {
  ageRange: [number, number];
  gender: 'male' | 'female' | 'all';
  locations: string[];
  income: string;
  education: string;
  occupation: string[];
}

export interface Psychographics {
  interests: string[];
  values: string[];
  lifestyle: string[];
  challenges: string[];
}

export interface Competitor {
  name: string;
  website: string;
  socialHandles: Record<PlatformType, string>;
  strengths: string[];
  weaknesses: string[];
}

// ==================== CONTENT ====================

export interface Content {
  id: string;
  organizationId: string;
  brandId: string;
  type: ContentType;
  title: string;
  body: string;
  status: ContentStatus;
  platform: PlatformType;
  metadata: ContentMetadata;
  aiGenerated: boolean;
  variations: ContentVariation[];
  scheduledFor?: Date;
  publishedAt?: Date;
  performance?: ContentPerformance;
  tags: string[];
  createdBy: string;
  createdAt: Date;
  updatedAt: Date;
}

export enum ContentType {
  SOCIAL_POST = 'social_post',
  BLOG_ARTICLE = 'blog_article',
  EMAIL = 'email',
  VIDEO_SCRIPT = 'video_script',
  AD_COPY = 'ad_copy',
  PRODUCT_DESCRIPTION = 'product_description',
  PRESS_RELEASE = 'press_release',
  PODCAST_SCRIPT = 'podcast_script',
  LANDING_PAGE = 'landing_page',
  NEWSLETTER = 'newsletter'
}

export enum ContentStatus {
  DRAFT = 'draft',
  SCHEDULED = 'scheduled',
  PUBLISHED = 'published',
  ARCHIVED = 'archived',
  FAILED = 'failed'
}

export enum PlatformType {
  INSTAGRAM = 'instagram',
  FACEBOOK = 'facebook',
  TWITTER = 'twitter',
  LINKEDIN = 'linkedin',
  TIKTOK = 'tiktok',
  YOUTUBE = 'youtube',
  PINTEREST = 'pinterest',
  TWITCH = 'twitch',
  SNAPCHAT = 'snapchat',
  REDDIT = 'reddit',
  EMAIL = 'email',
  BLOG = 'blog',
  WEBSITE = 'website'
}

export interface ContentMetadata {
  hashtags?: string[];
  mentions?: string[];
  links?: string[];
  mediaUrls?: string[];
  captions?: string;
  alt?: string;
  seo?: SEOMetadata;
  cta?: CallToAction;
}

export interface SEOMetadata {
  metaTitle: string;
  metaDescription: string;
  keywords: string[];
  slug: string;
  canonicalUrl?: string;
  focusKeyword: string;
  readabilityScore?: number;
  seoScore?: number;
}

export interface CallToAction {
  text: string;
  url: string;
  type: 'link' | 'button' | 'swipe_up' | 'shop_now' | 'learn_more';
}

export interface ContentVariation {
  id: string;
  body: string;
  platform: PlatformType;
  performance?: ContentPerformance;
}

export interface ContentPerformance {
  impressions: number;
  reach: number;
  engagement: number;
  likes: number;
  comments: number;
  shares: number;
  clicks: number;
  conversions: number;
  revenue?: number;
  engagementRate: number;
  clickThroughRate: number;
  conversionRate: number;
  roi?: number;
}

// ==================== CAMPAIGNS ====================

export interface Campaign {
  id: string;
  organizationId: string;
  brandId: string;
  name: string;
  description: string;
  type: CampaignType;
  status: CampaignStatus;
  objective: CampaignObjective;
  budget?: Budget;
  content: Content[];
  schedule: CampaignSchedule;
  targeting: TargetingConfig;
  performance: CampaignPerformance;
  tags: string[];
  createdBy: string;
  createdAt: Date;
  updatedAt: Date;
  startsAt: Date;
  endsAt?: Date;
}

export enum CampaignType {
  PRODUCT_LAUNCH = 'product_launch',
  SEASONAL = 'seasonal',
  BRAND_AWARENESS = 'brand_awareness',
  LEAD_GENERATION = 'lead_generation',
  SALES = 'sales',
  ENGAGEMENT = 'engagement',
  RETARGETING = 'retargeting',
  EVENT = 'event',
  INFLUENCER = 'influencer'
}

export enum CampaignStatus {
  DRAFT = 'draft',
  SCHEDULED = 'scheduled',
  ACTIVE = 'active',
  PAUSED = 'paused',
  COMPLETED = 'completed',
  CANCELLED = 'cancelled'
}

export enum CampaignObjective {
  AWARENESS = 'awareness',
  CONSIDERATION = 'consideration',
  CONVERSION = 'conversion',
  RETENTION = 'retention'
}

export interface Budget {
  total: number;
  spent: number;
  currency: string;
  allocation: Record<PlatformType, number>;
}

export interface CampaignSchedule {
  startDate: Date;
  endDate?: Date;
  postingFrequency: PostingFrequency;
  optimalTiming: boolean;
  timezone: string;
}

export interface PostingFrequency {
  instagram: number;
  facebook: number;
  twitter: number;
  linkedin: number;
  tiktok: number;
  youtube: number;
}

export interface TargetingConfig {
  demographics: Demographics;
  interests: string[];
  behaviors: string[];
  locations: string[];
  languages: string[];
  customAudiences?: string[];
  lookalikeSimilarity?: number;
}

export interface CampaignPerformance {
  totalReach: number;
  totalImpressions: number;
  totalEngagement: number;
  totalClicks: number;
  totalConversions: number;
  totalRevenue: number;
  roi: number;
  cpa: number;
  cpc: number;
  cpm: number;
  platformBreakdown: Record<PlatformType, ContentPerformance>;
}

// ==================== EMAIL MARKETING ====================

export interface EmailCampaign {
  id: string;
  organizationId: string;
  brandId: string;
  name: string;
  type: EmailCampaignType;
  status: CampaignStatus;
  subject: string;
  preheader: string;
  fromName: string;
  fromEmail: string;
  replyTo: string;
  htmlContent: string;
  textContent: string;
  segments: EmailSegment[];
  abTest?: ABTest;
  schedule: EmailSchedule;
  performance: EmailPerformance;
  createdAt: Date;
  updatedAt: Date;
}

export enum EmailCampaignType {
  NEWSLETTER = 'newsletter',
  PROMOTIONAL = 'promotional',
  TRANSACTIONAL = 'transactional',
  DRIP = 'drip',
  ABANDONED_CART = 'abandoned_cart',
  WELCOME = 'welcome',
  RE_ENGAGEMENT = 're_engagement'
}

export interface EmailSegment {
  id: string;
  name: string;
  criteria: SegmentCriteria;
  recipientCount: number;
}

export interface SegmentCriteria {
  behavior?: BehaviorCriteria;
  demographics?: Demographics;
  engagement?: EngagementCriteria;
  customFields?: Record<string, any>;
}

export interface BehaviorCriteria {
  purchaseHistory?: string;
  websiteActivity?: string;
  emailActivity?: string;
  productInterests?: string[];
}

export interface EngagementCriteria {
  opened?: boolean;
  clicked?: boolean;
  converted?: boolean;
  lastEngagement?: Date;
}

export interface ABTest {
  enabled: boolean;
  variants: ABVariant[];
  winningMetric: 'open_rate' | 'click_rate' | 'conversion_rate';
  sampleSize: number;
  duration: number;
  winner?: string;
}

export interface ABVariant {
  id: string;
  name: string;
  subject?: string;
  content?: string;
  percentage: number;
  performance?: EmailPerformance;
}

export interface EmailSchedule {
  sendAt?: Date;
  sendImmediately?: boolean;
  timezone: string;
  optimizeSendTime?: boolean;
}

export interface EmailPerformance {
  sent: number;
  delivered: number;
  opened: number;
  clicked: number;
  converted: number;
  bounced: number;
  unsubscribed: number;
  spam: number;
  openRate: number;
  clickRate: number;
  conversionRate: number;
  bounceRate: number;
  unsubscribeRate: number;
  revenue?: number;
}

// ==================== SOCIAL MEDIA ====================

export interface SocialPost {
  id: string;
  contentId: string;
  platform: PlatformType;
  accountId: string;
  status: PostStatus;
  scheduledFor?: Date;
  publishedAt?: Date;
  platformPostId?: string;
  mediaUrls: string[];
  error?: string;
  retryCount: number;
  performance?: ContentPerformance;
}

export enum PostStatus {
  DRAFT = 'draft',
  SCHEDULED = 'scheduled',
  PUBLISHING = 'publishing',
  PUBLISHED = 'published',
  FAILED = 'failed',
  DELETED = 'deleted'
}

export interface SocialAccount {
  id: string;
  organizationId: string;
  platform: PlatformType;
  username: string;
  displayName: string;
  accessToken: string;
  refreshToken?: string;
  tokenExpiresAt?: Date;
  profileUrl: string;
  avatarUrl?: string;
  followers: number;
  isActive: boolean;
  lastSyncedAt?: Date;
  createdAt: Date;
}

export interface EngagementAction {
  id: string;
  accountId: string;
  type: EngagementType;
  targetId: string;
  targetType: 'comment' | 'message' | 'mention';
  response?: string;
  sentiment?: 'positive' | 'neutral' | 'negative';
  automated: boolean;
  createdAt: Date;
}

export enum EngagementType {
  REPLY = 'reply',
  LIKE = 'like',
  FOLLOW = 'follow',
  SHARE = 'share',
  DM = 'dm'
}

// ==================== ANALYTICS ====================

export interface AnalyticsReport {
  id: string;
  organizationId: string;
  type: ReportType;
  period: DateRange;
  metrics: Record<string, number>;
  insights: Insight[];
  recommendations: Recommendation[];
  generatedAt: Date;
}

export enum ReportType {
  OVERVIEW = 'overview',
  CONTENT_PERFORMANCE = 'content_performance',
  CAMPAIGN_PERFORMANCE = 'campaign_performance',
  AUDIENCE_INSIGHTS = 'audience_insights',
  COMPETITOR_ANALYSIS = 'competitor_analysis',
  ROI = 'roi'
}

export interface DateRange {
  startDate: Date;
  endDate: Date;
}

export interface Insight {
  title: string;
  description: string;
  importance: 'high' | 'medium' | 'low';
  category: string;
  data?: any;
}

export interface Recommendation {
  title: string;
  description: string;
  priority: 'high' | 'medium' | 'low';
  impact: 'high' | 'medium' | 'low';
  effort: 'high' | 'medium' | 'low';
  actionItems: string[];
}

// ==================== INTEGRATIONS ====================

export interface Integration {
  id: string;
  organizationId: string;
  platform: IntegrationPlatform;
  name: string;
  credentials: Record<string, string>;
  config: IntegrationConfig;
  status: IntegrationStatus;
  lastSyncedAt?: Date;
  createdAt: Date;
  updatedAt: Date;
}

export enum IntegrationPlatform {
  // Social Media
  INSTAGRAM = 'instagram',
  FACEBOOK = 'facebook',
  TWITTER = 'twitter',
  LINKEDIN = 'linkedin',
  TIKTOK = 'tiktok',
  YOUTUBE = 'youtube',

  // Email
  MAILCHIMP = 'mailchimp',
  SENDGRID = 'sendgrid',
  CONVERTKIT = 'convertkit',
  ACTIVECAMPAIGN = 'activecampaign',
  KLAVIYO = 'klaviyo',

  // CRM
  HUBSPOT = 'hubspot',
  SALESFORCE = 'salesforce',
  PIPEDRIVE = 'pipedrive',
  ZOHO = 'zoho',

  // E-commerce
  SHOPIFY = 'shopify',
  WOOCOMMERCE = 'woocommerce',
  MAGENTO = 'magento',
  BIGCOMMERCE = 'bigcommerce',

  // Analytics
  GOOGLE_ANALYTICS = 'google_analytics',
  MIXPANEL = 'mixpanel',
  AMPLITUDE = 'amplitude',

  // Payment
  STRIPE = 'stripe',
  PAYPAL = 'paypal',
  SQUARE = 'square',

  // Other
  ZAPIER = 'zapier',
  SLACK = 'slack',
  DISCORD = 'discord'
}

export interface IntegrationConfig {
  syncEnabled: boolean;
  syncFrequency?: number;
  webhookUrl?: string;
  customFields?: Record<string, any>;
}

export enum IntegrationStatus {
  ACTIVE = 'active',
  INACTIVE = 'inactive',
  ERROR = 'error',
  EXPIRED = 'expired'
}

// ==================== AI GENERATION ====================

export interface GenerationRequest {
  id: string;
  organizationId: string;
  brandId: string;
  type: ContentType;
  platform: PlatformType;
  prompt: string;
  parameters: GenerationParameters;
  status: GenerationStatus;
  result?: GenerationResult;
  error?: string;
  createdAt: Date;
  completedAt?: Date;
}

export interface GenerationParameters {
  tone?: string;
  style?: ContentStyle;
  length?: number;
  keywords?: string[];
  includeHashtags?: boolean;
  includeEmojis?: boolean;
  targetAudience?: string;
  callToAction?: string;
  variations?: number;
  temperature?: number;
  model?: 'gpt-4' | 'claude' | 'custom';
}

export enum GenerationStatus {
  PENDING = 'pending',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed'
}

export interface GenerationResult {
  content: string[];
  metadata: {
    tokensUsed: number;
    processingTime: number;
    model: string;
    confidence?: number;
  };
}

// ==================== VISUAL CONTENT ====================

export interface VisualAsset {
  id: string;
  organizationId: string;
  brandId: string;
  type: AssetType;
  url: string;
  thumbnailUrl?: string;
  width?: number;
  height?: number;
  size: number;
  format: string;
  metadata: AssetMetadata;
  tags: string[];
  aiGenerated: boolean;
  createdAt: Date;
}

export enum AssetType {
  IMAGE = 'image',
  VIDEO = 'video',
  GRAPHIC = 'graphic',
  LOGO = 'logo',
  THUMBNAIL = 'thumbnail',
  BANNER = 'banner',
  INFOGRAPHIC = 'infographic'
}

export interface AssetMetadata {
  alt?: string;
  caption?: string;
  generationPrompt?: string;
  processingSteps?: string[];
  sourceAsset?: string;
}

// ==================== AUTOMATION ====================

export interface Workflow {
  id: string;
  organizationId: string;
  name: string;
  description: string;
  trigger: WorkflowTrigger;
  actions: WorkflowAction[];
  conditions: WorkflowCondition[];
  status: 'active' | 'inactive';
  executions: number;
  lastExecutedAt?: Date;
  createdAt: Date;
  updatedAt: Date;
}

export interface WorkflowTrigger {
  type: TriggerType;
  config: Record<string, any>;
}

export enum TriggerType {
  SCHEDULE = 'schedule',
  WEBHOOK = 'webhook',
  EVENT = 'event',
  MANUAL = 'manual'
}

export interface WorkflowAction {
  id: string;
  type: ActionType;
  config: Record<string, any>;
  order: number;
}

export enum ActionType {
  GENERATE_CONTENT = 'generate_content',
  POST_SOCIAL = 'post_social',
  SEND_EMAIL = 'send_email',
  CREATE_CAMPAIGN = 'create_campaign',
  UPDATE_CRM = 'update_crm',
  NOTIFY = 'notify',
  WAIT = 'wait',
  CONDITIONAL = 'conditional'
}

export interface WorkflowCondition {
  field: string;
  operator: 'equals' | 'contains' | 'greater_than' | 'less_than';
  value: any;
}

// ==================== COMPLIANCE ====================

export interface ComplianceSettings {
  gdprEnabled: boolean;
  ccpaEnabled: boolean;
  dataRetentionDays: number;
  consentRequired: boolean;
  allowedCountries?: string[];
  blockedCountries?: string[];
  contentModeration: ContentModerationSettings;
}

export interface ContentModerationSettings {
  enabled: boolean;
  autoReject: boolean;
  categories: string[];
  customRules: ModerationRule[];
}

export interface ModerationRule {
  name: string;
  pattern: string;
  action: 'flag' | 'reject' | 'auto_fix';
  severity: 'high' | 'medium' | 'low';
}

export interface BrandingSettings {
  allowWhiteLabel: boolean;
  customDomain?: string;
  customLogo?: string;
  customColors?: Record<string, string>;
}

// ==================== WEBHOOKS ====================

export interface Webhook {
  id: string;
  organizationId: string;
  url: string;
  events: WebhookEvent[];
  secret: string;
  status: 'active' | 'inactive';
  retryConfig: RetryConfig;
  lastTriggeredAt?: Date;
  createdAt: Date;
}

export enum WebhookEvent {
  CONTENT_PUBLISHED = 'content.published',
  CAMPAIGN_STARTED = 'campaign.started',
  CAMPAIGN_COMPLETED = 'campaign.completed',
  EMAIL_SENT = 'email.sent',
  ENGAGEMENT_RECEIVED = 'engagement.received',
  ANALYTICS_UPDATED = 'analytics.updated'
}

export interface RetryConfig {
  maxAttempts: number;
  backoffMultiplier: number;
  initialDelay: number;
}

// ==================== API RESPONSES ====================

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: ApiError;
  metadata?: ResponseMetadata;
}

export interface ApiError {
  code: string;
  message: string;
  details?: any;
}

export interface ResponseMetadata {
  page?: number;
  limit?: number;
  total?: number;
  hasMore?: boolean;
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  metadata: ResponseMetadata;
}
