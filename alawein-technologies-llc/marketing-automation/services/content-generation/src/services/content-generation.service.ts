import {
  GenerationRequest,
  GenerationResult,
  GenerationStatus,
  ContentType,
  PlatformType,
  GenerationParameters,
  VoiceAndTone
} from '@marketing-automation/types';
import { OpenAIProvider } from './ai-providers/openai.provider';
import { ClaudeProvider } from './ai-providers/claude.provider';
import { logger } from '../utils/logger';

export class ContentGenerationService {
  private openaiProvider: OpenAIProvider;
  private claudeProvider: ClaudeProvider;

  constructor() {
    this.openaiProvider = new OpenAIProvider();
    this.claudeProvider = new ClaudeProvider();
  }

  async generateContent(request: GenerationRequest): Promise<GenerationResult> {
    try {
      logger.info(`Generating content: ${request.type} for ${request.platform}`);

      const provider = this.selectProvider(request.parameters.model);
      const enhancedPrompt = this.enhancePrompt(request);

      const result = await provider.generateContent(
        enhancedPrompt,
        request.type,
        request.platform,
        request.parameters
      );

      // Post-process the content
      result.content = result.content.map(content =>
        this.postProcessContent(content, request)
      );

      return result;
    } catch (error) {
      logger.error('Content generation failed:', error);
      throw error;
    }
  }

  private selectProvider(model?: string) {
    if (model === 'claude') {
      return this.claudeProvider;
    }
    return this.openaiProvider;
  }

  private enhancePrompt(request: GenerationRequest): string {
    let prompt = request.prompt;

    // Add brand voice if available
    if (request.brandId) {
      // In production, fetch brand voice from database
      // For now, we'll use the basic prompt
    }

    return prompt;
  }

  private postProcessContent(content: string, request: GenerationRequest): string {
    let processed = content;

    // Platform-specific post-processing
    switch (request.platform) {
      case PlatformType.TWITTER:
        processed = this.ensureTwitterLength(processed);
        break;
      case PlatformType.INSTAGRAM:
        processed = this.formatInstagramCaption(processed);
        break;
      case PlatformType.LINKEDIN:
        processed = this.formatLinkedInPost(processed);
        break;
    }

    return processed.trim();
  }

  private ensureTwitterLength(content: string): string {
    if (content.length > 280) {
      return content.substring(0, 277) + '...';
    }
    return content;
  }

  private formatInstagramCaption(content: string): string {
    // Add line breaks for better readability
    const paragraphs = content.split('\n\n');
    return paragraphs.join('\n.\n');
  }

  private formatLinkedInPost(content: string): string {
    // Ensure proper formatting for LinkedIn
    return content;
  }

  // ==================== SPECIALIZED CONTENT GENERATORS ====================

  async generateSocialMediaPost(params: {
    platform: PlatformType;
    topic: string;
    tone?: string;
    includeHashtags?: boolean;
    includeEmojis?: boolean;
    targetAudience?: string;
  }): Promise<string[]> {
    const request: GenerationRequest = {
      id: this.generateId(),
      organizationId: '',
      brandId: '',
      type: ContentType.SOCIAL_POST,
      platform: params.platform,
      prompt: `Create an engaging social media post about: ${params.topic}`,
      parameters: {
        tone: params.tone,
        includeHashtags: params.includeHashtags ?? true,
        includeEmojis: params.includeEmojis ?? true,
        targetAudience: params.targetAudience,
        variations: 3
      },
      status: GenerationStatus.PENDING,
      createdAt: new Date()
    };

    const result = await this.generateContent(request);
    return result.content;
  }

  async generateBlogArticle(params: {
    topic: string;
    keywords: string[];
    targetLength: number;
    tone?: string;
    includeOutline?: boolean;
  }): Promise<string> {
    const prompt = params.includeOutline
      ? `Write a comprehensive blog article about "${params.topic}". Include an outline with H2/H3 headings, introduction, detailed body sections, and conclusion.`
      : `Write a comprehensive blog article about "${params.topic}".`;

    const request: GenerationRequest = {
      id: this.generateId(),
      organizationId: '',
      brandId: '',
      type: ContentType.BLOG_ARTICLE,
      platform: PlatformType.BLOG,
      prompt,
      parameters: {
        tone: params.tone,
        length: params.targetLength,
        keywords: params.keywords,
        variations: 1
      },
      status: GenerationStatus.PENDING,
      createdAt: new Date()
    };

    const result = await this.generateContent(request);
    return result.content[0];
  }

  async generateEmailCampaign(params: {
    type: 'promotional' | 'newsletter' | 'drip' | 'welcome';
    subject: string;
    topic: string;
    tone?: string;
    includePersonalization?: boolean;
  }): Promise<{ subject: string; preview: string; body: string }> {
    const prompt = `Create an email campaign (${params.type}) about: ${params.topic}
    ${params.includePersonalization ? 'Include personalization tokens like {firstName}, {company}, etc.' : ''}`;

    const request: GenerationRequest = {
      id: this.generateId(),
      organizationId: '',
      brandId: '',
      type: ContentType.EMAIL,
      platform: PlatformType.EMAIL,
      prompt,
      parameters: {
        tone: params.tone,
        variations: 1
      },
      status: GenerationStatus.PENDING,
      createdAt: new Date()
    };

    const result = await this.generateContent(request);
    const content = result.content[0];

    // Parse email content (in production, use more sophisticated parsing)
    return {
      subject: params.subject,
      preview: content.substring(0, 100),
      body: content
    };
  }

  async generateVideoScript(params: {
    platform: PlatformType.YOUTUBE | PlatformType.TIKTOK | PlatformType.INSTAGRAM;
    topic: string;
    duration: number; // in seconds
    style: 'educational' | 'entertaining' | 'promotional';
  }): Promise<string> {
    const prompt = `Write a ${params.style} video script for ${params.platform} about: ${params.topic}
    Duration: approximately ${params.duration} seconds
    Include: hook, main content, transitions, and outro
    Add visual and audio cues where appropriate`;

    const request: GenerationRequest = {
      id: this.generateId(),
      organizationId: '',
      brandId: '',
      type: ContentType.VIDEO_SCRIPT,
      platform: params.platform,
      prompt,
      parameters: {
        style: params.style,
        variations: 1
      },
      status: GenerationStatus.PENDING,
      createdAt: new Date()
    };

    const result = await this.generateContent(request);
    return result.content[0];
  }

  async generateAdCopy(params: {
    platform: 'google' | 'facebook' | 'instagram' | 'linkedin';
    objective: 'awareness' | 'consideration' | 'conversion';
    product: string;
    targetAudience: string;
    uniqueSellingPoint: string;
  }): Promise<{ headline: string; description: string; cta: string }[]> {
    const prompt = `Create ad copy for ${params.platform} ads targeting ${params.targetAudience}
    Product/Service: ${params.product}
    Objective: ${params.objective}
    Unique Selling Point: ${params.uniqueSellingPoint}
    Include compelling headlines, descriptions, and strong CTAs`;

    const request: GenerationRequest = {
      id: this.generateId(),
      organizationId: '',
      brandId: '',
      type: ContentType.AD_COPY,
      platform: this.mapAdPlatform(params.platform),
      prompt,
      parameters: {
        targetAudience: params.targetAudience,
        variations: 5 // Generate multiple ad variations
      },
      status: GenerationStatus.PENDING,
      createdAt: new Date()
    };

    const result = await this.generateContent(request);

    // Parse ad copy variations
    return result.content.map(content => this.parseAdCopy(content));
  }

  async generateProductDescription(params: {
    productName: string;
    features: string[];
    benefits: string[];
    targetAudience: string;
    platform: 'ecommerce' | 'amazon' | 'shopify';
  }): Promise<string> {
    const prompt = `Write a compelling product description for "${params.productName}"
    Features: ${params.features.join(', ')}
    Benefits: ${params.benefits.join(', ')}
    Target Audience: ${params.targetAudience}
    Platform: ${params.platform}
    Focus on benefits, address pain points, and create desire`;

    const request: GenerationRequest = {
      id: this.generateId(),
      organizationId: '',
      brandId: '',
      type: ContentType.PRODUCT_DESCRIPTION,
      platform: PlatformType.WEBSITE,
      prompt,
      parameters: {
        targetAudience: params.targetAudience,
        variations: 1
      },
      status: GenerationStatus.PENDING,
      createdAt: new Date()
    };

    const result = await this.generateContent(request);
    return result.content[0];
  }

  async generatePressRelease(params: {
    headline: string;
    announcement: string;
    companyInfo: {
      name: string;
      description: string;
      contact: string;
    };
    quotes?: { person: string; title: string; quote: string }[];
  }): Promise<string> {
    const quotesText = params.quotes?.map(q =>
      `${q.person}, ${q.title}, said: "${q.quote}"`
    ).join('\n') || '';

    const prompt = `Write a professional press release:
    Headline: ${params.headline}
    Announcement: ${params.announcement}
    Company: ${params.companyInfo.name} - ${params.companyInfo.description}
    ${quotesText ? `Quotes:\n${quotesText}` : ''}

    Follow AP style. Include dateline, lead paragraph with 5Ws, body paragraphs, boilerplate, and contact info.`;

    const request: GenerationRequest = {
      id: this.generateId(),
      organizationId: '',
      brandId: '',
      type: ContentType.PRESS_RELEASE,
      platform: PlatformType.WEBSITE,
      prompt,
      parameters: {
        style: 'professional',
        variations: 1
      },
      status: GenerationStatus.PENDING,
      createdAt: new Date()
    };

    const result = await this.generateContent(request);
    return result.content[0];
  }

  async generatePodcastScript(params: {
    episodeTitle: string;
    topic: string;
    format: 'solo' | 'interview' | 'panel';
    duration: number;
    guests?: { name: string; bio: string }[];
  }): Promise<string> {
    const guestsInfo = params.guests?.map(g =>
      `${g.name} - ${g.bio}`
    ).join('\n') || '';

    const prompt = `Write a podcast script for "${params.episodeTitle}"
    Format: ${params.format}
    Duration: ${params.duration} minutes
    Topic: ${params.topic}
    ${guestsInfo ? `Guests:\n${guestsInfo}` : ''}

    Include: intro hook, main segments, questions/talking points, transitions, and outro`;

    const request: GenerationRequest = {
      id: this.generateId(),
      organizationId: '',
      brandId: '',
      type: ContentType.PODCAST_SCRIPT,
      platform: PlatformType.WEBSITE,
      prompt,
      parameters: {
        variations: 1
      },
      status: GenerationStatus.PENDING,
      createdAt: new Date()
    };

    const result = await this.generateContent(request);
    return result.content[0];
  }

  // ==================== BULK OPERATIONS ====================

  async generateContentCalendar(params: {
    platforms: PlatformType[];
    topics: string[];
    startDate: Date;
    endDate: Date;
    postsPerWeek: number;
  }): Promise<any[]> {
    // Generate a month's worth of content across multiple platforms
    const calendar = [];
    const days = Math.ceil((params.endDate.getTime() - params.startDate.getTime()) / (1000 * 60 * 60 * 24));

    // This would generate scheduled content for the entire period
    // Implementation would involve complex scheduling logic

    return calendar;
  }

  async generateCampaignContent(params: {
    campaignName: string;
    platforms: PlatformType[];
    theme: string;
    duration: number;
  }): Promise<any> {
    // Generate complete campaign content across all platforms
    return {};
  }

  // ==================== UTILITY METHODS ====================

  private mapAdPlatform(platform: string): PlatformType {
    const mapping: Record<string, PlatformType> = {
      'google': PlatformType.WEBSITE,
      'facebook': PlatformType.FACEBOOK,
      'instagram': PlatformType.INSTAGRAM,
      'linkedin': PlatformType.LINKEDIN
    };
    return mapping[platform] || PlatformType.WEBSITE;
  }

  private parseAdCopy(content: string): { headline: string; description: string; cta: string } {
    // Simple parsing - in production, use more sophisticated methods
    const lines = content.split('\n').filter(line => line.trim());
    return {
      headline: lines[0] || '',
      description: lines.slice(1, -1).join(' ') || '',
      cta: lines[lines.length - 1] || 'Learn More'
    };
  }

  private generateId(): string {
    return `gen_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

export const contentGenerationService = new ContentGenerationService();
