import OpenAI from 'openai';
import { GenerationParameters, GenerationResult, ContentType, PlatformType } from '@marketing-automation/types';
import { logger } from '../../utils/logger';

export class OpenAIProvider {
  private client: OpenAI;

  constructor() {
    this.client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
  }

  async generateContent(
    prompt: string,
    contentType: ContentType,
    platform: PlatformType,
    parameters: GenerationParameters
  ): Promise<GenerationResult> {
    const startTime = Date.now();

    try {
      const systemPrompt = this.buildSystemPrompt(contentType, platform, parameters);
      const userPrompt = this.buildUserPrompt(prompt, parameters);

      const response = await this.client.chat.completions.create({
        model: parameters.model === 'gpt-4' ? 'gpt-4-turbo-preview' : 'gpt-3.5-turbo',
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt }
        ],
        temperature: parameters.temperature || 0.7,
        max_tokens: this.getMaxTokens(contentType),
        n: parameters.variations || 1,
      });

      const content = response.choices.map(choice => choice.message.content || '');
      const processingTime = Date.now() - startTime;

      logger.info(`OpenAI generation completed in ${processingTime}ms`);

      return {
        content,
        metadata: {
          tokensUsed: response.usage?.total_tokens || 0,
          processingTime,
          model: response.model,
        }
      };
    } catch (error) {
      logger.error('OpenAI generation failed:', error);
      throw error;
    }
  }

  private buildSystemPrompt(
    contentType: ContentType,
    platform: PlatformType,
    parameters: GenerationParameters
  ): string {
    const basePrompt = `You are an expert marketing content creator specialized in ${contentType} for ${platform}.`;

    let instructions = [basePrompt];

    // Add tone and style
    if (parameters.tone) {
      instructions.push(`Write in a ${parameters.tone} tone.`);
    }

    if (parameters.style) {
      instructions.push(`Use a ${parameters.style} style.`);
    }

    // Platform-specific guidelines
    instructions.push(this.getPlatformGuidelines(platform));

    // Content type specific instructions
    instructions.push(this.getContentTypeInstructions(contentType));

    // Additional parameters
    if (parameters.includeHashtags) {
      instructions.push('Include relevant hashtags.');
    }

    if (parameters.includeEmojis) {
      instructions.push('Use appropriate emojis to enhance engagement.');
    }

    if (parameters.targetAudience) {
      instructions.push(`Target audience: ${parameters.targetAudience}`);
    }

    if (parameters.callToAction) {
      instructions.push(`Include this call-to-action: ${parameters.callToAction}`);
    }

    if (parameters.keywords && parameters.keywords.length > 0) {
      instructions.push(`Incorporate these keywords naturally: ${parameters.keywords.join(', ')}`);
    }

    return instructions.join('\n\n');
  }

  private buildUserPrompt(prompt: string, parameters: GenerationParameters): string {
    let userPrompt = prompt;

    if (parameters.length) {
      userPrompt += `\n\nTarget length: approximately ${parameters.length} words.`;
    }

    return userPrompt;
  }

  private getPlatformGuidelines(platform: PlatformType): string {
    const guidelines: Record<PlatformType, string> = {
      [PlatformType.INSTAGRAM]: 'Instagram best practices: Use visual-first language, keep captions concise (under 2,200 characters), front-load important content, use line breaks for readability, and include 5-15 relevant hashtags.',
      [PlatformType.FACEBOOK]: 'Facebook best practices: Write engaging, conversational content. Optimal length is 40-80 characters for maximum engagement, but can go longer. Use questions to drive comments, and create shareable, value-driven posts.',
      [PlatformType.TWITTER]: 'Twitter/X best practices: Keep posts under 280 characters, make the first few words count, use 1-2 hashtags max, tag relevant accounts, and create content that sparks conversation.',
      [PlatformType.LINKEDIN]: 'LinkedIn best practices: Write professional, value-driven content. Optimal posts are 1,300-1,600 characters. Start with a hook, provide insights, and end with a question or CTA. Use 3-5 hashtags.',
      [PlatformType.TIKTOK]: 'TikTok best practices: Create attention-grabbing, trend-aware content. Use casual, authentic language. Keep captions short and punchy. Include trending hashtags and sounds.',
      [PlatformType.YOUTUBE]: 'YouTube best practices: Create compelling titles and descriptions. Use keywords for SEO. Include timestamps for longer videos. Write engaging, detailed descriptions (200-300 words).',
      [PlatformType.PINTEREST]: 'Pinterest best practices: Create SEO-optimized, descriptive pins. Use keywords in titles and descriptions. Include clear CTAs. Descriptions should be 100-200 words.',
      [PlatformType.TWITCH]: 'Twitch best practices: Write casual, community-focused content. Use gamer-friendly language. Keep it authentic and engaging. Include relevant emotes.',
      [PlatformType.SNAPCHAT]: 'Snapchat best practices: Create urgent, FOMO-driven content. Use casual, conversational language. Keep it short and authentic.',
      [PlatformType.REDDIT]: 'Reddit best practices: Write authentic, value-first content. Avoid overly promotional language. Be helpful and honest. Follow subreddit-specific guidelines.',
      [PlatformType.EMAIL]: 'Email best practices: Write compelling subject lines (40-50 characters). Use personalization. Create scannable content with clear hierarchy. Include strong CTAs.',
      [PlatformType.BLOG]: 'Blog best practices: Write SEO-optimized, comprehensive content. Use headings, bullet points, and short paragraphs. Include internal and external links. Aim for 1,500+ words for in-depth topics.',
      [PlatformType.WEBSITE]: 'Website best practices: Write clear, concise copy. Focus on benefits over features. Use headings and white space. Include strong CTAs.',
    };

    return guidelines[platform] || 'Create engaging, high-quality content optimized for this platform.';
  }

  private getContentTypeInstructions(contentType: ContentType): string {
    const instructions: Record<ContentType, string> = {
      [ContentType.SOCIAL_POST]: 'Create an engaging social media post that drives engagement (likes, comments, shares). Make it relatable, valuable, or entertaining.',
      [ContentType.BLOG_ARTICLE]: 'Write a comprehensive, well-structured blog article with clear introduction, body sections with subheadings, and conclusion. Include SEO optimization.',
      [ContentType.EMAIL]: 'Write a compelling email with attention-grabbing subject line, personalized greeting, clear value proposition, and strong call-to-action.',
      [ContentType.VIDEO_SCRIPT]: 'Write a video script with clear intro hook, main content sections, transitions, and outro. Include visual and audio cues.',
      [ContentType.AD_COPY]: 'Create persuasive ad copy with attention-grabbing headline, clear benefits, social proof if relevant, and compelling CTA. Focus on conversions.',
      [ContentType.PRODUCT_DESCRIPTION]: 'Write compelling product descriptions highlighting features, benefits, use cases, and what makes it unique. Address customer pain points.',
      [ContentType.PRESS_RELEASE]: 'Write a professional press release following AP style. Include headline, dateline, lead paragraph with 5Ws, body, boilerplate, and contact info.',
      [ContentType.PODCAST_SCRIPT]: 'Write a podcast script with engaging intro, clear segment structure, conversation prompts, and outro. Make it conversational and natural.',
      [ContentType.LANDING_PAGE]: 'Write landing page copy with clear hero headline, value proposition, benefits, social proof, FAQs, and multiple CTAs. Focus on conversions.',
      [ContentType.NEWSLETTER]: 'Write an engaging newsletter with catchy subject line, personalized intro, curated content sections, and CTAs. Keep it scannable.',
    };

    return instructions[contentType] || 'Create high-quality, engaging content.';
  }

  private getMaxTokens(contentType: ContentType): number {
    const limits: Record<ContentType, number> = {
      [ContentType.SOCIAL_POST]: 500,
      [ContentType.BLOG_ARTICLE]: 4000,
      [ContentType.EMAIL]: 1000,
      [ContentType.VIDEO_SCRIPT]: 2000,
      [ContentType.AD_COPY]: 300,
      [ContentType.PRODUCT_DESCRIPTION]: 800,
      [ContentType.PRESS_RELEASE]: 1500,
      [ContentType.PODCAST_SCRIPT]: 2500,
      [ContentType.LANDING_PAGE]: 2000,
      [ContentType.NEWSLETTER]: 1500,
    };

    return limits[contentType] || 1000;
  }

  async generateImage(prompt: string, size: '1024x1024' | '1792x1024' | '1024x1792' = '1024x1024'): Promise<string> {
    try {
      const response = await this.client.images.generate({
        model: 'dall-e-3',
        prompt,
        n: 1,
        size,
        quality: 'hd',
      });

      return response.data[0].url || '';
    } catch (error) {
      logger.error('Image generation failed:', error);
      throw error;
    }
  }
}
