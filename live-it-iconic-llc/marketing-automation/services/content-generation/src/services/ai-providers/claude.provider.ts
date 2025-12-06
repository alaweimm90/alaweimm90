import Anthropic from '@anthropic-ai/sdk';
import { GenerationParameters, GenerationResult, ContentType, PlatformType } from '@marketing-automation/types';
import { logger } from '../../utils/logger';

export class ClaudeProvider {
  private client: Anthropic;

  constructor() {
    this.client = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
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

      const variations = parameters.variations || 1;
      const results: string[] = [];

      // Generate multiple variations sequentially
      for (let i = 0; i < variations; i++) {
        const response = await this.client.messages.create({
          model: 'claude-3-5-sonnet-20241022',
          max_tokens: this.getMaxTokens(contentType),
          temperature: parameters.temperature || 0.7,
          system: systemPrompt,
          messages: [
            {
              role: 'user',
              content: userPrompt + (variations > 1 ? `\n\nVariation ${i + 1}:` : '')
            }
          ]
        });

        const content = response.content[0];
        if (content.type === 'text') {
          results.push(content.text);
        }
      }

      const processingTime = Date.now() - startTime;

      logger.info(`Claude generation completed in ${processingTime}ms`);

      return {
        content: results,
        metadata: {
          tokensUsed: 0, // Claude API doesn't return token count in the same way
          processingTime,
          model: 'claude-3-5-sonnet-20241022',
        }
      };
    } catch (error) {
      logger.error('Claude generation failed:', error);
      throw error;
    }
  }

  private buildSystemPrompt(
    contentType: ContentType,
    platform: PlatformType,
    parameters: GenerationParameters
  ): string {
    const basePrompt = `You are an expert marketing content creator with deep expertise in ${contentType} for ${platform}.`;

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
      instructions.push('Include relevant, high-performing hashtags.');
    }

    if (parameters.includeEmojis) {
      instructions.push('Use appropriate emojis strategically to enhance engagement.');
    }

    if (parameters.targetAudience) {
      instructions.push(`Target audience: ${parameters.targetAudience}. Tailor the language and messaging accordingly.`);
    }

    if (parameters.callToAction) {
      instructions.push(`Include this call-to-action: ${parameters.callToAction}`);
    }

    if (parameters.keywords && parameters.keywords.length > 0) {
      instructions.push(`Incorporate these keywords naturally and strategically: ${parameters.keywords.join(', ')}`);
    }

    instructions.push('Focus on quality, engagement, and conversion potential.');

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
      [PlatformType.INSTAGRAM]: 'Instagram optimization: Create visually-descriptive, engaging captions. Use line breaks for readability. First line should hook attention. Include 10-15 strategic hashtags. Encourage saves and shares through valuable content.',
      [PlatformType.FACEBOOK]: 'Facebook optimization: Create shareable, discussion-worthy content. Use emotional triggers. Ask questions to boost comments. Optimize for both desktop and mobile. Consider algorithm preferences for meaningful interactions.',
      [PlatformType.TWITTER]: 'Twitter/X optimization: Create concise, impactful messages. Use threads for longer content. Tag strategically. Use 1-2 hashtags max. Create tweetable moments. Focus on retweets and replies.',
      [PlatformType.LINKEDIN]: 'LinkedIn optimization: Provide professional value and insights. Use data and statistics. Share lessons learned. Start strong to encourage "see more" clicks. Use industry-relevant hashtags. Build authority.',
      [PlatformType.TIKTOK]: 'TikTok optimization: Hook viewers in first 3 seconds. Use trending sounds and hashtags. Keep it authentic and relatable. Encourage duets and stitches. Create shareable, entertaining content.',
      [PlatformType.YOUTUBE]: 'YouTube optimization: Create SEO-rich titles (60 chars). Write comprehensive descriptions (200-300 words). Use strategic keywords. Include timestamps. Add card and end screen suggestions.',
      [PlatformType.PINTEREST]: 'Pinterest optimization: Create keyword-rich, SEO-optimized descriptions. Use 500 character max. Include relevant boards. Add strong CTAs. Focus on searchability.',
      [PlatformType.TWITCH]: 'Twitch optimization: Be authentic and community-focused. Use channel-specific emotes. Create anticipation for streams. Engage with community culture. Keep it real and relatable.',
      [PlatformType.SNAPCHAT]: 'Snapchat optimization: Create FOMO and urgency. Use casual, friend-to-friend language. Keep it exclusive and timely. Encourage screenshots and shares.',
      [PlatformType.REDDIT]: 'Reddit optimization: Provide genuine value first. Be transparent about commercial intent. Follow subreddit rules strictly. Use appropriate tone for each community. Avoid marketing-speak.',
      [PlatformType.EMAIL]: 'Email optimization: Write compelling, personalized subject lines (40-50 chars). Use preview text strategically. Create scannable content. Use multiple CTAs. Optimize for mobile. Test everything.',
      [PlatformType.BLOG]: 'Blog optimization: Write comprehensive, SEO-optimized content. Use H2/H3 headings. Include meta description. Add internal links. Use bullet points and visuals. Aim for featured snippets.',
      [PlatformType.WEBSITE]: 'Website optimization: Write clear, benefit-focused copy. Use F-pattern layout. Include trust signals. Optimize for conversions. Use action-oriented language.',
    };

    return guidelines[platform] || 'Create high-quality, platform-optimized content.';
  }

  private getContentTypeInstructions(contentType: ContentType): string {
    const instructions: Record<ContentType, string> = {
      [ContentType.SOCIAL_POST]: 'Create an engaging social media post that maximizes engagement. Use psychological triggers, create curiosity, provide value, or entertain. Make it shareable.',
      [ContentType.BLOG_ARTICLE]: 'Write a comprehensive, well-researched blog article. Use clear structure with H2/H3 subheadings. Include introduction with hook, detailed body sections, and conclusion with CTA. Optimize for SEO and readability.',
      [ContentType.EMAIL]: 'Write a high-converting email. Compelling subject line that drives opens. Personalized, value-first content. Clear hierarchy. Multiple CTAs. Mobile-optimized. Address pain points and benefits.',
      [ContentType.VIDEO_SCRIPT]: 'Write an engaging video script. Strong hook in first 5 seconds. Clear structure with intro, main content, and outro. Include visual cues, B-roll suggestions, and music notes. Keep language conversational.',
      [ContentType.AD_COPY]: 'Create high-converting ad copy. Attention-grabbing headline. Clear unique value proposition. Address objections. Include social proof or statistics. Strong, action-oriented CTA. Focus on benefits over features.',
      [ContentType.PRODUCT_DESCRIPTION]: 'Write compelling, conversion-focused product descriptions. Lead with primary benefit. Address customer pain points. Highlight unique features. Use sensory language. Include specifications. Add urgency if appropriate.',
      [ContentType.PRESS_RELEASE]: 'Write a professional, newsworthy press release. AP style formatting. Compelling headline and subheading. Strong lead paragraph with 5Ws. Supporting paragraphs with quotes. Company boilerplate. Contact information.',
      [ContentType.PODCAST_SCRIPT]: 'Write a natural, conversational podcast script. Engaging intro with hook. Clear segment structure. Interview questions or talking points. Natural transitions. Outro with CTAs. Include timing notes.',
      [ContentType.LANDING_PAGE]: 'Write high-converting landing page copy. Compelling hero headline. Clear value proposition above fold. Benefit-focused sections. Social proof and testimonials. FAQ section. Multiple CTAs. Urgency/scarcity elements.',
      [ContentType.NEWSLETTER]: 'Write an engaging newsletter. Catchy subject line. Personal intro that builds connection. Curated content sections with summaries. Internal linking. Clear CTAs for each section. Sign-off with personality.',
    };

    return instructions[contentType] || 'Create exceptional, high-quality content.';
  }

  private getMaxTokens(contentType: ContentType): number {
    const limits: Record<ContentType, number> = {
      [ContentType.SOCIAL_POST]: 600,
      [ContentType.BLOG_ARTICLE]: 4096,
      [ContentType.EMAIL]: 1200,
      [ContentType.VIDEO_SCRIPT]: 2500,
      [ContentType.AD_COPY]: 400,
      [ContentType.PRODUCT_DESCRIPTION]: 1000,
      [ContentType.PRESS_RELEASE]: 1800,
      [ContentType.PODCAST_SCRIPT]: 3000,
      [ContentType.LANDING_PAGE]: 2500,
      [ContentType.NEWSLETTER]: 1800,
    };

    return limits[contentType] || 1200;
  }
}
