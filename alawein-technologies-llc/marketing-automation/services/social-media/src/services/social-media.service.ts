import { PlatformType, SocialPost, PostStatus, ContentPerformance } from '@marketing-automation/types';
import { InstagramProvider } from './platforms/instagram.provider';
import { FacebookProvider } from './platforms/facebook.provider';
import { TwitterProvider } from './platforms/twitter.provider';
import { LinkedInProvider } from './platforms/linkedin.provider';
import { TikTokProvider } from './platforms/tiktok.provider';
import { YouTubeProvider } from './platforms/youtube.provider';

export class SocialMediaService {
  private providers: Map<PlatformType, any>;

  constructor() {
    this.providers = new Map([
      [PlatformType.INSTAGRAM, new InstagramProvider()],
      [PlatformType.FACEBOOK, new FacebookProvider()],
      [PlatformType.TWITTER, new TwitterProvider()],
      [PlatformType.LINKEDIN, new LinkedInProvider()],
      [PlatformType.TIKTOK, new TikTokProvider()],
      [PlatformType.YOUTUBE, new YouTubeProvider()],
    ]);
  }

  async publishPost(post: SocialPost): Promise<SocialPost> {
    const provider = this.providers.get(post.platform);
    if (!provider) {
      throw new Error(`Platform ${post.platform} not supported`);
    }

    try {
      const platformPostId = await provider.publish(post);
      return {
        ...post,
        status: PostStatus.PUBLISHED,
        platformPostId,
        publishedAt: new Date()
      };
    } catch (error: any) {
      return {
        ...post,
        status: PostStatus.FAILED,
        error: error.message,
        retryCount: post.retryCount + 1
      };
    }
  }

  async schedulePost(post: SocialPost): Promise<SocialPost> {
    // Add to scheduling queue
    return {
      ...post,
      status: PostStatus.SCHEDULED
    };
  }

  async deletePost(post: SocialPost): Promise<boolean> {
    const provider = this.providers.get(post.platform);
    if (!provider || !post.platformPostId) {
      return false;
    }

    return provider.delete(post.platformPostId);
  }

  async getPostPerformance(post: SocialPost): Promise<ContentPerformance> {
    const provider = this.providers.get(post.platform);
    if (!provider || !post.platformPostId) {
      throw new Error('Cannot fetch performance for unpublished post');
    }

    return provider.getPerformance(post.platformPostId);
  }

  async replyToComment(platform: PlatformType, commentId: string, message: string): Promise<void> {
    const provider = this.providers.get(platform);
    if (!provider) {
      throw new Error(`Platform ${platform} not supported`);
    }

    await provider.replyToComment(commentId, message);
  }

  async likePost(platform: PlatformType, postId: string): Promise<void> {
    const provider = this.providers.get(platform);
    if (!provider) {
      throw new Error(`Platform ${platform} not supported`);
    }

    await provider.like(postId);
  }

  async followUser(platform: PlatformType, userId: string): Promise<void> {
    const provider = this.providers.get(platform);
    if (!provider) {
      throw new Error(`Platform ${platform} not supported`);
    }

    await provider.follow(userId);
  }

  async getAccountAnalytics(platform: PlatformType, accountId: string, dateRange: {start: Date, end: Date}): Promise<any> {
    const provider = this.providers.get(platform);
    if (!provider) {
      throw new Error(`Platform ${platform} not supported`);
    }

    return provider.getAnalytics(accountId, dateRange);
  }

  async searchHashtags(platform: PlatformType, hashtag: string): Promise<any> {
    const provider = this.providers.get(platform);
    if (!provider) {
      throw new Error(`Platform ${platform} not supported`);
    }

    return provider.searchHashtag(hashtag);
  }

  async getOptimalPostingTimes(platform: PlatformType, accountId: string): Promise<{ dayOfWeek: number; hour: number; score: number }[]> {
    // Analyze historical performance to determine optimal posting times
    const provider = this.providers.get(platform);
    if (!provider) {
      throw new Error(`Platform ${platform} not supported`);
    }

    return provider.getOptimalTimes(accountId);
  }

  async autoModerateComments(platform: PlatformType, postId: string, rules: any): Promise<void> {
    const provider = this.providers.get(platform);
    if (!provider) {
      throw new Error(`Platform ${platform} not supported`);
    }

    await provider.moderateComments(postId, rules);
  }
}

export const socialMediaService = new SocialMediaService();
