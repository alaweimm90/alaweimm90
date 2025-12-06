import Bull from 'bull';
import cron from 'node-cron';
import { DateTime } from 'luxon';
import {
  SocialPost,
  EmailCampaign,
  Campaign,
  PlatformType,
  Workflow,
  WorkflowTrigger,
  TriggerType
} from '@marketing-automation/types';

export class SchedulingService {
  private postQueue: Bull.Queue;
  private emailQueue: Bull.Queue;
  private workflowQueue: Bull.Queue;

  constructor() {
    const redisConfig = {
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT || '6379')
    };

    this.postQueue = new Bull('social-posts', { redis: redisConfig });
    this.emailQueue = new Bull('emails', { redis: redisConfig });
    this.workflowQueue = new Bull('workflows', { redis: redisConfig });

    this.initializeQueues();
  }

  private initializeQueues(): void {
    // Process social media posts
    this.postQueue.process(async (job) => {
      const post: SocialPost = job.data;
      // Call social media service to publish
      console.log(`Publishing post: ${post.id}`);
    });

    // Process emails
    this.emailQueue.process(async (job) => {
      const campaign: EmailCampaign = job.data;
      // Call email service to send
      console.log(`Sending email campaign: ${campaign.id}`);
    });

    // Process workflows
    this.workflowQueue.process(async (job) => {
      const workflow: Workflow = job.data;
      await this.executeWorkflow(workflow);
    });
  }

  async scheduleSocialPost(post: SocialPost, publishAt: Date): Promise<void> {
    const delay = publishAt.getTime() - Date.now();

    await this.postQueue.add(post, {
      delay: delay > 0 ? delay : 0,
      attempts: 3,
      backoff: {
        type: 'exponential',
        delay: 5000
      }
    });
  }

  async scheduleEmailCampaign(campaign: EmailCampaign, sendAt: Date): Promise<void> {
    const delay = sendAt.getTime() - Date.now();

    await this.emailQueue.add(campaign, {
      delay: delay > 0 ? delay : 0,
      attempts: 3
    });
  }

  async scheduleRecurring(params: {
    type: 'post' | 'email';
    data: any;
    cronExpression: string;
  }): Promise<void> {
    cron.schedule(params.cronExpression, async () => {
      if (params.type === 'post') {
        await this.postQueue.add(params.data);
      } else {
        await this.emailQueue.add(params.data);
      }
    });
  }

  async calculateOptimalPostTime(
    platform: PlatformType,
    accountId: string,
    timezone: string
  ): Promise<Date> {
    // Analyze historical performance data
    const optimalTimes = await this.getOptimalTimesForPlatform(platform, accountId);

    // Get current time in account's timezone
    const now = DateTime.now().setZone(timezone);

    // Find next optimal time slot
    const nextOptimal = this.findNextOptimalTime(now, optimalTimes);

    return nextOptimal.toJSDate();
  }

  private async getOptimalTimesForPlatform(
    platform: PlatformType,
    accountId: string
  ): Promise<{ dayOfWeek: number; hour: number; score: number }[]> {
    // Query analytics database for historical performance
    // Return times with highest engagement
    return [
      { dayOfWeek: 1, hour: 10, score: 95 },
      { dayOfWeek: 3, hour: 14, score: 92 },
      { dayOfWeek: 5, hour: 18, score: 90 }
    ];
  }

  private findNextOptimalTime(
    now: DateTime,
    optimalTimes: { dayOfWeek: number; hour: number; score: number }[]
  ): DateTime {
    // Sort by score
    const sorted = optimalTimes.sort((a, b) => b.score - a.score);

    // Find next occurrence of top optimal time
    let nextTime = now;
    const topTime = sorted[0];

    while (nextTime.weekday !== topTime.dayOfWeek || nextTime.hour !== topTime.hour) {
      nextTime = nextTime.plus({ hours: 1 });
    }

    return nextTime;
  }

  async createContentCalendar(params: {
    startDate: Date;
    endDate: Date;
    platforms: PlatformType[];
    postsPerWeek: number;
    timezone: string;
  }): Promise<any[]> {
    const calendar = [];
    const start = DateTime.fromJSDate(params.startDate).setZone(params.timezone);
    const end = DateTime.fromJSDate(params.endDate).setZone(params.timezone);

    let current = start;

    while (current < end) {
      for (const platform of params.platforms) {
        const optimalTime = await this.calculateOptimalPostTime(
          platform,
          'default',
          params.timezone
        );

        calendar.push({
          platform,
          scheduledFor: optimalTime,
          status: 'pending'
        });
      }

      current = current.plus({ weeks: 1 });
    }

    return calendar;
  }

  async scheduleWorkflow(workflow: Workflow): Promise<void> {
    switch (workflow.trigger.type) {
      case TriggerType.SCHEDULE:
        // Schedule based on cron expression
        const cronExpression = workflow.trigger.config.cronExpression;
        cron.schedule(cronExpression, async () => {
          await this.workflowQueue.add(workflow);
        });
        break;

      case TriggerType.WEBHOOK:
        // Webhook triggers are handled by API endpoints
        break;

      case TriggerType.EVENT:
        // Event-based triggers
        break;

      case TriggerType.MANUAL:
        // Manual triggers don't need scheduling
        break;
    }
  }

  private async executeWorkflow(workflow: Workflow): Promise<void> {
    for (const action of workflow.actions.sort((a, b) => a.order - b.order)) {
      // Check conditions
      const conditionsMet = this.evaluateConditions(workflow.conditions, {});

      if (!conditionsMet) {
        continue;
      }

      // Execute action
      await this.executeAction(action);
    }
  }

  private evaluateConditions(conditions: any[], context: any): boolean {
    if (conditions.length === 0) return true;

    return conditions.every(condition => {
      const value = context[condition.field];
      switch (condition.operator) {
        case 'equals':
          return value === condition.value;
        case 'contains':
          return value?.includes(condition.value);
        case 'greater_than':
          return value > condition.value;
        case 'less_than':
          return value < condition.value;
        default:
          return false;
      }
    });
  }

  private async executeAction(action: any): Promise<void> {
    switch (action.type) {
      case 'generate_content':
        // Call content generation service
        break;
      case 'post_social':
        // Call social media service
        break;
      case 'send_email':
        // Call email service
        break;
      case 'wait':
        await new Promise(resolve => setTimeout(resolve, action.config.duration));
        break;
    }
  }

  async pauseScheduledItem(itemId: string, queueName: string): Promise<void> {
    const queue = this.getQueue(queueName);
    const job = await queue.getJob(itemId);
    if (job) {
      await job.remove();
    }
  }

  async rescheduleItem(itemId: string, queueName: string, newDate: Date): Promise<void> {
    const queue = this.getQueue(queueName);
    const job = await queue.getJob(itemId);

    if (job) {
      await job.remove();
      const delay = newDate.getTime() - Date.now();
      await queue.add(job.data, { delay: delay > 0 ? delay : 0 });
    }
  }

  private getQueue(queueName: string): Bull.Queue {
    switch (queueName) {
      case 'posts':
        return this.postQueue;
      case 'emails':
        return this.emailQueue;
      case 'workflows':
        return this.workflowQueue;
      default:
        throw new Error(`Unknown queue: ${queueName}`);
    }
  }

  async getScheduledItems(queueName: string): Promise<any[]> {
    const queue = this.getQueue(queueName);
    const jobs = await queue.getDelayed();
    return jobs.map(job => ({
      id: job.id,
      data: job.data,
      scheduledFor: new Date(job.timestamp + (job.opts.delay || 0))
    }));
  }
}

export const schedulingService = new SchedulingService();
