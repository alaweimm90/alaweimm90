/**
 * LiveItIconic Launch Platform - Event Bus
 *
 * Centralized pub/sub system for agent communication and coordination
 */

import { AgentMessage, MessageType, AgentPriority } from '../types';

export type EventHandler = (message: AgentMessage) => void | Promise<void>;

export interface Subscription {
  id: string;
  topic: string;
  handler: EventHandler;
  priority: AgentPriority;
}

export class EventBus {
  private subscriptions: Map<string, Subscription[]> = new Map();
  private messageQueue: AgentMessage[] = [];
  private processing: boolean = false;
  private messageHistory: AgentMessage[] = [];
  private maxHistorySize: number = 1000;

  /**
   * Subscribe to a topic
   */
  subscribe(
    topic: string,
    handler: EventHandler,
    priority: AgentPriority = AgentPriority.NORMAL
  ): string {
    const subscriptionId = `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    const subscription: Subscription = {
      id: subscriptionId,
      topic,
      handler,
      priority,
    };

    if (!this.subscriptions.has(topic)) {
      this.subscriptions.set(topic, []);
    }

    const subs = this.subscriptions.get(topic)!;
    subs.push(subscription);

    // Sort by priority
    subs.sort((a, b) => {
      const priorityOrder = {
        [AgentPriority.CRITICAL]: 0,
        [AgentPriority.HIGH]: 1,
        [AgentPriority.NORMAL]: 2,
        [AgentPriority.LOW]: 3,
      };
      return priorityOrder[a.priority] - priorityOrder[b.priority];
    });

    console.log(`[EventBus] Subscribed to topic: ${topic} (${subscriptionId})`);
    return subscriptionId;
  }

  /**
   * Unsubscribe from a topic
   */
  unsubscribe(subscriptionId: string): boolean {
    for (const [topic, subs] of this.subscriptions.entries()) {
      const index = subs.findIndex(s => s.id === subscriptionId);
      if (index !== -1) {
        subs.splice(index, 1);
        console.log(`[EventBus] Unsubscribed: ${subscriptionId} from ${topic}`);

        // Clean up empty topic arrays
        if (subs.length === 0) {
          this.subscriptions.delete(topic);
        }

        return true;
      }
    }
    return false;
  }

  /**
   * Publish a message to a topic
   */
  async publish(message: AgentMessage): Promise<void> {
    console.log(
      `[EventBus] Publishing message from ${message.from} to ${message.to} (${message.type})`
    );

    // Add to message history
    this.messageHistory.push(message);
    if (this.messageHistory.length > this.maxHistorySize) {
      this.messageHistory.shift();
    }

    // Determine topic(s)
    const topics = Array.isArray(message.to) ? message.to : [message.to];

    // Add to queue
    this.messageQueue.push(message);

    // Process queue
    if (!this.processing) {
      await this.processQueue();
    }
  }

  /**
   * Process message queue
   */
  private async processQueue(): Promise<void> {
    if (this.processing || this.messageQueue.length === 0) {
      return;
    }

    this.processing = true;

    while (this.messageQueue.length > 0) {
      // Sort by priority
      this.messageQueue.sort((a, b) => {
        const priorityOrder = {
          [AgentPriority.CRITICAL]: 0,
          [AgentPriority.HIGH]: 1,
          [AgentPriority.NORMAL]: 2,
          [AgentPriority.LOW]: 3,
        };
        return priorityOrder[a.priority] - priorityOrder[b.priority];
      });

      const message = this.messageQueue.shift()!;
      await this.deliverMessage(message);
    }

    this.processing = false;
  }

  /**
   * Deliver message to all subscribers
   */
  private async deliverMessage(message: AgentMessage): Promise<void> {
    const recipients = Array.isArray(message.to) ? message.to : [message.to];

    for (const recipient of recipients) {
      const subscriptions = this.subscriptions.get(recipient) || [];

      for (const subscription of subscriptions) {
        try {
          await subscription.handler(message);

          // If message requires acknowledgment, send response
          if (message.requiresAck) {
            const ack: AgentMessage = {
              id: `ack_${message.id}`,
              from: recipient,
              to: message.from,
              type: MessageType.RESPONSE,
              priority: message.priority,
              payload: { acknowledged: true, originalMessageId: message.id },
              timestamp: new Date(),
              requiresAck: false,
            };
            this.messageHistory.push(ack);
          }
        } catch (error) {
          console.error(`[EventBus] Error delivering message to ${recipient}:`, error);

          // Send error response if acknowledgment was required
          if (message.requiresAck) {
            const errorResponse: AgentMessage = {
              id: `error_${message.id}`,
              from: recipient,
              to: message.from,
              type: MessageType.ERROR,
              priority: AgentPriority.HIGH,
              payload: {
                error: error instanceof Error ? error.message : String(error),
                originalMessageId: message.id,
              },
              timestamp: new Date(),
              requiresAck: false,
            };
            this.messageHistory.push(errorResponse);
          }
        }
      }
    }
  }

  /**
   * Broadcast message to all subscribers of a topic
   */
  async broadcast(topic: string, payload: Record<string, unknown>, from: string): Promise<void> {
    const message: AgentMessage = {
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      from,
      to: topic,
      type: MessageType.BROADCAST,
      priority: AgentPriority.NORMAL,
      payload,
      timestamp: new Date(),
      requiresAck: false,
    };

    await this.publish(message);
  }

  /**
   * Request-response pattern
   */
  async request(to: string, payload: Record<string, unknown>, from: string, timeout: number = 30000): Promise<unknown> {
    return new Promise((resolve, reject) => {
      const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      // Set up response handler
      const responseHandler = (message: AgentMessage) => {
        if (
          message.type === MessageType.RESPONSE &&
          message.payload?.originalMessageId === requestId
        ) {
          this.unsubscribe(subscriptionId);
          clearTimeout(timeoutHandle);
          resolve(message.payload);
        }
      };

      const subscriptionId = this.subscribe(from, responseHandler, AgentPriority.HIGH);

      // Set timeout
      const timeoutHandle = setTimeout(() => {
        this.unsubscribe(subscriptionId);
        reject(new Error(`Request timeout after ${timeout}ms`));
      }, timeout);

      // Send request
      const message: AgentMessage = {
        id: requestId,
        from,
        to,
        type: MessageType.REQUEST,
        priority: AgentPriority.NORMAL,
        payload,
        timestamp: new Date(),
        requiresAck: true,
      };

      this.publish(message);
    });
  }

  /**
   * Get message history
   */
  getMessageHistory(filter?: {
    from?: string;
    to?: string;
    type?: MessageType;
    since?: Date;
  }): AgentMessage[] {
    let history = [...this.messageHistory];

    if (filter) {
      if (filter.from) {
        history = history.filter(m => m.from === filter.from);
      }
      if (filter.to) {
        history = history.filter(m => m.to === filter.to || (Array.isArray(m.to) && m.to.includes(filter.to!)));
      }
      if (filter.type) {
        history = history.filter(m => m.type === filter.type);
      }
      if (filter.since) {
        history = history.filter(m => m.timestamp >= filter.since!);
      }
    }

    return history;
  }

  /**
   * Get all active subscriptions
   */
  getActiveSubscriptions(): Map<string, Subscription[]> {
    return new Map(this.subscriptions);
  }

  /**
   * Clear all subscriptions
   */
  clearAll(): void {
    this.subscriptions.clear();
    this.messageQueue = [];
    console.log('[EventBus] All subscriptions cleared');
  }

  /**
   * Get queue size
   */
  getQueueSize(): number {
    return this.messageQueue.length;
  }

  /**
   * Get statistics
   */
  getStatistics(): {
    activeSubscriptions: number;
    totalMessages: number;
    queueSize: number;
    topicCount: number;
  } {
    let totalSubscriptions = 0;
    for (const subs of this.subscriptions.values()) {
      totalSubscriptions += subs.length;
    }

    return {
      activeSubscriptions: totalSubscriptions,
      totalMessages: this.messageHistory.length,
      queueSize: this.messageQueue.length,
      topicCount: this.subscriptions.size,
    };
  }
}

// Singleton instance
export const eventBus = new EventBus();
