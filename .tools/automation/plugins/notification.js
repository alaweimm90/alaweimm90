/* eslint no-console: off */
/**
 * Notification Plugin
 * Sends notifications for important automation events
 */

const chalk = require('chalk');

class NotificationPlugin {
  constructor(framework) {
    this.framework = framework;
    this.logger = framework.logger;
    this.name = 'notification';

    this.config = {
      enableConsole: true,
      enableFile: true,
      logLevel: 'info',
      highlights: ['error', 'success', 'warning'],
    };

    this.notificationLog = [];
  }

  async init() {
    this.logger.info('Notification Plugin initialized');

    // Subscribe to framework events
    this.framework.on('task:start', this.onTaskStart.bind(this));
    this.framework.on('task:success', this.onTaskSuccess.bind(this));
    this.framework.on('task:failure', this.onTaskFailure.bind(this));
    this.framework.on('health:check', this.onHealthCheck.bind(this));
  }

  onTaskStart({ taskName, params }) {
    this.notify('info', `📋 Task Started: ${taskName}`, { params });
  }

  onTaskSuccess({ taskName, executionTime }) {
    this.notify('success', `✅ Task Completed: ${taskName} (${executionTime}ms)`, {
      taskName,
      executionTime,
    });
  }

  onTaskFailure({ taskName, error }) {
    this.notify('error', `❌ Task Failed: ${taskName}`, {
      taskName,
      error: error.message,
    });
  }

  onHealthCheck(health) {
    if (health.status !== 'healthy') {
      this.notify('warning', `⚠️ Health Check: ${health.status}`, {
        issues: health.issues,
      });
    }
  }

  notify(level, message, data = {}) {
    const notification = {
      timestamp: new Date().toISOString(),
      level,
      message,
      data,
    };

    // Store notification
    this.notificationLog.push(notification);
    if (this.notificationLog.length > 100) {
      this.notificationLog.shift();
    }

    // Console output with colors
    if (this.config.enableConsole) {
      this.printToConsole(notification);
    }

    // Log to file
    if (this.config.enableFile) {
      const levelMap = { warning: 'warn', success: 'info', error: 'error', info: 'info' };
      const winstonLevel = levelMap[level] || 'info';
      this.logger[winstonLevel](message, data);
    }

    // Emit notification event
    this.framework.emit('notification', notification);
  }

  printToConsole(notification) {
    const { level, message, timestamp } = notification;
    const time = new Date(timestamp).toLocaleTimeString();

    let coloredMessage;
    switch (level) {
      case 'error':
        coloredMessage = chalk.red(`[${time}] ${message}`);
        break;
      case 'success':
        coloredMessage = chalk.green(`[${time}] ${message}`);
        break;
      case 'warning':
        coloredMessage = chalk.yellow(`[${time}] ${message}`);
        break;
      case 'info':
        coloredMessage = chalk.blue(`[${time}] ${message}`);
        break;
      default:
        coloredMessage = chalk.white(`[${time}] ${message}`);
    }

    console.log(coloredMessage);
  }

  getRecentNotifications(count = 10) {
    return this.notificationLog.slice(-count);
  }

  clearNotifications() {
    this.notificationLog = [];
  }
}

module.exports = NotificationPlugin;
