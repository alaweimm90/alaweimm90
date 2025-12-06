#!/usr/bin/env node

import fs from 'fs';
import path from 'path';

const alertConfig = JSON.parse(fs.readFileSync('performance-alert-config.json', 'utf8'));

export const checkAlert = (metric, value) => {
  const threshold = alertConfig.thresholds[metric];
  if (!threshold) return null;
  
  if (value >= threshold.critical) {
    return {
      level: 'critical',
      metric,
      value,
      threshold: threshold.critical,
      message: `CRITICAL: ${metric} is ${value} (threshold: ${threshold.critical})`
    };
  } else if (value >= threshold.warning) {
    return {
      level: 'warning',
      metric,
      value,
      threshold: threshold.warning,
      message: `WARNING: ${metric} is ${value} (threshold: ${threshold.warning})`
    };
  }
  
  return null;
};

export const sendAlert = async (alert) => {
  console.log(`🚨 Performance Alert: ${alert.message}`);
  
  // Send to configured channels
  if (alertConfig.channels.slack.enabled) {
    await sendSlackAlert(alert);
  }
  
  if (alertConfig.channels.email.enabled) {
    await sendEmailAlert(alert);
  }
};

const sendSlackAlert = async (alert) => {
  const color = alert.level === 'critical' ? 'danger' : 'warning';
  const emoji = alert.level === 'critical' ? '🔴' : '⚠️';
  
  const payload = {
    channel: alertConfig.channels.slack.channel,
    text: `${emoji} Performance Alert`,
    attachments: [{
      color,
      fields: [{
        title: 'Metric',
        value: alert.metric,
        short: true
      }, {
        title: 'Value',
        value: alert.value.toString(),
        short: true
      }, {
        title: 'Threshold',
        value: alert.threshold.toString(),
        short: true
      }]
    }]
  };
  
  // Implement Slack webhook call
  console.log('Would send to Slack:', JSON.stringify(payload, null, 2));
};

const sendEmailAlert = async (alert) => {
  // Implement email sending
  console.log('Would send email alert:', alert);
};
