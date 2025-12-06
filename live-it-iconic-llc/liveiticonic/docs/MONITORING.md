# Live It Iconic - Comprehensive Monitoring & Observability Guide

## Overview

Live It Iconic includes a comprehensive monitoring and observability system that provides real-time insights into system health, performance metrics, uptime tracking, and alert management. This guide covers all monitoring components, configuration, and best practices.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Installation & Setup](#installation--setup)
3. [Core Components](#core-components)
4. [Configuration](#configuration)
5. [Monitoring Dashboard](#monitoring-dashboard)
6. [API Endpoints](#api-endpoints)
7. [Alert System](#alert-system)
8. [Performance Monitoring](#performance-monitoring)
9. [Error Tracking](#error-tracking)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

## System Architecture

The monitoring system consists of several interconnected modules:

```
┌─────────────────────────────────────────────────────────────┐
│                  Monitoring Dashboard                        │
│                    (/admin/monitoring)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        v                v                v
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Health     │  │ Performance  │  │    Alert     │
│   Check      │  │   Monitor    │  │   Manager    │
└──────────────┘  └──────────────┘  └──────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        v                v                v
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Uptime     │  │    Sentry    │  │  Error       │
│   Monitor    │  │  Integration │  │  Tracking    │
└──────────────┘  └──────────────┘  └──────────────┘
```

## Installation & Setup

### 1. Install Dependencies

The required monitoring packages have been installed:

```bash
npm install @sentry/react @sentry/tracing
```

### 2. Environment Variables

Add the following environment variables to your `.env` file:

```env
# Sentry Configuration
VITE_SENTRY_DSN="https://examplePublicKey@o0.ingest.sentry.io/0"
VITE_SENTRY_ENVIRONMENT="production"

# Alert Notifications (Optional)
VITE_SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
VITE_ALERT_EMAIL_WEBHOOK="https://your-email-service.com/alerts"
```

### 3. Initialize Sentry

In your main application file (`src/main.tsx`), initialize Sentry:

```typescript
import { initSentry } from '@/lib/sentry';

// Initialize Sentry as early as possible
initSentry();

// ... rest of your app initialization
```

### 4. Access the Dashboard

Navigate to `/admin/monitoring` to access the comprehensive monitoring dashboard.

## Core Components

### Health Check Module (`src/lib/healthCheck.ts`)

Monitors the health of all critical services.

**Key Functions:**

- `performHealthCheck()` - Performs comprehensive health check on all services
- `getHealthCheckHistory()` - Returns health check history
- `getRecentErrors(count)` - Gets recent error logs
- `isSystemCritical(health)` - Checks if system is critically degraded

**Example Usage:**

```typescript
import { performHealthCheck } from '@/lib/healthCheck';

const health = await performHealthCheck();
console.log(health.status); // 'healthy', 'degraded', or 'down'
console.log(health.services); // { api, database, cache, payment }
console.log(health.metrics); // { responseTime, errorRate, uptime, memoryUsage }
```

**Health Check Services:**

- **API** - Checks endpoint: `/api/health`
- **Database** - Checks endpoint: `/api/health/db`
- **Cache** - Checks endpoint: `/api/health/cache`
- **Payment** - Checks endpoint: `/api/health/payment`

### Performance Monitor (`src/lib/performance-monitor.ts`)

Tracks Core Web Vitals and performance metrics.

**Core Web Vitals Tracked:**

- **LCP** (Largest Contentful Paint) - Target: ≤ 2.5s
- **FID** (First Input Delay) - Target: ≤ 100ms
- **CLS** (Cumulative Layout Shift) - Target: ≤ 0.1
- **TTFB** (Time to First Byte) - Target: ≤ 600ms
- **FCP** (First Contentful Paint) - Target: ≤ 1.8s

**Key Functions:**

```typescript
import { perfMonitor } from '@/lib/performance-monitor';

// Get all metrics
const metrics = perfMonitor.getMetrics();

// Get Web Vitals
const vitals = perfMonitor.getWebVitals();

// Get performance summary
const summary = perfMonitor.getSummary();
// { vitalsScore: 85, allMetricsCount: 10, goodRatings: 7, ... }

// Record custom metric
perfMonitor.recordMetric('customMetric', 150);

// Get performance report
const report = perfMonitor.getReport();
```

### Alert Manager (`src/lib/alerts.ts`)

Manages system alerts and notifications.

**Alert Severity Levels:**

- `info` - Informational alerts
- `warning` - Warning alerts
- `error` - Error alerts
- `critical` - Critical alerts (trigger notifications)

**Key Functions:**

```typescript
import { alertManager } from '@/lib/alerts';

// Add an alert
const alert = alertManager.addAlert({
  severity: 'critical',
  title: 'Database Connection Failed',
  message: 'Unable to connect to primary database',
  source: 'database',
  metadata: { errorCode: 'ECONNREFUSED' }
});

// Get all alerts
const alerts = alertManager.getAlerts();

// Get active alerts
const activeAlerts = alertManager.getActiveAlerts();

// Subscribe to alerts
const unsubscribe = alertManager.subscribe((alerts) => {
  console.log('Alerts updated:', alerts);
});

// Resolve an alert
alertManager.resolveAlert(alertId);

// Get statistics
const stats = alertManager.getStatistics();
```

**Alert Thresholds (Configurable):**

```typescript
alertManager.setThresholds({
  errorRate: 5,        // Alert if error rate exceeds 5%
  responseTime: 5000,  // Alert if response time exceeds 5 seconds
  downtime: 5,         // Alert if downtime exceeds 5 minutes
  memoryUsage: 85      // Alert if memory usage exceeds 85%
});
```

### Uptime Monitor (`src/lib/uptime.ts`)

Tracks system uptime and downtime events.

**Key Functions:**

```typescript
import { uptimeMonitor } from '@/lib/uptime';

// Get uptime percentage
const uptime = uptimeMonitor.getUptime(); // e.g., 99.95

// Get uptime as string
const uptimeStr = uptimeMonitor.getUptimePercentage(); // "99.95%"

// Get duration
const duration = uptimeMonitor.getDuration(); // "5d 3h 22m"

// Get full report
const report = uptimeMonitor.getReport();
/*
{
  uptime: 99.95,
  downtime: 1800000,
  totalTime: 432000000,
  events: [...],
  startTime: 1699794000000,
  currentTime: 1699854000000,
  days: 5,
  hours: 3,
  minutes: 22
}
*/

// Get downtime statistics
const stats = uptimeMonitor.getDowntimeStats();
/*
{
  totalDowntimeEvents: 3,
  totalDowntimeDuration: 1800000,
  averageDowntimeDuration: 600000,
  minorIncidents: 1,
  majorIncidents: 1,
  criticalIncidents: 1,
  mttr: 600000,  // Mean Time To Recovery
  mtbf: 144000000  // Mean Time Between Failures
}
*/

// Record downtime
uptimeMonitor.recordDowntimeStart('Network disconnected');
// ... later ...
uptimeMonitor.recordDowntimeEnd('major', 'Network connection restored');
```

### Sentry Integration (`src/lib/sentry.ts`)

Comprehensive error tracking and session replay.

**Key Functions:**

```typescript
import {
  initSentry,
  captureException,
  captureMessage,
  setUser,
  addBreadcrumb,
  addNavigationBreadcrumb,
  addApiCallBreadcrumb,
  setContext,
  setTag,
  startTransaction,
} from '@/lib/sentry';

// Initialize (done once at app startup)
initSentry();

// Capture exceptions
try {
  // ... some code ...
} catch (error) {
  captureException(error as Error, { userId: '123' });
}

// Capture messages
captureMessage('Important event happened', 'warning');

// Set user context
setUser({
  id: '123',
  email: 'user@example.com',
  username: 'johndoe',
  ip_address: '192.168.1.1'
});

// Add breadcrumbs
addBreadcrumb('User logged in', 'auth');
addNavigationBreadcrumb('/shop', '/product/123');
addApiCallBreadcrumb('GET', '/api/products', 200);

// Set custom context
setContext('checkout', {
  items: 5,
  total: 299.99,
  currency: 'USD'
});

// Set tags for filtering
setTag('environment', 'production');
setTag('version', '1.0.0');

// Start performance transaction
const transaction = startTransaction('checkout-flow', 'transaction');
// ... do work ...
transaction?.finish();
```

## Configuration

### Alert Configuration

Configure alert thresholds and behavior:

```typescript
import { alertManager } from '@/lib/alerts';

// Set custom thresholds
alertManager.setThresholds({
  errorRate: 10,        // Alert at 10% error rate
  responseTime: 3000,   // Alert at 3 second response time
  downtime: 2,          // Alert at 2 minutes downtime
  memoryUsage: 90       // Alert at 90% memory usage
});
```

### Performance Thresholds

Core Web Vitals thresholds are built-in:

| Metric | Good | Needs Improvement | Poor |
|--------|------|------------------|------|
| LCP    | ≤ 2.5s | ≤ 4s | > 4s |
| FID    | ≤ 100ms | ≤ 300ms | > 300ms |
| CLS    | ≤ 0.1 | ≤ 0.25 | > 0.25 |
| TTFB   | ≤ 600ms | ≤ 1800ms | > 1800ms |
| FCP    | ≤ 1.8s | ≤ 3s | > 3s |

## Monitoring Dashboard

### Access

Navigate to `/admin/monitoring` to access the comprehensive dashboard.

### Dashboard Components

#### 1. Quick Stats

Displays four key metrics:
- **System Status** - Current system health (healthy/degraded/down)
- **Uptime** - System uptime percentage
- **Active Alerts** - Number of unresolved alerts
- **Performance Score** - Overall performance rating (0-100)

#### 2. Services Status

Grid showing health status of:
- API Service
- Database
- Cache
- Payment Service

#### 3. Core Web Vitals

Displays current values and ratings for:
- LCP, FID, CLS, FCP, TTFB

#### 4. Performance Metrics Chart

Bar chart showing average response times for various metrics.

#### 5. Alert Distribution

Pie chart showing breakdown of alerts by severity.

#### 6. Recent Alerts

List of latest 20 alerts with:
- Severity indicator
- Alert title and message
- Source and timestamp
- Resolution status

#### 7. Uptime Report

Summary showing:
- Total uptime percentage
- Total downtime duration
- Running duration
- Recent downtime events

### Dashboard Features

- **Live Auto-Refresh** - Data refreshes every 30 seconds
- **Pause/Resume** - Toggle live updates
- **Alert Filtering** - Filter by severity and status
- **Historical Data** - View recent events and trends
- **Export** - Export monitoring reports

## API Endpoints

### Health Check Endpoints

#### GET `/api/admin/monitoring/health`

Get current system health status.

**Response:**
```json
{
  "status": 200,
  "data": {
    "status": "healthy|degraded|down",
    "services": {
      "api": true,
      "database": true,
      "cache": true,
      "payment": true
    },
    "metrics": {
      "responseTime": 125,
      "errorRate": 0.5,
      "uptime": 99.95,
      "memoryUsage": 45
    },
    "timestamp": 1699800000000,
    "details": {
      "lastChecked": "2023-11-12T10:00:00Z",
      "checksPerformed": 4,
      "criticalIssues": []
    }
  }
}
```

### Performance Metrics Endpoints

#### GET `/api/admin/monitoring/metrics`

Get performance metrics and Web Vitals.

#### GET `/api/admin/monitoring/uptime`

Get uptime report and statistics.

### Alert Endpoints

#### GET `/api/admin/monitoring/alerts`

Get all alerts with optional filtering.

**Query Parameters:**
- `severity` - Filter by alert severity
- `resolved` - Filter by resolution status

**Response:**
```json
{
  "status": 200,
  "data": {
    "alerts": [
      {
        "id": "alert_123456_abc",
        "severity": "critical",
        "title": "Database Connection Failed",
        "message": "Unable to connect to primary database",
        "timestamp": "2023-11-12T10:00:00Z",
        "resolved": false,
        "source": "database",
        "metadata": {}
      }
    ],
    "statistics": {
      "totalAlerts": 45,
      "activeAlerts": 3,
      "criticalAlerts": 1,
      "bySource": { "database": 2, "api": 1 },
      "bySeverity": { "critical": 1, "error": 2 }
    },
    "timestamp": 1699800000000
  }
}
```

#### POST `/api/admin/monitoring/alerts`

Create a new alert.

**Request Body:**
```json
{
  "severity": "warning",
  "title": "High Memory Usage",
  "message": "Memory usage has exceeded 80%",
  "source": "system",
  "metadata": { "percentage": 82 }
}
```

#### PUT `/api/admin/monitoring/alerts/:alertId`

Resolve an alert.

### Comprehensive Report Endpoints

#### GET `/api/admin/monitoring/report`

Get comprehensive monitoring report.

**Response includes:**
- System health and uptime
- Performance metrics and Web Vitals
- Alert statistics
- Error logs

#### GET `/api/admin/monitoring/export`

Export monitoring data.

**Query Parameters:**
- `format` - `json` or `csv` (default: `json`)

## Alert System

### Alert Types

Alerts can be triggered for various events:

```typescript
// System alerts
alertManager.addAlert({
  severity: 'critical',
  title: 'API Service Down',
  message: 'API service is not responding',
  source: 'api'
});

// Performance alerts
alertManager.addAlert({
  severity: 'warning',
  title: 'Slow Response Time',
  message: 'Average response time exceeded threshold',
  source: 'performance'
});

// Database alerts
alertManager.addAlert({
  severity: 'error',
  title: 'Database Connection Error',
  message: 'Connection pool exhausted',
  source: 'database'
});

// Payment alerts
alertManager.addAlert({
  severity: 'critical',
  title: 'Payment Service Unavailable',
  message: 'Stripe API is temporarily unavailable',
  source: 'payment'
});
```

### Alert Subscriptions

Subscribe to alert changes:

```typescript
// Subscribe to all alerts
const unsubscribe = alertManager.subscribe((alerts) => {
  console.log('Alerts updated:', alerts);
});

// Subscribe to critical alerts only
const unsubscribeCritical = alertManager.subscribeToCriticalAlerts((alert) => {
  console.log('Critical alert:', alert);
  // Send notification, page on-call, etc.
});

// Cleanup when done
unsubscribe();
unsubscribeCritical();
```

### Critical Alert Notifications

Critical alerts automatically trigger:

1. Console logging
2. Slack notifications (if webhook configured)
3. Email notifications (if webhook configured)

## Performance Monitoring

### Web Vitals Monitoring

The performance monitor automatically tracks Core Web Vitals using PerformanceObserver:

```typescript
// Get current Web Vitals
const vitals = perfMonitor.getWebVitals();

console.log(vitals.LCP); // { name: 'LCP', value: 2300, rating: 'good', timestamp: ... }
console.log(vitals.FID); // { name: 'FID', value: 85, rating: 'good', timestamp: ... }
console.log(vitals.CLS); // { name: 'CLS', value: 0.08, rating: 'good', timestamp: ... }
```

### Custom Metrics

Record custom performance metrics:

```typescript
// Record a custom metric
perfMonitor.recordMetric('apiResponseTime', 245);
perfMonitor.recordMetric('pageRenderTime', 1200);
perfMonitor.recordMetric('cacheHitRate', 85);

// Get average for specific metric
const avgApiTime = perfMonitor.getAverageMetric('apiResponseTime');

// Get metric history
const history = perfMonitor.getMetricHistory('apiResponseTime');

// Get performance summary
const summary = perfMonitor.getSummary();
// { vitalsScore: 85, allMetricsCount: 10, goodRatings: 8, ... }
```

## Error Tracking

### Existing Error Tracking

The application already has comprehensive error tracking via `src/lib/errorTracking.ts`. The monitoring system integrates with this:

**Error Tracking Features:**

- Automatic global error handler
- Unhandled rejection tracking
- Network error monitoring
- API error tracking
- Breadcrumb trail recording
- Session tracking
- Severity determination

### Integration with Sentry

Use Sentry for enhanced error tracking:

```typescript
import { captureException, addBreadcrumb } from '@/lib/sentry';

try {
  // Some operation
  await fetchUserData();
} catch (error) {
  // Add context
  addBreadcrumb('fetchUserData failed', 'api');

  // Capture the error
  captureException(error as Error, {
    userId: currentUser.id,
    action: 'fetchUserData'
  });
}
```

## Best Practices

### 1. Regular Health Checks

Perform health checks regularly:

```typescript
// Every 30 seconds in the monitoring dashboard (automatic)
// Or manually check:
const health = await performHealthCheck();
if (health.status === 'down') {
  // Handle critical system failure
}
```

### 2. Alert Management

- Set appropriate thresholds based on SLA
- Subscribe to critical alerts for immediate action
- Regularly review and resolve alerts
- Document alert resolution for RCA

### 3. Performance Optimization

- Monitor Core Web Vitals regularly
- Address 'needs-improvement' metrics
- Use performance reports for optimization decisions
- Track custom metrics for application-specific performance

### 4. Uptime Monitoring

- Review uptime reports weekly
- Analyze downtime events
- Identify patterns and root causes
- Plan preventive measures

### 5. Error Tracking

- Review error logs regularly
- Prioritize critical errors
- Track error patterns
- Use breadcrumbs for context
- Set appropriate user context for debugging

### 6. Documentation

- Document all custom monitoring implementations
- Keep alert thresholds documented
- Record SLA commitments
- Maintain runbooks for critical alerts

## Troubleshooting

### Health Check Endpoints Not Responding

**Issue:** Health check returning 'degraded' or 'down'

**Solutions:**
1. Verify endpoints exist:
   - `/api/health`
   - `/api/health/db`
   - `/api/health/cache`
   - `/api/health/payment`

2. Check endpoint responses
3. Verify service connectivity
4. Check network/firewall settings

### Alerts Not Triggering

**Issue:** Expected alerts not appearing

**Solutions:**
1. Verify alert thresholds are set correctly
2. Check if duplicate alert prevention is blocking
3. Verify alert subscription is active
4. Check browser console for errors

### Performance Metrics Not Updating

**Issue:** Web Vitals showing no data

**Solutions:**
1. Ensure PerformanceObserver is supported
2. Check browser console for errors
3. Verify metrics have values (some may be null on first load)
4. Use different browser to test

### Sentry Not Capturing Errors

**Issue:** Errors not appearing in Sentry dashboard

**Solutions:**
1. Verify `VITE_SENTRY_DSN` is configured correctly
2. Check Sentry console for blocked events
3. Verify `beforeSend` filter isn't blocking errors
4. Check network requests to Sentry endpoint

### Dashboard Not Loading

**Issue:** Monitoring dashboard shows loading spinner indefinitely

**Solutions:**
1. Check browser console for errors
2. Verify `/api/admin/monitoring/health` is responding
3. Clear browser cache and reload
4. Verify user has admin permissions
5. Check network connectivity

## Performance Indicators

### Healthy System Indicators

- System Status: **Healthy**
- Uptime: **> 99.9%**
- Average Response Time: **< 500ms**
- Error Rate: **< 1%**
- LCP: **< 2.5s** (good)
- FID: **< 100ms** (good)
- CLS: **< 0.1** (good)
- Active Alerts: **0**

### Degraded System Indicators

- System Status: **Degraded**
- One or more services down
- Response time: **500ms - 2s**
- Error rate: **1-5%**
- Some Core Web Vitals in 'needs-improvement'
- Active alerts: **1-5**

### Critical System Indicators

- System Status: **Down**
- Multiple services unavailable
- Response time: **> 2s**
- Error rate: **> 5%**
- Critical Core Web Vitals
- Active critical alerts: **> 0**

## Support & Resources

For issues or questions about monitoring:

1. Check the troubleshooting section above
2. Review API documentation
3. Check browser console for errors
4. Review Sentry error tracking dashboard
5. Contact system administrator

## Version History

- **1.0.0** - Initial monitoring system implementation
  - Sentry integration
  - Health check module
  - Performance monitoring
  - Alert management
  - Uptime tracking
  - Comprehensive dashboard
