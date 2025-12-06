# REPZ Platform Integrations Guide

This document provides comprehensive information about all third-party integrations available in the REPZ fitness coaching platform.

## Table of Contents

1. [Overview](#overview)
2. [Whoop Integration](#whoop-integration)
3. [Apple Health Integration](#apple-health-integration)
4. [Google Calendar Integration](#google-calendar-integration)
5. [Strava Integration](#strava-integration)
6. [Referral System](#referral-system)
7. [Environment Variables](#environment-variables)
8. [Security Considerations](#security-considerations)

---

## Overview

REPZ platform integrates with popular fitness and productivity services to provide a comprehensive coaching experience. All integrations use OAuth 2.0 for secure authentication and store credentials in the `user_integrations` table with Row Level Security (RLS) enabled.

### Common Features

- **Secure OAuth 2.0 authentication**
- **Automatic data synchronization**
- **Easy connect/disconnect**
- **Real-time status checking**
- **GDPR-compliant data handling**

---

## Whoop Integration

### Overview

Whoop integration provides advanced biometric tracking including strain scores, recovery metrics, HRV, sleep performance, and respiratory rate.

### Features

- Strain score monitoring (0-21 scale)
- Recovery status (0-100%)
- Heart rate variability (HRV) tracking
- Sleep performance analysis
- Respiratory rate monitoring

### Implementation

```typescript
import {
  connectWhoop,
  syncWhoopData,
  disconnectWhoop,
  getWhoopStatus
} from '@/integrations/whoop';

// Connect to Whoop
const connected = await connectWhoop({ userId: 'user-id' });

// Sync data
const whoopData = await syncWhoopData('user-id');

// Check status
const isConnected = await getWhoopStatus('user-id');

// Disconnect
const disconnected = await disconnectWhoop('user-id');
```

### Data Structure

```typescript
interface WhoopData {
  strain?: {
    score: number; // 0-21
    averageHeartRate: number;
    maxHeartRate: number;
    calories: number;
    timestamp: Date;
  };
  recovery?: {
    score: number; // 0-100
    hrv: number; // milliseconds
    restingHeartRate: number;
    sleepPerformance: number; // percentage
    timestamp: Date;
  };
  sleep?: {
    duration: number; // minutes
    efficiency: number; // percentage
    remSleep: number; // minutes
    deepSleep: number; // minutes
    lightSleep: number; // minutes
    respiratoryRate: number;
    timestamp: Date;
  };
}
```

### Environment Variables

```bash
VITE_WHOOP_CLIENT_ID=your_whoop_client_id
VITE_WHOOP_CLIENT_SECRET=your_whoop_client_secret
```

### OAuth Flow

1. User clicks "Connect Whoop"
2. Redirects to `https://api.whoop.com/oauth/authorize`
3. User authorizes application
4. Callback to `/integrations/whoop/callback`
5. Exchange authorization code for access token
6. Store tokens in `user_integrations` table

### API Endpoints

- **Authorization**: `https://api.whoop.com/oauth/authorize`
- **Token Exchange**: `https://api.whoop.com/oauth/token`
- **Revoke Token**: `https://api.whoop.com/oauth/revoke`
- **Recovery Data**: `https://api.whoop.com/v1/recovery`
- **Cycle Data**: `https://api.whoop.com/v1/cycle`
- **Sleep Data**: `https://api.whoop.com/v1/sleep`

---

## Apple Health Integration

### Overview

Apple Health integration syncs biometric data from iOS devices using HealthKit (iOS only, requires Capacitor).

### Features

- Heart rate data synchronization
- Sleep tracking
- Activity rings data
- Workout sessions
- Step count and distance

### Implementation

```typescript
import {
  connectAppleHealth,
  syncAppleHealthData,
  disconnectAppleHealth,
  getAppleHealthStatus
} from '@/integrations/apple-health';

// Connect (iOS only)
const connected = await connectAppleHealth({
  userId: 'user-id',
  syncFrequency: 6, // hours
  dataTypes: ['HKQuantityTypeIdentifierHeartRate'] // optional
});

// Sync data
const healthData = await syncAppleHealthData('user-id');

// Check status
const isConnected = await getAppleHealthStatus('user-id');

// Disconnect
const disconnected = await disconnectAppleHealth('user-id');
```

### Data Structure

```typescript
interface HealthData {
  heartRate?: {
    average: number;
    min: number;
    max: number;
    timestamp: Date;
  };
  sleep?: {
    duration: number; // minutes
    quality: number; // 1-10
    deepSleep: number; // minutes
    timestamp: Date;
  };
  activity?: {
    steps: number;
    distance: number; // meters
    activeCalories: number;
    timestamp: Date;
  };
  workout?: {
    type: string;
    duration: number; // minutes
    calories: number;
    heartRateAvg: number;
    timestamp: Date;
  };
}
```

### Platform Requirements

- **iOS only** (requires Capacitor)
- HealthKit permissions requested at runtime
- Background sync enabled

### HealthKit Data Types

- `HKQuantityTypeIdentifierActiveEnergyBurned`
- `HKQuantityTypeIdentifierStepCount`
- `HKQuantityTypeIdentifierHeartRate`
- `HKQuantityTypeIdentifierDistanceWalkingRunning`
- `HKCategoryTypeIdentifierSleepAnalysis`
- `HKQuantityTypeIdentifierBodyMass`
- `HKQuantityTypeIdentifierHeight`

---

## Google Calendar Integration

### Overview

Google Calendar integration enables automatic workout scheduling, availability checking, and calendar invites.

### Features

- Sync workouts to calendar
- Show availability conflicts
- Send calendar invites for group sessions
- Reminder notifications

### Implementation

```typescript
import {
  connectGoogleCalendar,
  syncWorkoutToCalendar,
  checkAvailability,
  disconnectGoogleCalendar,
  getGoogleCalendarStatus
} from '@/integrations/google-calendar';

// Connect
const connected = await connectGoogleCalendar({
  userId: 'user-id',
  calendarId: 'primary' // optional
});

// Sync workout
const eventId = await syncWorkoutToCalendar('user-id', {
  title: 'Leg Day Workout',
  description: 'Focus on squats and lunges',
  startTime: new Date('2025-01-15T10:00:00'),
  endTime: new Date('2025-01-15T11:00:00'),
  location: 'Gym',
  reminders: [
    { method: 'popup', minutes: 30 },
    { method: 'email', minutes: 60 }
  ]
});

// Check availability
const isAvailable = await checkAvailability(
  'user-id',
  new Date('2025-01-15T10:00:00'),
  new Date('2025-01-15T11:00:00')
);

// Disconnect
const disconnected = await disconnectGoogleCalendar('user-id');
```

### Environment Variables

```bash
VITE_GOOGLE_CLIENT_ID=your_google_client_id
VITE_GOOGLE_CLIENT_SECRET=your_google_client_secret
```

### OAuth Scopes

- `https://www.googleapis.com/auth/calendar`
- `https://www.googleapis.com/auth/calendar.events`

### API Endpoints

- **Authorization**: `https://accounts.google.com/o/oauth2/v2/auth`
- **Token Exchange**: `https://oauth2.googleapis.com/token`
- **Revoke Token**: `https://oauth2.googleapis.com/revoke`
- **Calendar Events**: `https://www.googleapis.com/calendar/v3/calendars/{calendarId}/events`

---

## Strava Integration

### Overview

Strava integration syncs cycling and running activities, providing training load metrics and performance analytics.

### Features

- Activity data synchronization
- Training load metrics
- Performance analytics
- Route tracking

### Implementation

```typescript
import {
  connectStrava,
  syncStravaData,
  disconnectStrava,
  getStravaStatus
} from '@/integrations/strava';

// Connect
const connected = await connectStrava({ userId: 'user-id' });

// Sync activities (last 30 days by default)
const activities = await syncStravaData('user-id');

// Sync activities after specific date
const recentActivities = await syncStravaData(
  'user-id',
  new Date('2025-01-01')
);

// Check status
const isConnected = await getStravaStatus('user-id');

// Disconnect
const disconnected = await disconnectStrava('user-id');
```

### Data Structure

```typescript
interface StravaActivity {
  id: string;
  name: string;
  type: 'Run' | 'Ride' | 'Swim' | 'Other';
  distance: number; // meters
  duration: number; // seconds
  elevationGain: number; // meters
  averageHeartRate?: number;
  maxHeartRate?: number;
  calories?: number;
  timestamp: Date;
}
```

### Environment Variables

```bash
VITE_STRAVA_CLIENT_ID=your_strava_client_id
VITE_STRAVA_CLIENT_SECRET=your_strava_client_secret
```

### OAuth Scopes

- `read` - Read public profile data
- `activity:read_all` - Read all activity data
- `profile:read_all` - Read all profile data

### API Endpoints

- **Authorization**: `https://www.strava.com/oauth/authorize`
- **Token Exchange**: `https://www.strava.com/oauth/token`
- **Deauthorize**: `https://www.strava.com/oauth/deauthorize`
- **Athlete Activities**: `https://www.strava.com/api/v3/athlete/activities`

---

## Referral System

### Overview

The referral system allows users to earn rewards by inviting friends to join REPZ. Each user gets a unique referral code upon signup.

### Features

- Automatic referral code generation
- Tier-based rewards
- Usage tracking
- Reward claiming system
- Analytics dashboard

### Implementation

```typescript
import {
  getUserReferralCode,
  applyReferralCode,
  getUserReferrals,
  getReferralStats,
  claimReferralReward
} from '@/services/referralService';

// Get user's referral code
const referralCode = await getUserReferralCode('user-id');

// Apply referral code (for new signups)
const result = await applyReferralCode('ABC123XY', 'new-user-id');

// Get referrals made by user
const referrals = await getUserReferrals('user-id');

// Get referral statistics
const stats = await getReferralStats('user-id');

// Claim a reward
const claimed = await claimReferralReward('referral-id');
```

### Database Schema

#### referral_codes

- `id` - Unique identifier
- `user_id` - User who owns the code
- `code` - 8-character unique code
- `uses_remaining` - Number of uses left (NULL = unlimited)
- `expires_at` - Expiration timestamp
- `is_active` - Active status

#### referrals

- `id` - Unique identifier
- `referrer_user_id` - User who referred
- `referred_user_id` - User who was referred
- `referral_code` - Code used
- `status` - pending, completed, cancelled
- `reward_type` - Type of reward
- `reward_amount` - Reward value
- `reward_claimed` - Whether reward was claimed

#### referral_rewards

- `tier` - Subscription tier
- `reward_type` - discount, free_month, credit
- `reward_value` - Numerical value
- `referrer_reward` - Description for referrer
- `referee_reward` - Description for referee

### Default Rewards

- **Core**: 10% off next month for both parties
- **Adaptive**: 15% off next month for both parties
- **Performance**: 20% off next month for both parties
- **Longevity**: 1 free month for both parties

---

## Environment Variables

### Required Variables

```bash
# Whoop
VITE_WHOOP_CLIENT_ID=your_whoop_client_id
VITE_WHOOP_CLIENT_SECRET=your_whoop_client_secret

# Google Calendar
VITE_GOOGLE_CLIENT_ID=your_google_client_id
VITE_GOOGLE_CLIENT_SECRET=your_google_client_secret

# Strava
VITE_STRAVA_CLIENT_ID=your_strava_client_id
VITE_STRAVA_CLIENT_SECRET=your_strava_client_secret

# Supabase (already configured)
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

### Setup Instructions

1. Copy `.env.example` to `.env`
2. Fill in all required API credentials
3. Restart development server
4. Configure OAuth callback URLs in respective platforms

---

## Security Considerations

### Data Protection

- All credentials stored in encrypted database columns
- Row Level Security (RLS) policies enforce access control
- OAuth tokens refreshed automatically
- Sensitive data never exposed to client

### GDPR Compliance

- Users can disconnect integrations at any time
- All user data can be exported
- Data deletion cascades properly
- Consent tracked for each integration

### Best Practices

1. **Never log access tokens**
2. **Always use HTTPS for API calls**
3. **Implement rate limiting**
4. **Validate webhook signatures**
5. **Monitor for suspicious activity**
6. **Regularly audit integration access**

### Token Refresh

All integrations should implement automatic token refresh:

```typescript
async function refreshToken(provider: string, userId: string) {
  const { data: integration } = await supabase
    .from('user_integrations')
    .select('refresh_token, access_token')
    .eq('user_id', userId)
    .eq('provider', provider)
    .single();

  // Implement provider-specific token refresh
  // Update database with new tokens
}
```

---

## Troubleshooting

### Common Issues

1. **OAuth Redirect Mismatch**
   - Ensure redirect URIs match exactly in provider settings
   - Check for trailing slashes

2. **Token Expiration**
   - Implement automatic token refresh
   - Check token_expires_at field

3. **Rate Limiting**
   - Implement exponential backoff
   - Cache API responses when possible

4. **Scope Errors**
   - Verify all required scopes are requested
   - User may need to re-authorize

### Debug Mode

Enable debug logging:

```typescript
// In integration files
const DEBUG = import.meta.env.VITE_DEBUG_INTEGRATIONS === 'true';

if (DEBUG) {
  console.log('[Integration] Debug info:', data);
}
```

---

## Support

For integration support:
- Email: support@repz-platform.com
- Documentation: https://docs.repz-platform.com
- GitHub Issues: https://github.com/alawein-business/repz/issues

---

## Changelog

### v1.0.0 (2025-01-18)

- Initial implementation of Whoop integration
- Initial implementation of Apple Health integration
- Initial implementation of Google Calendar integration
- Initial implementation of Strava integration
- Referral system launch
- Comprehensive documentation

---

*Last updated: January 18, 2025*
