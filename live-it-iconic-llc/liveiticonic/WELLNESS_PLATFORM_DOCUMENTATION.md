# Live It Iconic - Comprehensive Wellness Platform Documentation

## Overview

This document provides comprehensive documentation for the wellness platform features integrated into Live It Iconic. The platform now includes advanced health tracking, social features, AI-powered recommendations, and robust privacy controls.

## Table of Contents

1. [Features Overview](#features-overview)
2. [Architecture](#architecture)
3. [Core Services](#core-services)
4. [Security & Privacy](#security--privacy)
5. [API Reference](#api-reference)
6. [User Guide](#user-guide)
7. [Development Guide](#development-guide)
8. [Testing](#testing)

---

## Features Overview

### 1. Wearable Integration

Connect and sync data from popular fitness wearables:

- **Supported Devices:**
  - Fitbit
  - Apple Watch
  - Garmin
  - WHOOP
  - Oura Ring
  - Samsung Health

- **Tracked Metrics:**
  - Steps
  - Heart rate (resting & active)
  - Calories burned
  - Sleep duration and quality
  - Active minutes
  - Distance
  - VO2 Max
  - Heart Rate Variability (HRV)

### 2. Mental Health Tracking

Comprehensive mental wellness monitoring:

- **Features:**
  - Daily mood check-ins
  - Stress and anxiety tracking
  - Energy level monitoring
  - Meditation session tracking
  - Trigger identification
  - Beneficial activity tracking

- **Meditation Types:**
  - Guided meditation
  - Unguided meditation
  - Breathing exercises
  - Body scan
  - Loving-kindness meditation

### 3. Habit Tracking with Streaks

Build and maintain healthy habits:

- **Features:**
  - Custom habit creation
  - Multiple frequency options (daily, weekly, monthly)
  - Streak tracking (current and longest)
  - Completion rate analytics
  - Category-based organization
  - Visual progress indicators

- **Habit Categories:**
  - Health
  - Fitness
  - Nutrition
  - Mindfulness
  - Sleep
  - Productivity
  - Social
  - Custom

### 4. Gamification System

Motivate users through game-like elements:

- **Components:**
  - Experience points (XP) and levels
  - Achievements and badges
  - Challenges (individual, team, community)
  - Leaderboards
  - Reward system
  - Rank progression

- **Rank Tiers:**
  - Novice (< 500 points)
  - Apprentice (500-2000 points)
  - Practitioner (2000-5000 points)
  - Expert (5000-10000 points)
  - Master (10000-25000 points)
  - Grand Master (25000-50000 points)
  - Legend (50000+ points)

### 5. Social Features

Build a supportive wellness community:

- **Features:**
  - Community posts and feed
  - Comments and interactions
  - Friend connections
  - Wellness groups (public and private)
  - Post categories (achievement, tip, question, motivation, recipe, workout)
  - Like and share functionality

### 6. Nutrition Planning

Comprehensive nutrition tracking and planning:

- **Features:**
  - Custom nutrition plans
  - Meal logging (breakfast, lunch, dinner, snacks)
  - Macro tracking (protein, carbs, fats)
  - Calorie tracking
  - Water intake monitoring
  - Food database with search
  - Custom food entries
  - Dietary preference support
  - Allergy tracking

### 7. AI-Powered Recommendations

Personalized, intelligent recommendations:

- **Recommendation Types:**
  - Workout suggestions
  - Meal recommendations
  - Habit reminders
  - Meditation prompts
  - Product suggestions
  - Content recommendations

- **Recommendation Priority Levels:**
  - High: Urgent or critical suggestions
  - Medium: Important but not urgent
  - Low: Nice-to-have suggestions

### 8. Offline Capabilities

Full PWA support with offline functionality:

- **Features:**
  - Service worker implementation
  - Offline data submission with background sync
  - Cached static assets
  - API response caching
  - IndexedDB for offline storage
  - Push notifications
  - Install prompt

### 9. Security & Privacy

Robust privacy protection and data security:

- **Privacy Features:**
  - Differential privacy implementation
  - Data anonymization
  - K-anonymity checks
  - PII removal
  - Privacy settings (public, friends, private)
  - Data export (GDPR compliance)
  - Right to be forgotten
  - Data retention policies

- **Security Features:**
  - Rate limiting (multiple tiers)
  - API protection
  - Input validation
  - Secure data transmission
  - Token-based authentication

### 10. Progress Visualization

Comprehensive dashboard and analytics:

- **Visualizations:**
  - Health metrics charts
  - Habit completion calendars
  - Streak indicators
  - Nutrition breakdown
  - Mood trends
  - Progress over time
  - Achievement gallery

---

## Architecture

### Service Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Wellness Dashboard Component            │   │
│  └─────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  Service Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Wearable    │  │   Mental     │  │    Habit     │ │
│  │ Integration  │  │   Health     │  │   Tracking   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Gamification │  │    Social    │  │  Nutrition   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│  ┌──────────────┐                                      │
│  │      AI      │                                      │
│  │ Recommenda-  │                                      │
│  │    tions     │                                      │
│  └──────────────┘                                      │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                 Middleware Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │     Rate     │  │   Privacy    │  │   Caching    │ │
│  │   Limiting   │  │ Enhancement  │  │              │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Database (Supabase PostgreSQL)              │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Interaction** → Frontend Component
2. **Service Call** → Service Layer (singleton pattern)
3. **Middleware** → Rate Limiting, Privacy Checks, Caching
4. **Database** → Supabase PostgreSQL
5. **Response** → Processed through middleware
6. **Update UI** → React state management

---

## Core Services

### WearableIntegrationService

**Purpose:** Manage connections to fitness wearables and sync health data.

**Key Methods:**
- `connectDevice(userId, deviceType)` - Initiate device connection
- `disconnectDevice(userId, deviceId)` - Disconnect device
- `syncHealthData(userId)` - Sync data from all connected devices
- `getHealthMetrics(userId, days)` - Retrieve health metrics
- `getConnectedDevices(userId)` - List connected devices
- `calculateDailySummary(metrics)` - Calculate daily health summary

### MentalHealthService

**Purpose:** Track and analyze mental wellness.

**Key Methods:**
- `logMentalHealthEntry(entry)` - Log mood and mental state
- `getMentalHealthEntries(userId, days)` - Retrieve entries
- `startMeditationSession(userId, type)` - Begin meditation
- `completeMeditationSession(sessionId, duration)` - Complete meditation
- `getMentalHealthInsights(userId, days)` - Generate insights

### HabitTrackingService

**Purpose:** Manage habit creation, logging, and streak tracking.

**Key Methods:**
- `createHabit(habit)` - Create new habit
- `getUserHabits(userId, activeOnly)` - Get user's habits
- `logHabitCompletion(habitId, userId, value, notes)` - Log completion
- `getHabitStreak(habitId)` - Get streak information
- `getHabitCompletionRate(habitId, days)` - Calculate completion rate
- `archiveHabit(habitId)` - Archive inactive habit

### GamificationService

**Purpose:** Manage points, levels, achievements, and challenges.

**Key Methods:**
- `getUserProgress(userId)` - Get level and points
- `awardPoints(userId, points, reason)` - Award points to user
- `getAllAchievements()` - List all achievements
- `getUserAchievements(userId)` - Get unlocked achievements
- `joinChallenge(userId, challengeId)` - Join a challenge
- `updateChallengeProgress(userId, challengeId, progress)` - Update progress

### SocialFeaturesService

**Purpose:** Enable social interactions and community building.

**Key Methods:**
- `createPost(post)` - Create community post
- `getCommunityFeed(userId, category, limit, offset)` - Get feed
- `likePost(postId, userId)` - Like/unlike post
- `addComment(postId, userId, content)` - Add comment
- `sendFriendRequest(userId, friendId)` - Send friend request
- `createGroup(group)` - Create wellness group
- `joinGroup(groupId, userId)` - Join a group

### NutritionService

**Purpose:** Manage nutrition plans and meal logging.

**Key Methods:**
- `createNutritionPlan(plan)` - Create nutrition plan
- `getActiveNutritionPlan(userId)` - Get active plan
- `logMeal(meal)` - Log a meal
- `getDailyNutritionSummary(userId, date)` - Get daily summary
- `searchFoods(query)` - Search food database
- `getNutritionInsights(userId, days)` - Generate insights

### AIRecommendationsService

**Purpose:** Generate personalized AI-driven recommendations.

**Key Methods:**
- `generateRecommendations(userId)` - Generate recommendations
- `getActiveRecommendations(userId)` - Get active recommendations
- `acceptRecommendation(recommendationId)` - Accept recommendation
- `dismissRecommendation(recommendationId)` - Dismiss recommendation
- `getUserPreferences(userId)` - Get user preferences
- `updateUserPreferences(userId, preferences)` - Update preferences

---

## Security & Privacy

### Rate Limiting

Multiple rate limiters protect different API endpoints:

```typescript
// General API - 100 requests per 15 minutes
apiRateLimiter.check(userId)

// Authentication - 5 attempts per 15 minutes
authRateLimiter.check(userId)

// Data logging - 50 logs per minute
dataLoggingRateLimiter.check(userId)

// Social features - 30 actions per minute
socialRateLimiter.check(userId)

// Sensitive operations - 3 per hour
strictRateLimiter.check(userId)
```

### Differential Privacy

Adds calibrated noise to protect individual privacy:

```typescript
const dp = new DifferentialPrivacy(epsilon = 1.0, delta = 1e-5)

// Add Laplacian noise
const noisyValue = dp.addLaplaceNoise(value, sensitivity)

// Add Gaussian noise
const noisyValue = dp.addGaussianNoise(value, sensitivity)

// Privatize aggregate statistics
const privateStats = dp.privatizeAggregate({
  count: 100,
  sum: 5000,
  average: 50,
  min: 10,
  max: 90
})
```

### Privacy Settings

Users have granular control over their data:

```typescript
interface PrivacySettings {
  profileVisibility: 'public' | 'friends' | 'private'
  showAchievements: boolean
  showProgress: boolean
  showActivity: boolean
  allowDataSharing: boolean
  allowAnonymousAnalytics: boolean
  dataRetentionDays: number
}
```

---

## User Guide

### Getting Started

1. **Connect Your Devices**
   - Navigate to Settings > Wearables
   - Select your device type
   - Follow OAuth flow to authorize

2. **Create Your First Habit**
   - Go to Habits section
   - Click "Create Habit"
   - Set name, frequency, and category
   - Start tracking!

3. **Log Your Meals**
   - Open Nutrition section
   - Click "Log Meal"
   - Search for foods or add custom entries
   - Track your macros

4. **Join the Community**
   - Visit the Social tab
   - Create a post or join groups
   - Connect with friends
   - Share your progress

### Best Practices

1. **Daily Check-ins:** Log mood and activities daily for best insights
2. **Consistent Habits:** Start with 2-3 habits and build gradually
3. **Sync Regularly:** Keep wearables synced for accurate data
4. **Review Insights:** Check weekly insights to track progress
5. **Engage Socially:** Community support improves adherence

---

## Development Guide

### Adding a New Service

1. Create service file in `/src/services/`
2. Implement singleton pattern
3. Add types to `/src/types/wellness.ts`
4. Create tests in `/src/services/__tests__/`
5. Update documentation

### Database Schema

All wellness data is stored in Supabase PostgreSQL with the following tables:

- `health_metrics` - Wearable data
- `mental_health_entries` - Mood and meditation
- `habits` - Habit definitions
- `habit_logs` - Habit completions
- `habit_streaks` - Streak tracking
- `meal_logs` - Nutrition data
- `nutrition_plans` - User nutrition plans
- `community_posts` - Social posts
- `user_achievements` - Unlocked achievements
- `recommendations` - AI recommendations
- `privacy_settings` - User privacy preferences

### Environment Variables

Required environment variables:

```env
# Wearable API Credentials
VITE_FITBIT_CLIENT_ID=your_fitbit_client_id
VITE_APPLE_CLIENT_ID=your_apple_client_id
VITE_GARMIN_CLIENT_ID=your_garmin_client_id
VITE_WHOOP_CLIENT_ID=your_whoop_client_id
VITE_OURA_CLIENT_ID=your_oura_client_id
VITE_SAMSUNG_CLIENT_ID=your_samsung_client_id

# Supabase
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

---

## Testing

### Running Tests

```bash
# Run all tests
npm test

# Run tests with coverage
npm run test:coverage

# Run tests in watch mode
npm run test:ui

# Run E2E tests
npm run test:e2e
```

### Test Coverage Goals

- Services: 80%+ coverage
- Components: 70%+ coverage
- Utilities: 90%+ coverage

---

## Roadmap

### Upcoming Features

1. **Advanced Analytics**
   - Predictive insights
   - Trend analysis
   - Goal recommendations

2. **Team Features**
   - Team challenges
   - Group competitions
   - Shared goals

3. **Integration Expansions**
   - MyFitnessPal integration
   - Strava integration
   - Google Fit support

4. **Enhanced AI**
   - Natural language meal logging
   - Image-based food recognition
   - Personalized workout generation

---

## Support & Resources

- **Documentation:** https://docs.liveiticonic.com/wellness
- **API Reference:** https://api.liveiticonic.com/docs
- **Community Forum:** https://community.liveiticonic.com
- **Support:** support@liveiticonic.com

---

## License

Proprietary - All rights reserved.

## Version

1.0.0 - Initial Wellness Platform Release

Last Updated: 2025-11-18
