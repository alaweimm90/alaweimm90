import { User } from '@supabase/supabase-js'

// Mock user data for testing
export const mockUsers = {
  coreUser: {
    id: '123e4567-e89b-12d3-a456-426614174000',
    email: 'core.user@test.com',
    email_confirmed_at: '2024-01-01T00:00:00Z',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    app_metadata: {},
    aud: 'authenticated',
    user_metadata: {
      name: 'Core User'
    }
  } as User,
  
  adaptiveUser: {
    id: '123e4567-e89b-12d3-a456-426614174001',
    email: 'adaptive.user@test.com',
    email_confirmed_at: '2024-01-01T00:00:00Z',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    app_metadata: {},
    aud: 'authenticated',
    user_metadata: {
      name: 'Adaptive User'
    }
  } as User,
  
  performanceUser: {
    id: '123e4567-e89b-12d3-a456-426614174002',
    email: 'performance.user@test.com',
    email_confirmed_at: '2024-01-01T00:00:00Z',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    app_metadata: {},
    aud: 'authenticated',
    user_metadata: {
      name: 'Performance User'
    }
  } as User,
  
  longevityUser: {
    id: '123e4567-e89b-12d3-a456-426614174003',
    email: 'longevity.user@test.com',
    email_confirmed_at: '2024-01-01T00:00:00Z',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    app_metadata: {},
    aud: 'authenticated',
    user_metadata: {
      name: 'Longevity User'
    }
  } as User,
  
  adminUser: {
    id: '123e4567-e89b-12d3-a456-426614174004',
    email: 'admin@test.com',
    email_confirmed_at: '2024-01-01T00:00:00Z',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    app_metadata: {},
    aud: 'authenticated',
    user_metadata: {
      name: 'Admin User'
    }
  } as User
}

// Mock client profiles
export const mockClientProfiles = {
  core: {
    id: '456e7890-f12c-34d5-b678-901234567000',
    auth_user_id: mockUsers.coreUser.id,
    client_name: 'Core User',
    subscription_tier: 'core' as const,
    age_years: 30,
    sex: 'male' as const,
    height_cm: 180,
    start_weight_kg: 80,
    target_weight_kg: 75,
    primary_goal: 'weight_loss' as const,
    tier_features: {
      progressPhotos: true,
      nutritionTracking: true,
      customWorkouts: true,
      coachAccess: true,
      weeklyCheckins: false,
      formAnalysis: false,
      ai_coaching: false,
      live_coaching: false,
      response_time_hours: 72
    },
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z'
  },
  
  adaptive: {
    id: '456e7890-f12c-34d5-b678-901234567001',
    auth_user_id: mockUsers.adaptiveUser.id,
    client_name: 'Adaptive User',
    subscription_tier: 'adaptive' as const,
    age_years: 28,
    sex: 'female' as const,
    height_cm: 165,
    start_weight_kg: 65,
    target_weight_kg: 60,
    primary_goal: 'muscle_gain' as const,
    tier_features: {
      progressPhotos: true,
      nutritionTracking: true,
      customWorkouts: true,
      coachAccess: true,
      weeklyCheckins: true,
      formAnalysis: true,
      ai_coaching: false,
      live_coaching: false,
      response_time_hours: 48
    },
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z'
  },
  
  performance: {
    id: '456e7890-f12c-34d5-b678-901234567002',
    auth_user_id: mockUsers.performanceUser.id,
    client_name: 'Performance User',
    subscription_tier: 'performance' as const,
    age_years: 35,
    sex: 'male' as const,
    height_cm: 175,
    start_weight_kg: 85,
    target_weight_kg: 80,
    primary_goal: 'strength' as const,
    tier_features: {
      progressPhotos: true,
      nutritionTracking: true,
      customWorkouts: true,
      coachAccess: true,
      weeklyCheckins: true,
      formAnalysis: true,
      ai_coaching: true,
      live_coaching: true,
      response_time_hours: 24
    },
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z'
  }
}

// Mock analytics data
export const mockAnalyticsData = {
  overview: {
    totalWorkouts: 45,
    avgIntensity: 7.5,
    streakDays: 12,
    weightChange: -2.5,
    bodyFatChange: -1.8,
    muscleGain: 1.2
  },
  
  progressData: [
    { date: '2024-01-01', weight: 80, bodyFat: 18, muscle: 65 },
    { date: '2024-01-08', weight: 79.2, bodyFat: 17.5, muscle: 65.3 },
    { date: '2024-01-15', weight: 78.5, bodyFat: 17, muscle: 65.6 },
    { date: '2024-01-22', weight: 77.8, bodyFat: 16.5, muscle: 65.9 },
    { date: '2024-01-29', weight: 77.5, bodyFat: 16.2, muscle: 66.2 }
  ],
  
  workoutData: [
    { date: '2024-01-01', duration: 45, intensity: 7, exercises: 6 },
    { date: '2024-01-03', duration: 50, intensity: 8, exercises: 7 },
    { date: '2024-01-05', duration: 40, intensity: 6, exercises: 5 },
    { date: '2024-01-08', duration: 55, intensity: 9, exercises: 8 }
  ]
}

// Mock tier data
export const mockTierData = {
  core: {
    name: 'Core',
    price: 97,
    description: 'Essential fitness tracking and coach access',
    features: [
      'Progress Photos',
      'Nutrition Tracking',
      'Custom Workouts',
      'Coach Access (72h response)'
    ],
    tier: 'core' as const
  },
  
  adaptive: {
    name: 'Adaptive',
    price: 199,
    description: 'Personalized coaching with weekly check-ins',
    features: [
      'Everything in Core',
      'Weekly Check-ins',
      'Form Analysis',
      'Body Composition Tracking',
      'Coach Access (48h response)'
    ],
    highlight: 'Most Popular',
    tier: 'adaptive' as const
  },
  
  performance: {
    name: 'Performance',
    price: 299,
    description: 'Advanced AI coaching and live sessions',
    features: [
      'Everything in Adaptive',
      'AI Coaching',
      'Live Sessions',
      'Advanced Analytics',
      'Coach Access (24h response)'
    ],
    tier: 'performance' as const
  },
  
  longevity: {
    name: 'Longevity',
    price: 449,
    description: 'Complete longevity optimization suite',
    features: [
      'Everything in Performance',
      'Biomarker Integration',
      'Peptide Protocols',
      'Bioregulators',
      'Coach Access (12h response)'
    ],
    tier: 'longevity' as const
  }
}

// Mock Stripe responses
export const mockStripeResponses = {
  successfulCheckout: {
    data: {
      sessionUrl: 'https://checkout.stripe.com/pay/cs_test_session123'
    },
    error: null
  },
  
  failedCheckout: {
    data: null,
    error: {
      message: 'Your card was declined.',
      type: 'card_error',
      code: 'card_declined'
    }
  },
  
  customerPortal: {
    data: {
      url: 'https://billing.stripe.com/p/session_portal123'
    },
    error: null
  },
  
  subscriptionCheck: {
    active: {
      data: {
        subscribed: true,
        subscription_tier: 'adaptive',
        subscription_end: '2024-12-31T23:59:59Z'
      },
      error: null
    },
    
    inactive: {
      data: {
        subscribed: false,
        subscription_tier: null,
        subscription_end: null
      },
      error: null
    }
  }
}

// Mock form data
export const mockFormData = {
  validSignup: {
    email: 'test@example.com',
    password: 'SecurePassword123!'
  },
  
  invalidEmail: {
    email: 'invalid-email',
    password: 'SecurePassword123!'
  },
  
  weakPassword: {
    email: 'test@example.com',
    password: '123'
  },
  
  existingUser: {
    email: 'existing@example.com',
    password: 'SecurePassword123!'
  }
}

// Mock error responses
export const mockErrors = {
  network: new Error('Network request failed'),
  authentication: { message: 'Invalid credentials' },
  validation: { message: 'Please enter a valid email address' },
  subscription: { message: 'Subscription not found' },
  payment: { message: 'Payment method required' }
}