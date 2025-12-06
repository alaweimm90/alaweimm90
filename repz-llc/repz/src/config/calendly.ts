// Calendly Integration Configuration

export const calendlyConfig = {
  baseUrl: import.meta.env.VITE_CALENDLY_BASE_URL || 'https://calendly.com/repz',
  
  // Event type URLs
  events: {
    consultation: import.meta.env.VITE_CALENDLY_CONSULTATION || 'https://calendly.com/repzmeshacoach/consultation-call',
    gymTraining: import.meta.env.VITE_CALENDLY_GYM_TRAINING || 'https://calendly.com/repz/gym-training',
    homeTraining: import.meta.env.VITE_CALENDLY_HOME_TRAINING || 'https://calendly.com/repz/home-training',
    cityTraining: import.meta.env.VITE_CALENDLY_CITY_TRAINING || 'https://calendly.com/repz/city-sports-club',
    virtualCheckIn: import.meta.env.VITE_CALENDLY_VIRTUAL_CHECKIN || 'https://calendly.com/repzmeshacoach/progress-review',
    personalTrainingMonthly: import.meta.env.CALENDLY_PERSONAL_TRAINING_MONTHLY || 'https://calendly.com/repzmeshacoach/personal-training-monthly',
    personalTrainingSemiWeekly: import.meta.env.CALENDLY_PERSONAL_TRAINING_SEMI_WEEKLY || 'https://calendly.com/repzmeshacoach/personal-training-semi-weekly',
    progressReview: import.meta.env.CALENDLY_PROGRESS_REVIEW || 'https://calendly.com/repzmeshacoach/progress-review',
    strategySession: import.meta.env.CALENDLY_STRATEGY_SESSION || 'https://calendly.com/repzmeshacoach/strategy-session',
  },
  
  // Embed options
  embedOptions: {
    hideEventTypeDetails: false,
    hideLandingPageDetails: false,
    hideGdprBanner: true,
    primaryColor: '#F15B23', // REPZ brand orange
    textColor: '#FFFFFF',
    backgroundColor: '#000000',
  },
  
  // UTM parameters for tracking
  utmParams: {
    utmSource: 'repz-platform',
    utmMedium: 'in-app',
    utmCampaign: 'booking',
  }
};

// Helper function to get booking URL with UTM params
export const getBookingUrl = (eventType: keyof typeof calendlyConfig.events, tierName?: string) => {
  const baseUrl = calendlyConfig.events[eventType];
  const utmParams = new URLSearchParams({
    ...calendlyConfig.utmParams,
    ...(tierName && { utmContent: tierName }),
  });
  
  return `${baseUrl}?${utmParams.toString()}`;
};

// Tier-based booking access
export const bookingAccessByTier = {
  core: ['consultation', 'virtualCheckIn'],
  adaptive: ['consultation', 'virtualCheckIn', 'progressReview'],
  performance: ['consultation', 'virtualCheckIn', 'progressReview', 'strategySession'],
  longevity: ['consultation', 'gymTraining', 'homeTraining', 'cityTraining', 'virtualCheckIn', 'personalTrainingMonthly', 'personalTrainingSemiWeekly', 'progressReview', 'strategySession'],
} as const;

// Check if user can book a specific event type
export const canBookEventType = (userTier: string, eventType: keyof typeof calendlyConfig.events): boolean => {
  const allowedEvents = bookingAccessByTier[userTier as keyof typeof bookingAccessByTier] || [];
  return (allowedEvents as readonly (keyof typeof calendlyConfig.events)[]).includes(eventType);
};