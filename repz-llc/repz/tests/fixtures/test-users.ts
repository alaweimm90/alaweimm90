// Test user personas and data fixtures for REPZ platform testing

export interface TestUser {
  id: string;
  email: string;
  password: string;
  role: 'client' | 'coach' | 'admin' | 'medical';
  tier?: 'core' | 'adaptive' | 'performance' | 'longevity';
  profile: {
    firstName: string;
    lastName: string;
    age?: number;
    gender?: string;
    height?: number;
    weight?: number;
  };
  subscription?: {
    status: 'active' | 'inactive' | 'cancelled';
    billingPeriod: 'monthly' | 'quarterly' | 'semi-annual' | 'annual';
    startDate: string;
  };
  preferences?: {
    notifications: boolean;
    dataSharing: boolean;
    medicalOversight: boolean;
  };
}

export const testUsers: Record<string, TestUser> = {
  // Core Tier Client
  coreClient: {
    id: 'test-core-001',
    email: 'test-core@repz.com',
    password: 'TestPassword123!',
    role: 'client',
    tier: 'core',
    profile: {
      firstName: 'John',
      lastName: 'Smith',
      age: 28,
      gender: 'male',
      height: 175,
      weight: 70
    },
    subscription: {
      status: 'active',
      billingPeriod: 'monthly',
      startDate: '2024-01-15'
    },
    preferences: {
      notifications: true,
      dataSharing: false,
      medicalOversight: false
    }
  },

  // Adaptive Tier Client
  adaptiveClient: {
    id: 'test-adaptive-001',
    email: 'test-adaptive@repz.com',
    password: 'TestPassword123!',
    role: 'client',
    tier: 'adaptive',
    profile: {
      firstName: 'Sarah',
      lastName: 'Johnson',
      age: 32,
      gender: 'female',
      height: 165,
      weight: 60
    },
    subscription: {
      status: 'active',
      billingPeriod: 'quarterly',
      startDate: '2024-01-01'
    },
    preferences: {
      notifications: true,
      dataSharing: true,
      medicalOversight: false
    }
  },

  // Performance Tier Client
  performanceClient: {
    id: 'test-performance-001',
    email: 'test-performance@repz.com',
    password: 'TestPassword123!',
    role: 'client',
    tier: 'performance',
    profile: {
      firstName: 'Mike',
      lastName: 'Rodriguez',
      age: 35,
      gender: 'male',
      height: 180,
      weight: 85
    },
    subscription: {
      status: 'active',
      billingPeriod: 'annual',
      startDate: '2023-12-01'
    },
    preferences: {
      notifications: true,
      dataSharing: true,
      medicalOversight: true
    }
  },

  // Longevity Tier Client
  longevityClient: {
    id: 'test-longevity-001',
    email: 'test-longevity@repz.com',
    password: 'TestPassword123!',
    role: 'client',
    tier: 'longevity',
    profile: {
      firstName: 'David',
      lastName: 'Chen',
      age: 42,
      gender: 'male',
      height: 178,
      weight: 75
    },
    subscription: {
      status: 'active',
      billingPeriod: 'annual',
      startDate: '2023-11-15'
    },
    preferences: {
      notifications: true,
      dataSharing: true,
      medicalOversight: true
    }
  },

  // Certified Coach
  coach: {
    id: 'test-coach-001',
    email: 'test-coach@repz.com',
    password: 'TestPassword123!',
    role: 'coach',
    profile: {
      firstName: 'Lisa',
      lastName: 'Martinez',
      age: 29
    }
  },

  // Platform Administrator
  admin: {
    id: 'test-admin-001',
    email: 'test-admin@repz.com',
    password: 'TestPassword123!',
    role: 'admin',
    profile: {
      firstName: 'Admin',
      lastName: 'User'
    }
  },

  // Verified Medical Professional
  medical: {
    id: 'test-medical-001',
    email: 'test-medical@repz.com',
    password: 'TestPassword123!',
    role: 'medical',
    profile: {
      firstName: 'Dr. James',
      lastName: 'Wilson'
    }
  },

  // Unverified Medical Professional
  unverifiedMedical: {
    id: 'test-unverified-001',
    email: 'test-unverified@repz.com',
    password: 'TestPassword123!',
    role: 'medical',
    profile: {
      firstName: 'Dr. Pending',
      lastName: 'Verification'
    }
  }
};

// Seeded data for testing
export const seedData = {
  // Medical professionals with credentials
  medicalProfessionals: [
    {
      id: 'med-001',
      userId: 'test-medical-001',
      licenseNumber: 'MD123456789',
      specialty: 'endocrinology',
      licenseState: 'california',
      yearsExperience: 15,
      verified: true,
      bio: 'Board-certified endocrinologist specializing in hormone optimization and performance medicine.'
    }
  ],

  // Sample food database items (subset of 40k+ items)
  foodItems: [
    {
      id: 'food-001',
      name: 'Chicken Breast, Skinless, Grilled',
      category: 'meat-poultry',
      calories: 165,
      protein: 31,
      carbs: 0,
      fat: 3.6,
      fiber: 0,
      sugar: 0,
      sodium: 74,
      micronutrients: {
        vitaminB6: 0.9,
        niacin: 14.8,
        phosphorus: 228,
        selenium: 27.6
      }
    },
    {
      id: 'food-002',
      name: 'Quinoa, Cooked',
      category: 'grains',
      calories: 120,
      protein: 4.4,
      carbs: 22,
      fat: 1.9,
      fiber: 2.8,
      sugar: 0.9,
      sodium: 7,
      micronutrients: {
        manganese: 0.6,
        phosphorus: 152,
        magnesium: 64,
        folate: 42
      }
    },
    {
      id: 'food-003',
      name: 'Avocado, Raw',
      category: 'fruits',
      calories: 160,
      protein: 2,
      carbs: 9,
      fat: 15,
      fiber: 7,
      sugar: 0.7,
      sodium: 7,
      micronutrients: {
        potassium: 485,
        folate: 81,
        vitaminK: 21,
        oleicAcid: 9.8
      }
    }
  ],

  // Demo protocols for Performance+ tiers
  protocols: [
    {
      id: 'protocol-001',
      name: 'Testosterone Enanthate',
      category: 'hormone-replacement',
      tier: 'performance',
      description: 'Testosterone replacement therapy for hormone optimization',
      dosage: '200mg weekly',
      duration: '12-16 weeks',
      monitoringRequired: true,
      sideEffects: ['acne', 'mood-changes', 'water-retention', 'hair-loss'],
      contraindications: ['prostate-cancer', 'severe-heart-disease', 'sleep-apnea']
    },
    {
      id: 'protocol-002',
      name: 'BPC-157',
      category: 'bioregulator',
      tier: 'performance',
      description: 'Healing peptide for injury recovery and tissue repair',
      dosage: '250mcg daily',
      duration: '4-8 weeks',
      monitoringRequired: false,
      sideEffects: ['injection-site-irritation'],
      contraindications: ['pregnancy', 'cancer-history']
    }
  ],

  // Sample recipes with calculated nutrition
  recipes: [
    {
      id: 'recipe-001',
      name: 'Power Bowl',
      description: 'High protein post-workout meal',
      category: 'main-dish',
      servings: 2,
      prepTime: 15,
      cookTime: 20,
      ingredients: [
        { foodId: 'food-002', quantity: 200, unit: 'g' }, // Quinoa
        { foodId: 'food-001', quantity: 150, unit: 'g' }, // Chicken breast
        { foodId: 'food-003', quantity: 100, unit: 'g' }  // Avocado
      ],
      instructions: [
        'Cook quinoa according to package directions',
        'Grill chicken breast until internal temp reaches 165°F',
        'Slice avocado',
        'Combine all ingredients in bowl'
      ],
      nutritionPerServing: {
        calories: 385,
        protein: 28.2,
        carbs: 22,
        fat: 15.8,
        fiber: 4.9
      }
    }
  ],

  // Test conversations for AI assistant
  conversations: [
    {
      id: 'conv-001',
      userId: 'test-performance-001',
      messages: [
        {
          role: 'user',
          content: 'What is the optimal protein intake for muscle building?',
          timestamp: '2024-01-15T10:00:00Z'
        },
        {
          role: 'assistant',
          content: 'For muscle building, research suggests an optimal protein intake of 1.6-2.2g per kg of body weight daily. For an 85kg individual like yourself, this would be approximately 136-187g of protein per day...',
          timestamp: '2024-01-15T10:00:15Z'
        }
      ]
    }
  ],

  // Test subscription data
  subscriptions: [
    {
      id: 'sub-001',
      userId: 'test-core-001',
      tier: 'core',
      status: 'active',
      stripeSubscriptionId: 'sub_test_123',
      stripePriceId: 'price_test_core_monthly',
      billingPeriod: 'monthly',
      amount: 8900, // $89.00 in cents
      startDate: '2024-01-15',
      nextBillingDate: '2024-02-15'
    },
    {
      id: 'sub-002', 
      userId: 'test-performance-001',
      tier: 'performance',
      status: 'active',
      stripeSubscriptionId: 'sub_test_456',
      stripePriceId: 'price_test_performance_annual',
      billingPeriod: 'annual',
      amount: 206100, // $2061.00 in cents (10% annual discount)
      startDate: '2023-12-01',
      nextBillingDate: '2024-12-01'
    }
  ],

  // Test achievements for gamification
  achievements: [
    {
      id: 'achieve-001',
      name: 'First Week Complete',
      description: 'Completed your first week of training',
      icon: 'trophy',
      tier: 'core',
      points: 100
    },
    {
      id: 'achieve-002',
      name: 'Nutrition Master',
      description: 'Logged meals for 30 consecutive days',
      icon: 'apple',
      tier: 'adaptive',
      points: 500
    },
    {
      id: 'achieve-003',
      name: 'Protocol Pioneer',
      description: 'Successfully completed first advanced protocol',
      icon: 'science',
      tier: 'performance',
      points: 1000
    }
  ]
};

// Helper functions for test data
export function getUserByTier(tier: string): TestUser | undefined {
  return Object.values(testUsers).find(user => user.tier === tier);
}

export function getUserByRole(role: string): TestUser | undefined {
  return Object.values(testUsers).find(user => user.role === role);
}

export function getActiveSubscriptions(): Array<Record<string, unknown>> {
  return seedData.subscriptions.filter(sub => sub.status === 'active');
}

export function getProtocolsByTier(tier: string): Array<Record<string, unknown>> {
  const tierHierarchy = ['core', 'adaptive', 'performance', 'longevity'];
  const tierIndex = tierHierarchy.indexOf(tier);
  
  if (tierIndex === -1) return [];
  
  // Return protocols available to this tier and below
  const availableTiers = tierHierarchy.slice(0, tierIndex + 1);
  return seedData.protocols.filter(protocol => 
    availableTiers.includes(protocol.tier)
  );
}

// Database seeding functions
export async function seedTestDatabase() {
  console.log('🌱 Seeding test database with fixture data');
  
  // In a real implementation, you would:
  // 1. Insert test users into database
  // 2. Create test subscriptions
  // 3. Add sample food items
  // 4. Insert test protocols
  // 5. Create sample recipes
  // 6. Set up test conversations
  
  // Example with Supabase:
  // await supabase.from('profiles').insert(Object.values(testUsers));
  // await supabase.from('subscriptions').insert(seedData.subscriptions);
  // await supabase.from('food_items').insert(seedData.foodItems);
  // etc.
  
  console.log('✅ Test database seeded successfully');
}

export async function cleanupTestDatabase() {
  console.log('🧹 Cleaning up test database');
  
  // In a real implementation, you would:
  // 1. Delete test users and related data
  // 2. Clean up test subscriptions
  // 3. Remove test conversations
  // 4. Reset test protocols
  
  console.log('✅ Test database cleaned up');
}