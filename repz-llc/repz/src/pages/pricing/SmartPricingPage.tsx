import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from "@/ui/atoms/Button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/ui/molecules/Tabs";
import { Card } from "@/ui/molecules/Card";
import { Badge } from "@/ui/atoms/Badge";
import { Check, X, ArrowRight, Dumbbell, Home, Building2 } from 'lucide-react';
import { UnifiedTierCard } from '@/components/ui/unified-tier-card';
import { InPersonTrainingCard } from '@/components/pricing/InPersonTrainingCard';
import { motion } from 'framer-motion';

/**
 * SmartPricingPage - Intelligent pricing display that adapts to user intent
 * Shows either monthly subscriptions or in-person training based on context
 */
export default function SmartPricingPage() {
  const navigate = useNavigate();
  const [selectedTab, setSelectedTab] = useState<'monthly' | 'in-person'>('monthly');

  return (
    <div className="min-h-screen bg-black">
      {/* Hero Section */}
      <section className="relative py-16 px-4 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-repz-primary/10 to-transparent" />
        
        <div className="max-w-7xl mx-auto text-center relative z-10">
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl md:text-6xl font-bold text-white mb-6"
          >
            Choose Your Training Path
          </motion.h1>
          
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-xl text-gray-300 mb-12 max-w-3xl mx-auto"
          >
            Premium coaching designed for your lifestyle. Monthly programs or in-person sessions.
          </motion.p>
        </div>
      </section>

      {/* Pricing Tabs */}
      <section className="py-12 px-4">
        <div className="max-w-7xl mx-auto">
          <Tabs value={selectedTab} onValueChange={(v) => setSelectedTab(v as 'monthly' | 'in-person')}>
            <TabsList className="grid w-full max-w-lg mx-auto grid-cols-2 mb-12">
              <TabsTrigger value="monthly" className="text-lg">
                <Dumbbell className="w-4 h-4 mr-2" />
                Monthly Programs
              </TabsTrigger>
              <TabsTrigger value="in-person" className="text-lg">
                <Home className="w-4 h-4 mr-2" />
                In-Person Training
              </TabsTrigger>
            </TabsList>

            {/* Monthly Subscriptions Tab */}
            <TabsContent value="monthly" className="space-y-8">
              <div className="text-center mb-12">
                <h2 className="text-3xl font-bold text-red-500 mb-4">
                  Monthly Coaching Programs
                </h2>
                <p className="text-gray-400 max-w-2xl mx-auto">
                  Comprehensive training and nutrition programs delivered through our premium app.
                  Cancel anytime.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <UnifiedTierCard 
                  tier="core" 
                  comingSoon={true}
                  comingSoonMessage="New, modern dashboards launching soon for all subscription plans."
                />
                <UnifiedTierCard 
                  tier="adaptive" 
                  comingSoon={true}
                  comingSoonMessage="New, modern dashboards launching soon for all subscription plans."
                />
                <UnifiedTierCard 
                  tier="performance" 
                  comingSoon={true}
                  comingSoonMessage="New, modern dashboards launching soon for all subscription plans."
                />
                <UnifiedTierCard 
                  tier="longevity" 
                  comingSoon={true}
                  comingSoonMessage="New, modern dashboards launching soon for all subscription plans."
                />
              </div>

              {/* Feature Comparison */}
              <Card className="mt-12 p-8 bg-gray-900/50 border-gray-800">
                <h3 className="text-2xl font-bold text-white mb-6 text-center">
                  Compare Features
                </h3>
                <div className="overflow-x-auto -mx-4 md:mx-0">
                  <table className="w-full min-w-[600px]">
                    <thead>
                      <tr className="border-b border-gray-800">
                        <th className="text-left py-4 px-2 md:px-4 text-gray-400 text-sm md:text-base">Feature</th>
                        <th className="text-center py-4 px-2 md:px-4 text-blue-400 text-sm md:text-base">Core</th>
                        <th className="text-center py-4 px-2 md:px-4 text-repz-primary text-sm md:text-base">Adaptive</th>
                        <th className="text-center py-4 px-2 md:px-4 text-purple-400 text-sm md:text-base">Performance</th>
                        <th className="text-center py-4 px-2 md:px-4 text-yellow-400 text-sm md:text-base">Longevity</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-gray-800/50">
                        <td className="py-3 md:py-4 px-2 md:px-4 text-gray-300 text-sm md:text-base">Personalized Training</td>
                        <td className="text-center py-3 md:py-4 px-2 md:px-4"><Check className="w-4 h-4 md:w-5 md:h-5 text-green-500 mx-auto" /></td>
                        <td className="text-center py-3 md:py-4 px-2 md:px-4"><Check className="w-4 h-4 md:w-5 md:h-5 text-green-500 mx-auto" /></td>
                        <td className="text-center py-3 md:py-4 px-2 md:px-4"><Check className="w-4 h-4 md:w-5 md:h-5 text-green-500 mx-auto" /></td>
                        <td className="text-center py-3 md:py-4 px-2 md:px-4"><Check className="w-4 h-4 md:w-5 md:h-5 text-green-500 mx-auto" /></td>
                      </tr>
                      <tr className="border-b border-gray-800/50">
                        <td className="py-3 md:py-4 px-2 md:px-4 text-gray-300 text-sm md:text-base">Weekly Check-ins</td>
                        <td className="text-center py-3 md:py-4 px-2 md:px-4"><X className="w-4 h-4 md:w-5 md:h-5 text-gray-600 mx-auto" /></td>
                        <td className="text-center py-3 md:py-4 px-2 md:px-4"><Check className="w-4 h-4 md:w-5 md:h-5 text-green-500 mx-auto" /></td>
                        <td className="text-center py-3 md:py-4 px-2 md:px-4"><Check className="w-4 h-4 md:w-5 md:h-5 text-green-500 mx-auto" /></td>
                        <td className="text-center py-3 md:py-4 px-2 md:px-4"><Check className="w-4 h-4 md:w-5 md:h-5 text-green-500 mx-auto" /></td>
                      </tr>
                      <tr className="border-b border-gray-800/50">
                        <td className="py-3 md:py-4 px-2 md:px-4 text-gray-300 text-sm md:text-base">AI Fitness Assistant</td>
                        <td className="text-center py-3 md:py-4 px-2 md:px-4"><X className="w-4 h-4 md:w-5 md:h-5 text-gray-600 mx-auto" /></td>
                        <td className="text-center py-3 md:py-4 px-2 md:px-4"><X className="w-4 h-4 md:w-5 md:h-5 text-gray-600 mx-auto" /></td>
                        <td className="text-center py-3 md:py-4 px-2 md:px-4"><Check className="w-4 h-4 md:w-5 md:h-5 text-green-500 mx-auto" /></td>
                        <td className="text-center py-3 md:py-4 px-2 md:px-4"><Check className="w-4 h-4 md:w-5 md:h-5 text-green-500 mx-auto" /></td>
                      </tr>
                      <tr className="border-b border-gray-800/50">
                        <td className="py-3 md:py-4 px-2 md:px-4 text-gray-300 text-sm md:text-base">In-Person Training</td>
                        <td className="text-center py-3 md:py-4 px-2 md:px-4"><X className="w-4 h-4 md:w-5 md:h-5 text-gray-600 mx-auto" /></td>
                        <td className="text-center py-3 md:py-4 px-2 md:px-4"><X className="w-4 h-4 md:w-5 md:h-5 text-gray-600 mx-auto" /></td>
                        <td className="text-center py-3 md:py-4 px-2 md:px-4"><X className="w-4 h-4 md:w-5 md:h-5 text-gray-600 mx-auto" /></td>
                        <td className="text-center py-3 md:py-4 px-2 md:px-4"><Check className="w-4 h-4 md:w-5 md:h-5 text-green-500 mx-auto" /></td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </Card>
            </TabsContent>

            {/* In-Person Training Tab */}
            <TabsContent value="in-person" className="space-y-8">
              <div className="text-center mb-12">
                <h2 className="text-3xl font-bold text-white mb-4">
                  In-Person Training Sessions
                </h2>
                <p className="text-gray-400 max-w-2xl mx-auto">
                  Work directly with our certified coaches. Perfect for hands-on guidance and form correction.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-5xl mx-auto">
                <InPersonTrainingCard 
                  location={{
                    id: 'your-gym',
                    name: 'Your Gym',
                    address: 'Train at your preferred gym location',
                    amenities: ['Equipment Available', 'Flexible Scheduling'],
                    rating: 4.7,
                    packages: [{ pricePerSession: 65 }]
                  }}
                  packages={[{ pricePerSession: 65 }]}
                  onBookSession={() => {}}
                />
                <InPersonTrainingCard 
                  location={{
                    id: 'city-sports',
                    name: 'City Sports Club',
                    address: 'Premium facility with top equipment',
                    isPreferred: true,
                    amenities: ['Premium Equipment', 'Sauna', 'Pool'],
                    rating: 4.9,
                    packages: [{ pricePerSession: 79 }]
                  }}
                  packages={[{ pricePerSession: 79 }]}
                  onBookSession={() => {}}
                />
                <InPersonTrainingCard 
                  location={{
                    id: 'home',
                    name: 'Home Training',
                    address: 'Convenient training at your home',
                    amenities: ['Mobile Equipment', 'Privacy', 'No Commute'],
                    rating: 4.8,
                    packages: [{ pricePerSession: 95 }]
                  }}
                  packages={[{ pricePerSession: 95 }]}
                  onBookSession={() => {}}
                />
              </div>

              {/* Session Benefits */}
              <Card className="mt-12 p-8 bg-gray-900/50 border-gray-800 max-w-4xl mx-auto">
                <h3 className="text-2xl font-bold text-white mb-6 text-center">
                  What's Included in Every Session
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="flex items-start">
                    <Check className="w-5 h-5 text-green-500 mt-1 mr-3 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-white">60-Minute Sessions</h4>
                      <p className="text-gray-400 text-sm">Full hour of dedicated coaching</p>
                    </div>
                  </div>
                  <div className="flex items-start">
                    <Check className="w-5 h-5 text-green-500 mt-1 mr-3 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-white">Form Analysis</h4>
                      <p className="text-gray-400 text-sm">Real-time technique correction</p>
                    </div>
                  </div>
                  <div className="flex items-start">
                    <Check className="w-5 h-5 text-green-500 mt-1 mr-3 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-white">Custom Programming</h4>
                      <p className="text-gray-400 text-sm">Workouts tailored to your goals</p>
                    </div>
                  </div>
                  <div className="flex items-start">
                    <Check className="w-5 h-5 text-green-500 mt-1 mr-3 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-white">Progress Tracking</h4>
                      <p className="text-gray-400 text-sm">Detailed session notes and metrics</p>
                    </div>
                  </div>
                </div>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 px-4 border-t border-gray-800">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold text-white mb-4">
            Ready to Transform?
          </h2>
          <p className="text-gray-400 mb-8">
            Join thousands of members achieving their fitness goals with REPZ.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button 
              size="lg" 
              className="bg-repz-primary hover:bg-repz-primary/90"
              onClick={() => navigate('/auth')}
            >
              Get Started Now
              <ArrowRight className="ml-2 w-5 h-5" />
            </Button>
            <Button 
              size="lg" 
              variant="outline"
              onClick={() => navigate('/auth')}
            >
              Schedule Consultation
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
}