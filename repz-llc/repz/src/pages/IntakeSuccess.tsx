import { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { CheckCircle, Calendar, Mail, Download, ArrowRight } from 'lucide-react';

const TIER_INFO = {
  core: {
    name: 'Core Program',
    price: '$89/mo',
    color: 'text-blue-400',
    features: ['Personalized training program', 'Nutrition plan', 'Dashboard access', '72h response time'],
  },
  adaptive: {
    name: 'Adaptive Engine',
    price: '$149/mo',
    color: 'text-repz-orange',
    features: ['Weekly check-ins', 'Wearable integration', 'Biomarker tracking', '48h response time'],
  },
  performance: {
    name: 'Prime Suite',
    price: '$229/mo',
    color: 'text-purple-400',
    features: ['AI fitness assistant', 'Form analysis', 'PEDs protocols', '24h response time'],
  },
  longevity: {
    name: 'Elite Concierge',
    price: '$349/mo',
    color: 'text-yellow-400',
    features: ['In-person training 2x/week', 'Unlimited Q&A', 'Concierge service', '12h response time'],
  },
};

export default function IntakeSuccess() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [countdown, setCountdown] = useState(10);

  const clientId = searchParams.get('client_id');
  const tier = searchParams.get('tier') as keyof typeof TIER_INFO;
  const tierInfo = TIER_INFO[tier] || TIER_INFO.core;

  useEffect(() => {
    const timer = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(timer);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  return (
    <div className="min-h-screen bg-surface-base flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        {/* Success Animation */}
        <div className="text-center mb-8">
          <div className="relative inline-flex">
            <div className="absolute inset-0 bg-success/20 rounded-full animate-ping" />
            <div className="relative bg-success/20 p-4 rounded-full">
              <CheckCircle className="h-16 w-16 text-success" />
            </div>
          </div>
        </div>

        <Card className="p-8 text-center">
          <h1 className="text-3xl font-bold text-white mb-2">Welcome to REPZ!</h1>
          <p className="text-xl text-white/80 mb-6">
            Your subscription to <span className={tierInfo.color}>{tierInfo.name}</span> is now active.
          </p>

          {/* Plan Summary */}
          <div className="bg-surface-elevated rounded-lg p-6 mb-6 text-left">
            <h2 className="text-lg font-semibold text-white mb-4">Your Plan Includes:</h2>
            <ul className="space-y-3">
              {tierInfo.features.map((feature, idx) => (
                <li key={idx} className="flex items-center gap-3 text-white/80">
                  <CheckCircle className="h-5 w-5 text-success flex-shrink-0" />
                  <span>{feature}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Next Steps */}
          <div className="bg-repz-orange/10 border border-repz-orange/30 rounded-lg p-6 mb-6 text-left">
            <h2 className="text-lg font-semibold text-repz-orange mb-4">What Happens Next?</h2>
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <div className="bg-repz-orange text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0">1</div>
                <div>
                  <p className="text-white font-medium">Plan Creation (24-48 hours)</p>
                  <p className="text-white/60 text-sm">Our coaches will review your intake and create your personalized training plan.</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="bg-repz-orange text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0">2</div>
                <div>
                  <p className="text-white font-medium">Email Delivery</p>
                  <p className="text-white/60 text-sm">You'll receive your training plan PDF via email along with access credentials.</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="bg-repz-orange text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0">3</div>
                <div>
                  <p className="text-white font-medium">Start Training</p>
                  <p className="text-white/60 text-sm">Begin your transformation with full support from the REPZ team!</p>
                </div>
              </div>
            </div>
          </div>

          {/* Email Notice */}
          <div className="flex items-center justify-center gap-2 text-white/60 mb-6">
            <Mail className="h-5 w-5" />
            <span className="text-sm">A confirmation email has been sent to your inbox.</span>
          </div>

          {/* Actions */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button
              onClick={() => navigate('/')}
              variant="outline"
              className="flex items-center gap-2"
            >
              <ArrowRight className="h-4 w-4" />
              Return to Home
            </Button>
            <Button
              onClick={() => window.location.href = 'https://calendly.com/repzcoaching'}
              className="bg-repz-orange hover:bg-repz-orange-dark flex items-center gap-2"
            >
              <Calendar className="h-4 w-4" />
              Schedule Consultation
            </Button>
          </div>

          {/* Client ID Reference */}
          {clientId && (
            <p className="text-xs text-white/40 mt-6">
              Reference ID: {clientId.substring(0, 8)}...
            </p>
          )}
        </Card>

        {/* Social Proof */}
        <div className="mt-8 text-center">
          <p className="text-white/60 text-sm mb-2">Join 500+ athletes transforming with REPZ</p>
          <div className="flex justify-center gap-1">
            {[...Array(5)].map((_, i) => (
              <span key={i} className="text-yellow-400 text-xl">★</span>
            ))}
          </div>
          <p className="text-white/40 text-xs mt-1">4.9/5 average rating</p>
        </div>
      </div>
    </div>
  );
}
