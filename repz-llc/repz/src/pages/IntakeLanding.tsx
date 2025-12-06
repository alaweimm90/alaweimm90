import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { CheckCircle, Clock, Mail, Shield } from 'lucide-react';

export default function IntakeLanding() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-surface-base">
      {/* Hero Section */}
      <div className="bg-gradient-to-b from-repz-orange/20 to-surface-base py-20 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-5xl font-bold text-white mb-6">
            Get Your Personalized Training Plan
          </h1>
          <p className="text-xl text-white/80 mb-8">
            No portal login required. Complete our intake form, choose your plan, and receive a custom training program via email within 48 hours.
          </p>
          <Button
            onClick={() => navigate('/intake-email')}
            size="lg"
            className="bg-repz-orange hover:bg-repz-orange-dark text-lg px-8 py-6"
          >
            Start Your Intake Form
          </Button>
        </div>
      </div>

      {/* How It Works */}
      <div className="max-w-6xl mx-auto py-16 px-4">
        <h2 className="text-3xl font-bold text-white text-center mb-12">How It Works</h2>
        <div className="grid md:grid-cols-4 gap-8">
          <Card className="p-6 text-center">
            <div className="bg-repz-orange/20 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl font-bold text-repz-orange">1</span>
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">Complete Intake</h3>
            <p className="text-white/70 text-sm">
              Fill out our comprehensive 7-step intake form with your goals, experience, and health information
            </p>
          </Card>

          <Card className="p-6 text-center">
            <div className="bg-repz-orange/20 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl font-bold text-repz-orange">2</span>
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">Choose Your Plan</h3>
            <p className="text-white/70 text-sm">
              Select from Basic, Premium, or Concierge options based on your needs and budget
            </p>
          </Card>

          <Card className="p-6 text-center">
            <div className="bg-repz-orange/20 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl font-bold text-repz-orange">3</span>
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">Secure Payment</h3>
            <p className="text-white/70 text-sm">
              Complete your one-time payment securely through Stripe checkout
            </p>
          </Card>

          <Card className="p-6 text-center">
            <div className="bg-repz-orange/20 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl font-bold text-repz-orange">4</span>
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">Receive Your Plan</h3>
            <p className="text-white/70 text-sm">
              Get your personalized training program delivered to your email within 48 hours
            </p>
          </Card>
        </div>
      </div>

      {/* Pricing */}
      <div className="bg-surface-elevated py-16 px-4">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold text-white text-center mb-12">Choose Your Plan</h2>
          <div className="grid md:grid-cols-3 gap-8">
            {/* Basic */}
            <Card className="p-8">
              <h3 className="text-2xl font-bold text-white mb-2">Basic</h3>
              <div className="text-4xl font-bold text-repz-orange mb-6">$299</div>
              <p className="text-white/70 mb-6">8-week personalized program</p>
              <ul className="space-y-3 mb-8">
                <li className="flex items-start gap-2 text-white/80">
                  <CheckCircle className="h-5 w-5 text-success mt-0.5 flex-shrink-0" />
                  <span>Custom training program</span>
                </li>
                <li className="flex items-start gap-2 text-white/80">
                  <CheckCircle className="h-5 w-5 text-success mt-0.5 flex-shrink-0" />
                  <span>Basic nutrition guidelines</span>
                </li>
                <li className="flex items-start gap-2 text-white/80">
                  <CheckCircle className="h-5 w-5 text-success mt-0.5 flex-shrink-0" />
                  <span>PDF delivery via email</span>
                </li>
                <li className="flex items-start gap-2 text-white/80">
                  <CheckCircle className="h-5 w-5 text-success mt-0.5 flex-shrink-0" />
                  <span>One revision included</span>
                </li>
              </ul>
              <Button
                onClick={() => navigate('/intake-email')}
                className="w-full bg-repz-orange hover:bg-repz-orange-dark"
              >
                Get Started
              </Button>
            </Card>

            {/* Premium */}
            <Card className="p-8 border-2 border-repz-orange relative">
              <div className="absolute -top-4 left-1/2 transform -translate-x-1/2 bg-repz-orange text-white px-4 py-1 rounded-full text-sm font-semibold">
                Most Popular
              </div>
              <h3 className="text-2xl font-bold text-white mb-2">Premium</h3>
              <div className="text-4xl font-bold text-repz-orange mb-6">$599</div>
              <p className="text-white/70 mb-6">12-week comprehensive program</p>
              <ul className="space-y-3 mb-8">
                <li className="flex items-start gap-2 text-white/80">
                  <CheckCircle className="h-5 w-5 text-success mt-0.5 flex-shrink-0" />
                  <span>Advanced training program</span>
                </li>
                <li className="flex items-start gap-2 text-white/80">
                  <CheckCircle className="h-5 w-5 text-success mt-0.5 flex-shrink-0" />
                  <span>Detailed meal planning</span>
                </li>
                <li className="flex items-start gap-2 text-white/80">
                  <CheckCircle className="h-5 w-5 text-success mt-0.5 flex-shrink-0" />
                  <span>Progress tracking templates</span>
                </li>
                <li className="flex items-start gap-2 text-white/80">
                  <CheckCircle className="h-5 w-5 text-success mt-0.5 flex-shrink-0" />
                  <span>Monthly email check-ins</span>
                </li>
                <li className="flex items-start gap-2 text-white/80">
                  <CheckCircle className="h-5 w-5 text-success mt-0.5 flex-shrink-0" />
                  <span>Two revisions included</span>
                </li>
              </ul>
              <Button
                onClick={() => navigate('/intake-email')}
                className="w-full bg-repz-orange hover:bg-repz-orange-dark"
              >
                Get Started
              </Button>
            </Card>

            {/* Concierge */}
            <Card className="p-8">
              <h3 className="text-2xl font-bold text-white mb-2">Concierge</h3>
              <div className="text-4xl font-bold text-repz-orange mb-6">$1,499</div>
              <p className="text-white/70 mb-6">12-week personalized concierge</p>
              <ul className="space-y-3 mb-8">
                <li className="flex items-start gap-2 text-white/80">
                  <CheckCircle className="h-5 w-5 text-success mt-0.5 flex-shrink-0" />
                  <span>Fully customized program</span>
                </li>
                <li className="flex items-start gap-2 text-white/80">
                  <CheckCircle className="h-5 w-5 text-success mt-0.5 flex-shrink-0" />
                  <span>Personalized nutrition plan</span>
                </li>
                <li className="flex items-start gap-2 text-white/80">
                  <CheckCircle className="h-5 w-5 text-success mt-0.5 flex-shrink-0" />
                  <span>Bi-weekly video check-ins</span>
                </li>
                <li className="flex items-start gap-2 text-white/80">
                  <CheckCircle className="h-5 w-5 text-success mt-0.5 flex-shrink-0" />
                  <span>WhatsApp/SMS support</span>
                </li>
                <li className="flex items-start gap-2 text-white/80">
                  <CheckCircle className="h-5 w-5 text-success mt-0.5 flex-shrink-0" />
                  <span>Unlimited revisions</span>
                </li>
                <li className="flex items-start gap-2 text-white/80">
                  <CheckCircle className="h-5 w-5 text-success mt-0.5 flex-shrink-0" />
                  <span>Priority support</span>
                </li>
              </ul>
              <Button
                onClick={() => navigate('/intake-email')}
                className="w-full bg-repz-orange hover:bg-repz-orange-dark"
              >
                Get Started
              </Button>
            </Card>
          </div>
        </div>
      </div>

      {/* Benefits */}
      <div className="max-w-6xl mx-auto py-16 px-4">
        <h2 className="text-3xl font-bold text-white text-center mb-12">Why Choose Email-Based Coaching?</h2>
        <div className="grid md:grid-cols-3 gap-8">
          <Card className="p-6">
            <Clock className="h-12 w-12 text-repz-orange mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">No Portal Hassle</h3>
            <p className="text-white/70">
              Skip the login process. Get your plan delivered directly to your inbox and start training immediately.
            </p>
          </Card>

          <Card className="p-6">
            <Mail className="h-12 w-12 text-repz-orange mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">Simple & Direct</h3>
            <p className="text-white/70">
              Everything you need in one PDF. Print it, save it to your phone, or keep it in your email for easy access.
            </p>
          </Card>

          <Card className="p-6">
            <Shield className="h-12 w-12 text-repz-orange mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">Expert Coaching</h3>
            <p className="text-white/70">
              Same quality programming as our portal users, tailored specifically to your goals and experience level.
            </p>
          </Card>
        </div>
      </div>

      {/* CTA */}
      <div className="bg-gradient-to-b from-surface-base to-repz-orange/20 py-16 px-4">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-3xl font-bold text-white mb-4">Ready to Transform Your Training?</h2>
          <p className="text-xl text-white/80 mb-8">
            Start your intake form now and receive your personalized plan within 48 hours.
          </p>
          <Button
            onClick={() => navigate('/intake-email')}
            size="lg"
            className="bg-repz-orange hover:bg-repz-orange-dark text-lg px-8 py-6"
          >
            Start Your Intake Form
          </Button>
        </div>
      </div>
    </div>
  );
}
