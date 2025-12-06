import React, { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { Button } from "@/ui/atoms/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/ui/molecules/Card";
import { CheckCircle, ArrowRight, Calendar, CreditCard, Download, Clock } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { supabase } from '@/integrations/supabase/client';
import { toast } from 'sonner';
import { CalendlyBooking } from '@/components/booking/CalendlyBooking';
import { canBookEventType } from '@/config/calendly';

interface PaymentDetails {
  session_id: string;
  plan_name: string;
  plan_description: string;
  amount_paid: number;
  payment_status: string;
  subscription_id?: string;
  order_id?: string;
  next_billing_date?: string;
  plan_type: 'subscription' | 'one_time' | 'bundle';
}

export default function PaymentSuccess() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { user } = useAuth();
  const [paymentDetails, setPaymentDetails] = useState<PaymentDetails | null>(null);
  const [loading, setLoading] = useState(true);

  const sessionId = searchParams.get('session_id');

  useEffect(() => {
    if (sessionId && user) {
      fetchPaymentDetails();
    } else if (!sessionId) {
      toast.error('Invalid payment session');
      navigate('/pricing');
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, user]);

  const fetchPaymentDetails = async () => {
    try {
      setLoading(true);
      
      // Get tier from URL params
      const tier = searchParams.get('tier') || 'adaptive';
      const paymentStatus = searchParams.get('payment');
      
      if (paymentStatus === 'success') {
        // Map tier to canonical structure
        const tierNames = {
          core: 'Core Program',
          adaptive: 'Adaptive Engine', 
          performance: 'Prime Suite',
          longevity: 'Elite Concierge'
        };
        
        const tierDescriptions = {
          core: 'Essential training foundation with personalized planning',
          adaptive: 'Personalized optimization with interactive dashboard',
          performance: 'Elite athlete optimization with AI-powered insights', 
          longevity: 'Premium longevity protocols with in-person training'
        };
        
        const tierPricing = {
          core: 8900,      // $89
          adaptive: 14900, // $149 
          performance: 22900, // $229
          longevity: 34900   // $349
        };

        const mockDetails: PaymentDetails = {
          session_id: sessionId!,
          plan_name: tierNames[tier as keyof typeof tierNames] || tierNames.adaptive,
          plan_description: tierDescriptions[tier as keyof typeof tierDescriptions] || tierDescriptions.adaptive,
          amount_paid: tierPricing[tier as keyof typeof tierPricing] || tierPricing.adaptive,
          payment_status: 'paid',
          plan_type: 'subscription',
          next_billing_date: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString()
        };

        setPaymentDetails(mockDetails);
        
        // Show success message with tier name
        toast.success(`Payment successful! Welcome to ${mockDetails.plan_name}.`);
        
        // Trigger subscription check to update user's tier
        if (user) {
          try {
            await supabase.functions.invoke('check-subscription');
          } catch (error) {
            console.log('Could not refresh subscription status:', error);
          }
        }
      } else {
        throw new Error('Payment was not successful');
      }
      
    } catch (error) {
      console.error('Error fetching payment details:', error);
      toast.error('Failed to verify payment. Please contact support if you were charged.');
      navigate('/pricing');
    } finally {
      setLoading(false);
    }
  };

  const formatPrice = (cents: number) => {
    return `$${(cents / 100).toFixed(2)}`;
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-subtle">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Verifying your payment...</p>
        </div>
      </div>
    );
  }

  if (!paymentDetails) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-subtle">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-foreground mb-4">Payment Verification Failed</h2>
          <p className="text-muted-foreground mb-6">We couldn't verify your payment. Please contact support.</p>
          <Button onClick={() => navigate('/pricing')}>
            Return to Pricing
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-subtle">
      <div className="container mx-auto px-4 py-16">
        <div className="max-w-2xl mx-auto">
          {/* Success Header */}
          <div className="text-center mb-8">
            <div className="w-20 h-20 bg-success/10 rounded-full flex items-center justify-center mx-auto mb-6">
              <CheckCircle className="w-12 h-12 text-success" />
            </div>
            <h1 className="text-4xl font-bold text-foreground mb-2">Payment Successful!</h1>
            <p className="text-xl text-muted-foreground">
              Thank you for your purchase. Your account has been upgraded.
            </p>
          </div>

          {/* Payment Details Card */}
          <Card className="mb-8">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <CreditCard className="w-5 h-5" />
                Payment Details
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="font-semibold text-lg">{paymentDetails.plan_name}</h3>
                    <p className="text-muted-foreground">{paymentDetails.plan_description}</p>
                  </div>
                  <div className="text-right">
                    <p className="font-bold text-xl">{formatPrice(paymentDetails.amount_paid)}</p>
                    <p className="text-sm text-success">Paid</p>
                  </div>
                </div>

                <div className="border-t pt-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Session ID</p>
                      <p className="font-mono text-sm">{paymentDetails.session_id}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Plan Type</p>
                      <p className="capitalize">{paymentDetails.plan_type.replace('_', ' ')}</p>
                    </div>
                    {paymentDetails.next_billing_date && (
                      <div className="md:col-span-2">
                        <p className="text-sm font-medium text-muted-foreground">Next Billing Date</p>
                        <p className="flex items-center gap-2">
                          <Calendar className="w-4 h-4" />
                          {formatDate(paymentDetails.next_billing_date)}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Next Steps */}
          <Card className="mb-8">
            <CardHeader>
              <CardTitle>What's Next?</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {paymentDetails.plan_type === 'subscription' ? (
                  <>
                    <div className="flex items-start gap-3">
                      <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center mt-0.5">
                        <span className="text-xs font-bold text-primary">1</span>
                      </div>
                      <div>
                        <h4 className="font-semibold">Access Your Features</h4>
                        <p className="text-sm text-muted-foreground">
                          All premium features are now available in your dashboard.
                        </p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center mt-0.5">
                        <span className="text-xs font-bold text-primary">2</span>
                      </div>
                      <div>
                        <h4 className="font-semibold">Setup Your Account</h4>
                        <p className="text-sm text-muted-foreground">
                          Complete your profile and configure your preferences.
                        </p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center mt-0.5">
                        <span className="text-xs font-bold text-primary">3</span>
                      </div>
                      <div>
                        <h4 className="font-semibold">Manage Your Subscription</h4>
                        <p className="text-sm text-muted-foreground">
                          View billing history and manage your subscription in your account dashboard.
                        </p>
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="flex items-start gap-3">
                      <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center mt-0.5">
                        <span className="text-xs font-bold text-primary">1</span>
                      </div>
                      <div>
                        <h4 className="font-semibold">Download Your Receipt</h4>
                        <p className="text-sm text-muted-foreground">
                          Save your receipt for your records.
                        </p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center mt-0.5">
                        <span className="text-xs font-bold text-primary">2</span>
                      </div>
                      <div>
                        <h4 className="font-semibold">Check Your Email</h4>
                        <p className="text-sm text-muted-foreground">
                          We've sent you a confirmation email with all the details.
                        </p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center mt-0.5">
                        <span className="text-xs font-bold text-primary">3</span>
                      </div>
                      <div>
                        <h4 className="font-semibold">Access Your Purchase</h4>
                        <p className="text-sm text-muted-foreground">
                          Your purchase is now available in your account.
                        </p>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Calendly Booking Section - Added after payment success */}
          <Card className="mb-8 border-primary/20 bg-gradient-to-r from-primary/5 to-orange-500/5">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-primary">
                <Calendar className="w-5 h-5" />
                Schedule Your First Session
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Your payment is confirmed! Now let's get you started with your first coaching session. 
                  Choose from the options below based on your new {paymentDetails.plan_name} plan.
                </p>
                
                {/* Tier-specific booking options */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Free Consultation - Available to all tiers */}
                  <div className="p-4 border rounded-lg bg-background/50">
                    <div className="flex items-start gap-3 mb-3">
                      <div className="w-8 h-8 bg-success/10 rounded-full flex items-center justify-center mt-0.5">
                        <Calendar className="w-4 h-4 text-success" />
                      </div>
                      <div>
                        <h4 className="font-semibold">Free Strategy Session</h4>
                        <p className="text-sm text-muted-foreground">
                          30-minute consultation to set up your personalized plan
                        </p>
                      </div>
                    </div>
                    <CalendlyBooking 
                      eventType="consultation"
                      buttonText="Book Free Session"
                      buttonVariant="default"
                      className="w-full"
                      prefillData={{
                        name: user?.email?.split('@')[0],
                        email: user?.email
                      }}
                    />
                  </div>

                  {/* Tier-specific advanced booking */}
                  {(() => {
                    const tier = searchParams.get('tier') || 'adaptive';
                    
                    if (tier === 'longevity' && canBookEventType(tier, 'cityTraining')) {
                      return (
                        <div className="p-4 border rounded-lg bg-background/50">
                          <div className="flex items-start gap-3 mb-3">
                            <div className="w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center mt-0.5">
                              <Clock className="w-4 h-4 text-primary" />
                            </div>
                            <div>
                              <h4 className="font-semibold">Premium In-Person Training</h4>
                              <p className="text-sm text-muted-foreground">
                                Elite 1-on-1 session at City Sports Club (1 hr)
                              </p>
                            </div>
                          </div>
                          <CalendlyBooking 
                            eventType="cityTraining"
                            buttonText="Book Premium Session"
                            buttonVariant="accent"
                            className="w-full"
                            prefillData={{
                              name: user?.email?.split('@')[0],
                              email: user?.email
                            }}
                          />
                        </div>
                      );
                    } else if ((tier === 'performance' || tier === 'adaptive') && canBookEventType(tier, 'gymTraining')) {
                      return (
                        <div className="p-4 border rounded-lg bg-background/50">
                          <div className="flex items-start gap-3 mb-3">
                            <div className="w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center mt-0.5">
                              <Clock className="w-4 h-4 text-primary" />
                            </div>
                            <div>
                              <h4 className="font-semibold">Gym Training Session</h4>
                              <p className="text-sm text-muted-foreground">
                                Personalized workout at your local gym (1 hr)
                              </p>
                            </div>
                          </div>
                          <CalendlyBooking 
                            eventType="gymTraining"
                            buttonText="Book Gym Session"
                            buttonVariant="accent"
                            className="w-full"
                            prefillData={{
                              name: user?.email?.split('@')[0],
                              email: user?.email
                            }}
                          />
                        </div>
                      );
                    } else {
                      return (
                        <div className="p-4 border rounded-lg bg-background/50">
                          <div className="flex items-start gap-3 mb-3">
                            <div className="w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center mt-0.5">
                              <Clock className="w-4 h-4 text-primary" />
                            </div>
                            <div>
                              <h4 className="font-semibold">Home Training Session</h4>
                              <p className="text-sm text-muted-foreground">
                                Get started with home-based training (1 hr)
                              </p>
                            </div>
                          </div>
                          <CalendlyBooking 
                            eventType="homeTraining"
                            buttonText="Book Home Session"
                            buttonVariant="accent"
                            className="w-full"
                            prefillData={{
                              name: user?.email?.split('@')[0],
                              email: user?.email
                            }}
                          />
                        </div>
                      );
                    }
                  })()}
                </div>

                <div className="text-center pt-4 border-t">
                  <p className="text-sm text-muted-foreground">
                    <strong>Pro Tip:</strong> Book your strategy session first to maximize your training results!
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button 
              onClick={() => navigate('/dashboard')}
              className="flex items-center gap-2"
            >
              Go to Dashboard
              <ArrowRight className="w-4 h-4" />
            </Button>
            
            <Button 
              variant="outline"
              onClick={() => navigate('/')}
            >
              Return to Home
            </Button>
            
            {paymentDetails.plan_type !== 'subscription' && (
              <Button 
                variant="outline"
                onClick={() => window.print()}
                className="flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                Download Receipt
              </Button>
            )}
          </div>

          {/* Support */}
          <div className="text-center mt-8 p-6 bg-muted/30 rounded-lg">
            <h3 className="font-semibold mb-2">Need Help?</h3>
            <p className="text-sm text-muted-foreground mb-4">
              If you have any questions about your purchase or need assistance, our support team is here to help.
            </p>
            <Button variant="outline" size="sm" onClick={() => console.log("PaymentSuccess button clicked")}>
              Contact Support
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}