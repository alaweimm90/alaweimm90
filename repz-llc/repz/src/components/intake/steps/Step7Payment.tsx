// src/components/intake/steps/Step7Payment.tsx
import React from 'react';
import { Button } from "@/ui/atoms/Button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/ui/molecules/Card";
import { Alert, AlertDescription } from "@/ui/molecules/Alert";
import { CheckCircle, CreditCard, Shield, Lock, AlertCircle, Clock } from 'lucide-react';
import { Badge } from "@/ui/atoms/Badge";
import { UnifiedTierCard } from "@/components/ui/unified-tier-card";
import { supabase } from "@/integrations/supabase/client";
import { useFormState } from "@/hooks/useFormState";
import { BillingPeriodSelector } from "@/components/ui/billing-period-selector";
import { getTierConfig } from "@/constants/tiers";

interface Step7FormData {
  email?: string;
  termsAccepted?: boolean;
  [key: string]: string | boolean | undefined;
}

interface Step7PaymentProps {
  formData: Step7FormData;
  selectedTier: string;
  tierPrice: string;
  billingPeriod: 'monthly' | 'annual';
  onBillingPeriodChange: (period: 'monthly' | 'annual') => void;
  onNext: () => void;
  onPrevious: () => void;
  isLoading: boolean;
}

const Step7PaymentComponent: React.FC<Step7PaymentProps> = ({
  formData,
  selectedTier,
  tierPrice,
  billingPeriod,
  onBillingPeriodChange,
  onNext,
  onPrevious,
  isLoading
}) => {
  const { data: formState, setField, setSubmitting, isSubmitting } = useFormState({
    isProcessing: false,
    error: '',
    isSetupComplete: false
  });

  const formatTierName = (tier: string) => {
    return tier.replace(/🧑‍💻|🟠|⚫/, '').trim();
  };

  // Get tier configuration and calculate pricing
  const tierConfig = getTierConfig(selectedTier);
  const monthlyPrice = parseInt(tierPrice);
  const annualPrice = Math.round(monthlyPrice * 0.8); // 20% discount for annual
  const displayPrice = billingPeriod === 'annual' ? annualPrice : monthlyPrice;

  const handlePaymentSetup = async () => {
    setSubmitting(true);
    setField('error', '');

    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) {
        throw new Error('Please login to continue');
      }

      // Call your create-zero-subscription function
      const { data, error } = await supabase.functions.invoke('create-zero-subscription', {
        body: {
          fullName: formData.fullName,
          email: formData.email,
          phone: formData.phone,
          dateOfBirth: formData.dateOfBirth,
          gender: formData.gender,
          location: formData.location,
          currentWeight: formData.currentWeight,
          height: formData.height,
          goals: formData.specificGoals || formData.goals || 'General fitness improvement',
          fitnessLevel: formData.fitnessLevel,
          specialRequirements: formData.specialRequirements || formData.additionalNotes,
          selectedTier: formatTierName(selectedTier),
          userId: session.user.id
        }
      });

      if (error) {
        throw new Error(error.message || 'Failed to complete registration');
      }

      setField('isSetupComplete', true);
      
      // Auto-advance after showing success message
      setTimeout(() => {
        onNext();
      }, 3000);

    } catch (error) {
      setField('error', error instanceof Error ? error.message : 'Registration failed');
    } finally {
      setSubmitting(false);
    }
  };

  if (formState.isSetupComplete) {
    return (
      <div className="step-container w-full max-w-none">
        <div className="max-w-4xl mx-auto space-y-10 pb-16 pt-4">
          <Card>
            <CardContent className="pt-6">
              <div className="text-center space-y-4">
                <CheckCircle className="h-16 w-16 text-green-500 mx-auto" />
                <h2 className="text-2xl font-bold text-green-800">Registration Complete!</h2>
                <p className="text-gray-600">
                  Your coaching application has been submitted successfully.
                </p>
                
                <div className="bg-green-50 p-6 rounded-lg border border-green-200 text-left">
                  <h3 className="font-semibold text-green-800 mb-3">What happens next:</h3>
                  <ul className="space-y-2 text-sm text-green-700">
                    <li className="flex items-start gap-2">
                      <span className="font-bold text-repz-orange">1.</span>
                      <span>Our coach will review your application within 24 hours</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="font-bold text-repz-orange">2.</span>
                      <span>A personalized plan will be created based on your goals</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="font-bold text-repz-orange">3.</span>
                      <span>You'll receive an email to review and approve your plan</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="font-bold text-repz-orange">4.</span>
                      <span>Monthly billing begins only after your approval</span>
                    </li>
                  </ul>
                </div>

                  <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                  <p className="text-sm text-blue-700">
                    <strong>Confirmation emails sent to:</strong> {formData.email}
                    <br />
                    <strong>Selected Plan:</strong> {formatTierName(selectedTier)} (${displayPrice}/month, billed {billingPeriod})
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="step-container w-full max-w-none">
      <div className="max-w-4xl mx-auto space-y-10 pb-16 pt-4">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold text-gray-900 tracking-tight">
            Complete Your Registration
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto leading-relaxed">
            Finalize your coaching application - no payment required until plan approval
          </p>
        </div>

        {/* Form Container */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm space-y-8">
          {/* Plan Summary */}
          <div className="p-8">
            <UnifiedTierCard
              tier={selectedTier}
              variant="comparison"
              showFeatures={true}
              className="border-l-4 border-l-orange-500"
            />
            <div className="mt-4 text-center">
              <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                ${displayPrice}/month • Billed {billingPeriod} after plan approval
              </Badge>
            </div>
          </div>

          {/* Billing Period Selection */}
          <div className="p-8 border-t border-gray-100">
            <BillingPeriodSelector
              selectedPeriod={billingPeriod}
              onPeriodChange={onBillingPeriodChange}
              monthlyPrice={monthlyPrice}
              annualPrice={annualPrice}
            />
          </div>

          {/* Application Summary */}
          <div className="p-8 border-t border-gray-100">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CreditCard className="h-5 w-5" />
                  Application Summary
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <strong>Name:</strong> {formData.fullName}
                  </div>
                  <div>
                    <strong>Email:</strong> {formData.email}
                  </div>
                  <div>
                    <strong>Primary Goal:</strong> {formData.specificGoals?.substring(0, 50) || formData.goals?.substring(0, 50) || 'General fitness'}...
                  </div>
                  <div>
                    <strong>Fitness Level:</strong> {formData.fitnessLevel || 'Not specified'}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* How It Works */}
          <div className="p-8 border-t border-gray-100">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock className="h-5 w-5 text-blue-500" />
                  How Our Process Works
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="flex gap-3">
                      <div className="flex-shrink-0 w-6 h-6 bg-repz-orange/10 text-repz-orange rounded-full flex items-center justify-center text-sm font-bold">1</div>
                      <div>
                        <h4 className="font-medium text-gray-900">Application Review</h4>
                        <p className="text-sm text-gray-600">Our coach reviews your information within 24 hours</p>
                      </div>
                    </div>
                    <div className="flex gap-3">
                      <div className="flex-shrink-0 w-6 h-6 bg-repz-orange/10 text-repz-orange rounded-full flex items-center justify-center text-sm font-bold">2</div>
                      <div>
                        <h4 className="font-medium text-gray-900">Plan Creation</h4>
                        <p className="text-sm text-gray-600">Personalized plan designed for your goals and preferences</p>
                      </div>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div className="flex gap-3">
                      <div className="flex-shrink-0 w-6 h-6 bg-repz-orange/10 text-repz-orange rounded-full flex items-center justify-center text-sm font-bold">3</div>
                      <div>
                        <h4 className="font-medium text-gray-900">Plan Approval</h4>
                        <p className="text-sm text-gray-600">You review and approve your custom plan before any billing</p>
                      </div>
                    </div>
                    <div className="flex gap-3">
                      <div className="flex-shrink-0 w-6 h-6 bg-repz-orange/10 text-repz-orange rounded-full flex items-center justify-center text-sm font-bold">4</div>
                      <div>
                        <h4 className="font-medium text-gray-900">Coaching Begins</h4>
                        <p className="text-sm text-gray-600">Monthly billing starts and you receive your complete program</p>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Security Notice */}
          <div className="p-8 border-t border-gray-100">
            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
              <div className="flex items-center gap-2 text-blue-800 mb-2">
                <Shield className="h-4 w-4" />
                <span className="font-medium">Secure Registration</span>
              </div>
              <p className="text-sm text-blue-700">
                Your information is encrypted and secure. No payment is required until you approve your personalized plan.
              </p>
            </div>
          </div>

          {/* Error Display */}
          {formState.error && (
            <div className="p-8 border-t border-gray-100">
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{formState.error}</AlertDescription>
              </Alert>
            </div>
          )}

          {/* Action Button */}
          <div className="p-8 border-t border-gray-100 text-center">
            <Button
              onClick={handlePaymentSetup}
              disabled={isSubmitting}
              size="lg"
              className="w-full md:w-auto px-8 py-3 bg-repz-orange hover:bg-repz-orange/90"
            >
              {isSubmitting ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                  Submitting Application...
                </>
              ) : (
                <>
                  <Lock className="h-4 w-4 mr-2" />
                  Complete Registration
                </>
              )}
            </Button>
          </div>

          {/* Navigation */}
          <div className="p-8 border-t border-gray-100">
            <div className="flex justify-between">
              <Button variant="outline" onClick={onPrevious} disabled={isSubmitting}>
                Previous
              </Button>
              <div className="text-sm text-gray-500">
                Step 7 of 7
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Memoized export for performance
export const Step7Payment = React.memo(Step7PaymentComponent);