import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/hooks/use-toast';
import { supabase } from '@/integrations/supabase/client';
import { ArrowLeft, ArrowRight, CheckCircle } from 'lucide-react';

interface IntakeFormData {
  // Step 1: Account Setup
  firstName: string;
  lastName: string;
  email: string;
  phone: string;

  // Step 2: Personal Profile
  age: string;
  sex: string;
  height: string;
  weight: string;
  activityLevel: string;
  healthConditions: string;
  medications: string;
  injuries: string;

  // Step 3: Health Baseline
  restingHeartRate: string;
  bloodPressure: string;
  vo2Max: string;
  sleepQuality: string;
  stressLevel: string;

  // Step 4: Training Experience
  trainingYears: string;
  primaryGoals: string;
  trainingFrequency: string;
  availableEquipment: string;
  trainingStyle: string;

  // Step 5: Nutrition & Lifestyle
  dietaryRestrictions: string;
  eatingSchedule: string;
  supplements: string;
  alcoholConsumption: string;
  smokingStatus: string;
  sleepSchedule: string;

  // Step 6: Goals & Expectations
  primaryGoal: string;
  secondaryGoals: string;
  timeline: string;
  weeklyCommitment: string;
  successMetrics: string;

  // Step 7: Payment Selection
  paymentType: 'core' | 'adaptive' | 'performance' | 'longevity' | '';
}

const TOTAL_STEPS = 7;

export default function IntakeEmail() {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [currentStep, setCurrentStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isComplete, setIsComplete] = useState(false);

  const [formData, setFormData] = useState<IntakeFormData>({
    firstName: '',
    lastName: '',
    email: '',
    phone: '',
    age: '',
    sex: '',
    height: '',
    weight: '',
    activityLevel: '',
    healthConditions: '',
    medications: '',
    injuries: '',
    restingHeartRate: '',
    bloodPressure: '',
    vo2Max: '',
    sleepQuality: '',
    stressLevel: '',
    trainingYears: '',
    primaryGoals: '',
    trainingFrequency: '',
    availableEquipment: '',
    trainingStyle: '',
    dietaryRestrictions: '',
    eatingSchedule: '',
    supplements: '',
    alcoholConsumption: '',
    smokingStatus: '',
    sleepSchedule: '',
    primaryGoal: '',
    secondaryGoals: '',
    timeline: '',
    weeklyCommitment: '',
    successMetrics: '',
    paymentType: '',
  });

  const updateFormData = (field: keyof IntakeFormData, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleNext = () => {
    if (currentStep < TOTAL_STEPS) {
      setCurrentStep(prev => prev + 1);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep(prev => prev - 1);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const handleSubmit = async () => {
    if (!formData.paymentType) {
      toast({
        title: 'Payment Option Required',
        description: 'Please select a payment option to continue.',
        variant: 'destructive',
      });
      return;
    }

    setIsSubmitting(true);

    try {
      // Insert into non_portal_clients table
      const { data, error } = await supabase
        .from('non_portal_clients')
        .insert({
          email: formData.email,
          full_name: `${formData.firstName} ${formData.lastName}`,
          phone: formData.phone,
          intake_data: formData,
          payment_type: formData.paymentType,
          payment_status: 'pending',
        })
        .select()
        .single();

      if (error) throw error;

      // Create Stripe checkout session
      const { data: checkoutData, error: checkoutError } = await supabase.functions.invoke(
        'create-intake-checkout',
        {
          body: {
            clientId: data.id,
            paymentType: formData.paymentType,
            email: formData.email,
            name: `${formData.firstName} ${formData.lastName}`,
          },
        }
      );

      if (checkoutError) throw checkoutError;

      // Redirect to Stripe checkout
      if (checkoutData?.url) {
        window.location.href = checkoutData.url;
      } else {
        throw new Error('No checkout URL received');
      }
    } catch (error) {
      console.error('Submission error:', error);
      toast({
        title: 'Submission Failed',
        description: 'An error occurred. Please try again or contact support.',
        variant: 'destructive',
      });
      setIsSubmitting(false);
    }
  };

  if (isComplete) {
    return (
      <div className="min-h-screen bg-surface-base flex items-center justify-center p-4">
        <Card className="max-w-2xl w-full p-8 text-center">
          <div className="flex justify-center mb-6">
            <CheckCircle className="h-16 w-16 text-success" />
          </div>
          <h1 className="text-3xl font-bold text-white mb-4">Thank You!</h1>
          <p className="text-lg text-white/80 mb-6">
            We have received your intake form. Our team will review your information and send your personalized training plan via email within 48 hours.
          </p>
          <p className="text-white/60 mb-8">
            Please check your email (including spam folder) for confirmation and your upcoming plan delivery.
          </p>
          <Button onClick={() => navigate('/')} className="bg-repz-orange hover:bg-repz-orange-dark">
            Return to Home
          </Button>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-surface-base py-8 px-4">
      <div className="max-w-3xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">REPZ Email-Based Intake</h1>
          <p className="text-white/70">Complete this form to receive your personalized training plan via email</p>
        </div>

        {/* Progress Indicator */}
        <div className="mb-8">
          <div className="flex justify-between mb-2">
            <span className="text-sm text-white/60">Step {currentStep} of {TOTAL_STEPS}</span>
            <span className="text-sm text-white/60">{Math.round((currentStep / TOTAL_STEPS) * 100)}%</span>
          </div>
          <div className="w-full bg-surface-elevated rounded-full h-2">
            <div
              className="bg-repz-orange h-2 rounded-full transition-all duration-300"
              style={{ width: `${(currentStep / TOTAL_STEPS) * 100}%` }}
            />
          </div>
        </div>

        <Card className="p-6">
          {/* Step 1: Account Setup */}
          {currentStep === 1 && (
            <div className="space-y-4">
              <h2 className="text-2xl font-bold text-white mb-4">Account Setup</h2>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="firstName">First Name</Label>
                  <Input
                    id="firstName"
                    value={formData.firstName}
                    onChange={(e) => updateFormData('firstName', e.target.value)}
                    required
                  />
                </div>
                <div>
                  <Label htmlFor="lastName">Last Name</Label>
                  <Input
                    id="lastName"
                    value={formData.lastName}
                    onChange={(e) => updateFormData('lastName', e.target.value)}
                    required
                  />
                </div>
              </div>

              <div>
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  value={formData.email}
                  onChange={(e) => updateFormData('email', e.target.value)}
                  required
                />
              </div>

              <div>
                <Label htmlFor="phone">Phone</Label>
                <Input
                  id="phone"
                  type="tel"
                  value={formData.phone}
                  onChange={(e) => updateFormData('phone', e.target.value)}
                />
              </div>
            </div>
          )}

          {/* Step 2: Personal Profile */}
          {currentStep === 2 && (
            <div className="space-y-4">
              <h2 className="text-2xl font-bold text-white mb-4">Personal Profile</h2>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="age">Age</Label>
                  <Input
                    id="age"
                    type="number"
                    value={formData.age}
                    onChange={(e) => updateFormData('age', e.target.value)}
                    required
                  />
                </div>
                <div>
                  <Label htmlFor="sex">Sex</Label>
                  <select
                    id="sex"
                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                    value={formData.sex}
                    onChange={(e) => updateFormData('sex', e.target.value)}
                  >
                    <option value="">Select</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="other">Other</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="height">Height (inches)</Label>
                  <Input
                    id="height"
                    type="number"
                    value={formData.height}
                    onChange={(e) => updateFormData('height', e.target.value)}
                  />
                </div>
                <div>
                  <Label htmlFor="weight">Weight (lbs)</Label>
                  <Input
                    id="weight"
                    type="number"
                    value={formData.weight}
                    onChange={(e) => updateFormData('weight', e.target.value)}
                  />
                </div>
              </div>

              <div>
                <Label htmlFor="activityLevel">Activity Level</Label>
                <select
                  id="activityLevel"
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  value={formData.activityLevel}
                  onChange={(e) => updateFormData('activityLevel', e.target.value)}
                >
                  <option value="">Select</option>
                  <option value="sedentary">Sedentary</option>
                  <option value="light">Light</option>
                  <option value="moderate">Moderate</option>
                  <option value="intense">Intense</option>
                  <option value="very-intense">Very Intense</option>
                </select>
              </div>

              <div>
                <Label htmlFor="healthConditions">Health Conditions</Label>
                <Textarea
                  id="healthConditions"
                  value={formData.healthConditions}
                  onChange={(e) => updateFormData('healthConditions', e.target.value)}
                  placeholder="List any health conditions, allergies, or medical concerns"
                />
              </div>

              <div>
                <Label htmlFor="medications">Current Medications</Label>
                <Textarea
                  id="medications"
                  value={formData.medications}
                  onChange={(e) => updateFormData('medications', e.target.value)}
                  placeholder="List any medications you are currently taking"
                />
              </div>

              <div>
                <Label htmlFor="injuries">Injuries or Limitations</Label>
                <Textarea
                  id="injuries"
                  value={formData.injuries}
                  onChange={(e) => updateFormData('injuries', e.target.value)}
                  placeholder="Describe any injuries, pain, or physical limitations"
                />
              </div>
            </div>
          )}

          {/* Step 3: Health Baseline */}
          {currentStep === 3 && (
            <div className="space-y-4">
              <h2 className="text-2xl font-bold text-white mb-4">Health Baseline</h2>

              <div>
                <Label htmlFor="restingHeartRate">Resting Heart Rate (bpm)</Label>
                <Input
                  id="restingHeartRate"
                  type="number"
                  value={formData.restingHeartRate}
                  onChange={(e) => updateFormData('restingHeartRate', e.target.value)}
                  placeholder="If known"
                />
              </div>

              <div>
                <Label htmlFor="bloodPressure">Blood Pressure</Label>
                <Input
                  id="bloodPressure"
                  value={formData.bloodPressure}
                  onChange={(e) => updateFormData('bloodPressure', e.target.value)}
                  placeholder="e.g., 120/80"
                />
              </div>

              <div>
                <Label htmlFor="vo2Max">VO2 Max (if known)</Label>
                <Input
                  id="vo2Max"
                  type="number"
                  value={formData.vo2Max}
                  onChange={(e) => updateFormData('vo2Max', e.target.value)}
                  placeholder="Optional"
                />
              </div>

              <div>
                <Label htmlFor="sleepQuality">Sleep Quality (1-10)</Label>
                <Input
                  id="sleepQuality"
                  type="number"
                  min="1"
                  max="10"
                  value={formData.sleepQuality}
                  onChange={(e) => updateFormData('sleepQuality', e.target.value)}
                />
              </div>

              <div>
                <Label htmlFor="stressLevel">Stress Level (1-10)</Label>
                <Input
                  id="stressLevel"
                  type="number"
                  min="1"
                  max="10"
                  value={formData.stressLevel}
                  onChange={(e) => updateFormData('stressLevel', e.target.value)}
                />
              </div>
            </div>
          )}

          {/* Step 4: Training Experience */}
          {currentStep === 4 && (
            <div className="space-y-4">
              <h2 className="text-2xl font-bold text-white mb-4">Training Experience</h2>

              <div>
                <Label htmlFor="trainingYears">Years of Training Experience</Label>
                <Input
                  id="trainingYears"
                  type="number"
                  value={formData.trainingYears}
                  onChange={(e) => updateFormData('trainingYears', e.target.value)}
                />
              </div>

              <div>
                <Label htmlFor="primaryGoals">Primary Training Goals</Label>
                <Textarea
                  id="primaryGoals"
                  value={formData.primaryGoals}
                  onChange={(e) => updateFormData('primaryGoals', e.target.value)}
                  placeholder="e.g., Strength, endurance, weight loss, muscle gain, general health"
                />
              </div>

              <div>
                <Label htmlFor="trainingFrequency">Training Frequency (days/week)</Label>
                <Input
                  id="trainingFrequency"
                  type="number"
                  min="1"
                  max="7"
                  value={formData.trainingFrequency}
                  onChange={(e) => updateFormData('trainingFrequency', e.target.value)}
                />
              </div>

              <div>
                <Label htmlFor="availableEquipment">Available Equipment</Label>
                <Textarea
                  id="availableEquipment"
                  value={formData.availableEquipment}
                  onChange={(e) => updateFormData('availableEquipment', e.target.value)}
                  placeholder="List equipment you have access to"
                />
              </div>

              <div>
                <Label htmlFor="trainingStyle">Preferred Training Style</Label>
                <select
                  id="trainingStyle"
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  value={formData.trainingStyle}
                  onChange={(e) => updateFormData('trainingStyle', e.target.value)}
                >
                  <option value="">Select</option>
                  <option value="strength">Strength Training</option>
                  <option value="cardio">Cardio</option>
                  <option value="hybrid">Hybrid</option>
                  <option value="functional">Functional</option>
                  <option value="sports">Sports Specific</option>
                </select>
              </div>
            </div>
          )}

          {/* Step 5: Nutrition & Lifestyle */}
          {currentStep === 5 && (
            <div className="space-y-4">
              <h2 className="text-2xl font-bold text-white mb-4">Nutrition & Lifestyle</h2>

              <div>
                <Label htmlFor="dietaryRestrictions">Dietary Restrictions</Label>
                <Textarea
                  id="dietaryRestrictions"
                  value={formData.dietaryRestrictions}
                  onChange={(e) => updateFormData('dietaryRestrictions', e.target.value)}
                  placeholder="e.g., Vegan, keto, paleo, allergies"
                />
              </div>

              <div>
                <Label htmlFor="eatingSchedule">Eating Schedule Preference</Label>
                <Input
                  id="eatingSchedule"
                  value={formData.eatingSchedule}
                  onChange={(e) => updateFormData('eatingSchedule', e.target.value)}
                  placeholder="e.g., 3 meals, intermittent fasting, etc."
                />
              </div>

              <div>
                <Label htmlFor="supplements">Current Supplements</Label>
                <Textarea
                  id="supplements"
                  value={formData.supplements}
                  onChange={(e) => updateFormData('supplements', e.target.value)}
                  placeholder="List any supplements you currently take"
                />
              </div>

              <div>
                <Label htmlFor="alcoholConsumption">Alcohol Consumption (drinks/week)</Label>
                <Input
                  id="alcoholConsumption"
                  type="number"
                  min="0"
                  value={formData.alcoholConsumption}
                  onChange={(e) => updateFormData('alcoholConsumption', e.target.value)}
                />
              </div>

              <div>
                <Label htmlFor="smokingStatus">Smoking Status</Label>
                <select
                  id="smokingStatus"
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  value={formData.smokingStatus}
                  onChange={(e) => updateFormData('smokingStatus', e.target.value)}
                >
                  <option value="">Select</option>
                  <option value="never">Never</option>
                  <option value="former">Former</option>
                  <option value="current">Current</option>
                </select>
              </div>

              <div>
                <Label htmlFor="sleepSchedule">Sleep Schedule</Label>
                <Input
                  id="sleepSchedule"
                  value={formData.sleepSchedule}
                  onChange={(e) => updateFormData('sleepSchedule', e.target.value)}
                  placeholder="e.g., 11pm - 7am"
                />
              </div>
            </div>
          )}

          {/* Step 6: Goals & Expectations */}
          {currentStep === 6 && (
            <div className="space-y-4">
              <h2 className="text-2xl font-bold text-white mb-4">Goals & Expectations</h2>

              <div>
                <Label htmlFor="primaryGoal">Primary Goal (Detailed)</Label>
                <Textarea
                  id="primaryGoal"
                  value={formData.primaryGoal}
                  onChange={(e) => updateFormData('primaryGoal', e.target.value)}
                  placeholder="Describe your main fitness goal in detail"
                  rows={4}
                />
              </div>

              <div>
                <Label htmlFor="secondaryGoals">Secondary Goals (up to 3)</Label>
                <Textarea
                  id="secondaryGoals"
                  value={formData.secondaryGoals}
                  onChange={(e) => updateFormData('secondaryGoals', e.target.value)}
                  placeholder="List your secondary goals"
                />
              </div>

              <div>
                <Label htmlFor="timeline">Timeline for Results</Label>
                <select
                  id="timeline"
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  value={formData.timeline}
                  onChange={(e) => updateFormData('timeline', e.target.value)}
                >
                  <option value="">Select</option>
                  <option value="8-weeks">8 weeks</option>
                  <option value="12-weeks">12 weeks</option>
                  <option value="6-months">6 months</option>
                  <option value="1-year">1 year</option>
                </select>
              </div>

              <div>
                <Label htmlFor="weeklyCommitment">Expected Commitment (hours/week)</Label>
                <Input
                  id="weeklyCommitment"
                  type="number"
                  min="1"
                  value={formData.weeklyCommitment}
                  onChange={(e) => updateFormData('weeklyCommitment', e.target.value)}
                />
              </div>

              <div>
                <Label htmlFor="successMetrics">Success Metrics</Label>
                <Textarea
                  id="successMetrics"
                  value={formData.successMetrics}
                  onChange={(e) => updateFormData('successMetrics', e.target.value)}
                  placeholder="How will you measure success?"
                  rows={3}
                />
              </div>
            </div>
          )}

          {/* Step 7: Payment Selection */}
          {currentStep === 7 && (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-white mb-4">Select Your Plan</h2>

              <div className="space-y-4">
                {/* Core Program */}
                <div
                  className={`p-6 border-2 rounded-lg cursor-pointer transition-all ${
                    formData.paymentType === 'core'
                      ? 'border-blue-500 bg-blue-500/10'
                      : 'border-surface-elevated hover:border-blue-500/50'
                  }`}
                  onClick={() => updateFormData('paymentType', 'core')}
                >
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="text-xl font-bold text-white">Core Program</h3>
                    <span className="text-2xl font-bold text-blue-400">$89<span className="text-sm font-normal">/mo</span></span>
                  </div>
                  <p className="text-white/70 mb-4">Build Your Foundation</p>
                  <ul className="space-y-2 text-sm text-white/80">
                    <li>✓ Personalized training program</li>
                    <li>✓ Nutrition plan</li>
                    <li>✓ Dashboard access (Static)</li>
                    <li>✓ Q&A Access (Limited)</li>
                    <li>✓ 72h response time</li>
                  </ul>
                </div>

                {/* Adaptive Engine */}
                <div
                  className={`p-6 border-2 rounded-lg cursor-pointer transition-all ${
                    formData.paymentType === 'adaptive'
                      ? 'border-repz-orange bg-repz-orange/10'
                      : 'border-surface-elevated hover:border-repz-orange/50'
                  }`}
                  onClick={() => updateFormData('paymentType', 'adaptive')}
                >
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <h3 className="text-xl font-bold text-white">Adaptive Engine</h3>
                      <span className="text-xs bg-repz-orange text-white px-2 py-0.5 rounded">NEW FEATURES</span>
                    </div>
                    <span className="text-2xl font-bold text-repz-orange">$149<span className="text-sm font-normal">/mo</span></span>
                  </div>
                  <p className="text-white/70 mb-4">Intelligent Optimization</p>
                  <ul className="space-y-2 text-sm text-white/80">
                    <li>✓ Everything in Core +</li>
                    <li>✓ Weekly check-ins & photos</li>
                    <li>✓ Wearable device integration</li>
                    <li>✓ Biomarker integration</li>
                    <li>✓ Auto grocery lists</li>
                    <li>✓ 48h response time</li>
                  </ul>
                </div>

                {/* Prime Suite (Performance) */}
                <div
                  className={`p-6 border-2 rounded-lg cursor-pointer transition-all relative ${
                    formData.paymentType === 'performance'
                      ? 'border-purple-500 bg-purple-500/10'
                      : 'border-surface-elevated hover:border-purple-500/50'
                  }`}
                  onClick={() => updateFormData('paymentType', 'performance')}
                >
                  <div className="absolute -top-3 right-4 bg-purple-500 text-white text-xs px-3 py-1 rounded-full font-bold">MOST POPULAR</div>
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="text-xl font-bold text-white">Prime Suite</h3>
                    <span className="text-2xl font-bold text-purple-400">$229<span className="text-sm font-normal">/mo</span></span>
                  </div>
                  <p className="text-white/70 mb-4">Elite Optimization</p>
                  <ul className="space-y-2 text-sm text-white/80">
                    <li>✓ Everything in Adaptive +</li>
                    <li>✓ AI fitness assistant</li>
                    <li>✓ Real-time form analysis</li>
                    <li>✓ PEDs protocols</li>
                    <li>✓ Custom cycling schemes</li>
                    <li>✓ 24h response time</li>
                  </ul>
                </div>

                {/* Elite Concierge (Longevity) */}
                <div
                  className={`p-6 border-2 rounded-lg cursor-pointer transition-all relative ${
                    formData.paymentType === 'longevity'
                      ? 'border-yellow-500 bg-yellow-500/10'
                      : 'border-surface-elevated hover:border-yellow-500/50'
                  }`}
                  onClick={() => updateFormData('paymentType', 'longevity')}
                >
                  <div className="absolute -top-3 right-4 bg-yellow-500 text-black text-xs px-3 py-1 rounded-full font-bold">LIMITED</div>
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="text-xl font-bold text-white">Elite Concierge</h3>
                    <span className="text-2xl font-bold text-yellow-400">$349<span className="text-sm font-normal">/mo</span></span>
                  </div>
                  <p className="text-white/70 mb-4">Ultimate Optimization</p>
                  <ul className="space-y-2 text-sm text-white/80">
                    <li>✓ Everything in Prime Suite +</li>
                    <li>✓ In-person training: 2×/week</li>
                    <li>✓ Unlimited Q&A access</li>
                    <li>✓ White-glove concierge service</li>
                    <li>✓ 12h response time</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {/* Navigation Buttons */}
          <div className="flex justify-between mt-8 pt-6 border-t border-surface-elevated">
            <Button
              onClick={handleBack}
              variant="outline"
              disabled={currentStep === 1 || isSubmitting}
              className="flex items-center gap-2"
            >
              <ArrowLeft className="h-4 w-4" />
              Back
            </Button>

            {currentStep < TOTAL_STEPS ? (
              <Button
                onClick={handleNext}
                className="flex items-center gap-2 bg-repz-orange hover:bg-repz-orange-dark"
              >
                Next
                <ArrowRight className="h-4 w-4" />
              </Button>
            ) : (
              <Button
                onClick={handleSubmit}
                disabled={isSubmitting || !formData.paymentType}
                className="flex items-center gap-2 bg-repz-orange hover:bg-repz-orange-dark"
              >
                {isSubmitting ? 'Submitting...' : 'Submit Application'}
                <CheckCircle className="h-4 w-4" />
              </Button>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
