import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useValidation, usePasswordValidation } from '@/lib/validation';
import { step1AccountSchema, Step1AccountData } from '@/lib/validation/schemas';
import {
  ValidationInput,
  ValidationCheckbox,
  PasswordStrengthIndicator,
  FormErrorSummary
} from '@/components/validation/ValidationComponents';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { toast } from 'sonner';

interface Step1AccountFormProps {
  onNext: (data: Step1AccountData) => void;
  initialData?: Partial<Step1AccountData>;
}

export const Step1AccountForm: React.FC<Step1AccountFormProps> = ({ 
  onNext, 
  initialData 
}) => {
  const navigate = useNavigate();
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Use the validation hook with Zod schema
  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
    validateForm
  } = useValidation<Step1AccountData>({
    schema: step1AccountSchema,
    mode: 'onChange',
    reValidateMode: 'onChange',
    defaultValues: initialData
  });

  // Password strength validation
  const { passwordStrength, validatePassword } = usePasswordValidation();
  const passwordValue = watch('password');

  // Update password strength when password changes
  React.useEffect(() => {
    if (passwordValue) {
      validatePassword(passwordValue);
    }
  }, [passwordValue, validatePassword]);

  const onSubmit = async (data: Step1AccountData) => {
    try {
      setIsSubmitting(true);

      // Final validation before submission
      const validationResult = validateForm(data);
      if (!validationResult.success) {
        toast.error('Please correct the errors before proceeding');
        return;
      }

      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Proceed to next step
      onNext(data);
      toast.success('Account information saved successfully!');

    } catch (error) {
      console.error('Form submission error:', error);
      toast.error('Failed to save account information. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleGoogleSignUp = () => {
    // Implement Google OAuth sign-up
    toast.info('Google sign-up coming soon!');
  };

  const handleAppleSignUp = () => {
    // Implement Apple sign-up
    toast.info('Apple sign-up coming soon!');
  };

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader className="space-y-1">
        <CardTitle className="text-2xl font-bold text-center">Create Account</CardTitle>
        <CardDescription className="text-center">
          Enter your information to get started
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Social Sign Up Options */}
        <div className="space-y-3">
          <Button
            type="button"
            variant="outline"
            className="w-full"
            onClick={handleGoogleSignUp}
            disabled={isSubmitting}
          >
            <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24">
              <path
                fill="currentColor"
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
              />
              <path
                fill="currentColor"
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
              />
              <path
                fill="currentColor"
                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
              />
              <path
                fill="currentColor"
                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
              />
            </svg>
            Continue with Google
          </Button>
          
          <Button
            type="button"
            variant="outline"
            className="w-full"
            onClick={handleAppleSignUp}
            disabled={isSubmitting}
          >
            <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="currentColor">
              <path d="M17.05 12.28c-.98.95-2.05 2.28-2.05 3.9 0 2.28 1.68 3.78 4.2 3.78 1.12 0 2.1-.23 3.15-.7-.28.98-.7 1.96-1.26 2.94-1.12 1.96-2.94 4.41-5.32 4.41-2.52 0-4.2-1.54-4.2-4.2 0-2.52 1.4-4.62 3.5-6.02.98-.7 2.1-1.26 3.22-1.54-.28.42-.56.98-.56 1.68zM12.72 2.8c1.4 0 2.66.56 3.5 1.54-.98.42-1.82 1.12-2.66 1.96-1.4 1.26-2.52 2.94-2.52 4.9 0 .98.28 1.96.84 2.66-1.96-.7-3.64-2.52-3.64-5.18 0-2.52 1.54-4.34 3.5-5.32.7-.28 1.4-.56 2.1-.56z"/>
            </svg>
            Continue with Apple
          </Button>
        </div>

        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <Separator />
          </div>
          <div className="relative flex justify-center text-xs uppercase">
            <span className="bg-background px-2 text-muted-foreground">
              Or continue with
            </span>
          </div>
        </div>

        {/* Error Summary */}
        {Object.keys(errors).length > 0 && (
          <FormErrorSummary errors={errors} />
        )}

        {/* Form Fields */}
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <ValidationInput
            name="fullName"
            label="Full Name"
            placeholder="John Doe"
            register={register}
            errors={errors}
            validation={{ required: 'Full name is required' }}
            disabled={isSubmitting}
          />

          <ValidationInput
            name="email"
            label="Email Address"
            type="email"
            placeholder="john@example.com"
            register={register}
            errors={errors}
            validation={{ required: 'Email is required' }}
            disabled={isSubmitting}
          />

          <div className="space-y-2">
            <ValidationInput
              name="password"
              label="Password"
              type="password"
              placeholder="Create a strong password"
              register={register}
              errors={errors}
              validation={{ required: 'Password is required' }}
              showPasswordToggle
              disabled={isSubmitting}
            />
            
            {passwordValue && (
              <PasswordStrengthIndicator 
                strength={passwordStrength}
                showFeedback={passwordStrength.level !== 'strong'}
              />
            )}
          </div>

          <ValidationInput
            name="confirmPassword"
            label="Confirm Password"
            type="password"
            placeholder="Confirm your password"
            register={register}
            errors={errors}
            validation={{ required: 'Please confirm your password' }}
            showPasswordToggle
            disabled={isSubmitting}
          />

          <ValidationCheckbox
            name="termsAccepted"
            label="I agree to the Terms of Service and Privacy Policy"
            register={register}
            errors={errors}
            validation={{ required: 'You must accept the terms' }}
            disabled={isSubmitting}
          />

          <Button
            type="submit"
            className="w-full"
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Creating Account...' : 'Continue'}
          </Button>
        </form>

        <div className="text-center text-sm text-gray-600">
          Already have an account?{' '}
          <button
            type="button"
            onClick={() => navigate('/login')}
            className="font-medium text-blue-600 hover:text-blue-500"
          >
            Sign in
          </button>
        </div>
      </CardContent>
    </Card>
  );
};

export default Step1AccountForm;