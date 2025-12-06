/**
 * Stripe Service Module
 *
 * Provides a clean interface for Stripe payment operations.
 * Handles payment intent creation, confirmation, and error handling.
 *
 * @module services/stripeService
 */

import { getStripe } from '@/lib/stripe';
import Stripe from '@stripe/stripe-js';

/**
 * Payment intent creation request payload
 */
export interface CreatePaymentIntentRequest {
  amount: number;
  currency?: string;
  metadata?: Record<string, string>;
  description?: string;
}

/**
 * Payment intent response
 */
export interface PaymentIntentResponse {
  clientSecret: string;
  paymentIntentId: string;
  status: string;
}

/**
 * Stripe service for payment operations
 */
export const stripeService = {
  /**
   * Create a payment intent on the server
   * @param request Payment intent request details
   * @returns Payment intent response with client secret
   * @throws Error if payment intent creation fails
   */
  async createPaymentIntent(request: CreatePaymentIntentRequest): Promise<PaymentIntentResponse> {
    const response = await fetch('/api/payments/create-intent', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to create payment intent');
    }

    const data = (await response.json()) as PaymentIntentResponse;
    return data;
  },

  /**
   * Confirm a card payment using the payment element
   * @param stripe Stripe instance
   * @param elements Stripe elements instance
   * @param clientSecret Payment intent client secret
   * @param returnUrl URL to redirect to after payment
   * @returns Stripe payment result
   */
  async confirmPayment(
    stripe: Stripe.Stripe,
    elements: Stripe.StripeElements,
    clientSecret: string,
    returnUrl: string
  ): Promise<Stripe.ConfirmPaymentResult> {
    return stripe.confirmPayment({
      elements,
      clientSecret,
      confirmParams: {
        return_url: returnUrl,
      },
      redirect: 'if_required',
    });
  },

  /**
   * Confirm a card payment using CardElement (legacy method)
   * @param stripe Stripe instance
   * @param clientSecret Payment intent client secret
   * @param cardElement Card element from form
   * @param billingDetails Customer billing information
   * @returns Payment confirmation result
   */
  async confirmCardPayment(
    stripe: Stripe.Stripe,
    clientSecret: string,
    cardElement: Stripe.StripeCardElement,
    billingDetails: Stripe.BillingDetails
  ): Promise<Stripe.PaymentIntentResult> {
    return stripe.confirmCardPayment(clientSecret, {
      payment_method: {
        card: cardElement,
        billing_details: billingDetails,
      },
    });
  },

  /**
   * Create a payment method from card element
   * @param stripe Stripe instance
   * @param cardElement Card element from form
   * @param billingDetails Customer billing information
   * @returns Created payment method
   */
  async createPaymentMethod(
    stripe: Stripe.Stripe,
    cardElement: Stripe.StripeCardElement,
    billingDetails: Stripe.BillingDetails
  ): Promise<Stripe.PaymentMethodResult> {
    return stripe.createPaymentMethod({
      type: 'card',
      card: cardElement,
      billing_details: billingDetails,
    });
  },

  /**
   * Retrieve a payment intent from Stripe
   * @param stripe Stripe instance
   * @param clientSecret Payment intent client secret
   * @returns Payment intent details
   */
  async retrievePaymentIntent(
    stripe: Stripe.Stripe,
    clientSecret: string
  ): Promise<Stripe.PaymentIntent> {
    const result = await stripe.retrievePaymentIntent(clientSecret);

    if (result.error) {
      throw new Error(result.error.message || 'Failed to retrieve payment intent');
    }

    return result.paymentIntent;
  },

  /**
   * Retry a failed payment
   * @param stripe Stripe instance
   * @param clientSecret Payment intent client secret
   * @param cardElement Card element from form
   * @param billingDetails Customer billing information
   * @returns Payment confirmation result
   */
  async retryPayment(
    stripe: Stripe.Stripe,
    clientSecret: string,
    cardElement: Stripe.StripeCardElement,
    billingDetails: Stripe.BillingDetails
  ): Promise<Stripe.PaymentIntentResult> {
    return stripeService.confirmCardPayment(stripe, clientSecret, cardElement, billingDetails);
  },

  /**
   * Validate a payment method
   * @param stripe Stripe instance
   * @param cardElement Card element from form
   * @returns Validation result
   */
  async validatePaymentMethod(
    stripe: Stripe.Stripe,
    cardElement: Stripe.StripeCardElement
  ): Promise<{ valid: boolean; error?: string }> {
    try {
      // Attempt to retrieve card details
      const { error, paymentMethod } = await stripe.createPaymentMethod({
        type: 'card',
        card: cardElement,
      });

      if (error) {
        return {
          valid: false,
          error: error.message || 'Invalid payment method',
        };
      }

      return {
        valid: !!paymentMethod,
      };
    } catch (error) {
      return {
        valid: false,
        error: error instanceof Error ? error.message : 'Validation failed',
      };
    }
  },

  /**
   * Handle payment error and return user-friendly message
   * @param error Stripe error
   * @returns User-friendly error message
   */
  getErrorMessage(error: Stripe.StripeError): string {
    const errorMessages: Record<string, string> = {
      card_declined: 'Your card was declined. Please try another payment method.',
      expired_card: 'Your card has expired. Please use a valid card.',
      incorrect_cvc: 'The CVC code you entered is incorrect.',
      processing_error: 'An error occurred while processing your payment. Please try again.',
      rate_limit:
        'Too many payment attempts. Please wait a moment and try again.',
      authentication_error: 'Payment authentication failed. Please try again.',
      invalid_expiry_month: 'The expiration month is invalid.',
      invalid_expiry_year: 'The expiration year is invalid.',
    };

    return (
      errorMessages[error.code || ''] || error.message || 'Payment failed. Please try again.'
    );
  },

  /**
   * Initialize Stripe instance
   * @returns Stripe instance promise
   */
  async initialize(): Promise<Stripe.Stripe | null> {
    return getStripe();
  },

  /**
   * Check if payment requires action (3D Secure, etc.)
   * @param paymentIntent Payment intent from Stripe
   * @returns True if additional action is required
   */
  requiresAction(paymentIntent: Stripe.PaymentIntent): boolean {
    return paymentIntent.status === 'requires_action';
  },

  /**
   * Check if payment was successful
   * @param paymentIntent Payment intent from Stripe
   * @returns True if payment succeeded
   */
  isSuccessful(paymentIntent: Stripe.PaymentIntent): boolean {
    return paymentIntent.status === 'succeeded';
  },

  /**
   * Format amount for display (converts cents to dollars)
   * @param amountInCents Amount in cents
   * @returns Formatted amount string
   */
  formatAmount(amountInCents: number): string {
    return `$${(amountInCents / 100).toFixed(2)}`;
  },

  /**
   * Convert dollars to cents
   * @param amountInDollars Amount in dollars
   * @returns Amount in cents
   */
  dollarsToCents(amountInDollars: number): number {
    return Math.round(amountInDollars * 100);
  },

  /**
   * Convert cents to dollars
   * @param amountInCents Amount in cents
   * @returns Amount in dollars
   */
  centsToDollars(amountInCents: number): number {
    return amountInCents / 100;
  },
};

/**
 * Type definitions for Stripe service operations
 */
export type { Stripe };
