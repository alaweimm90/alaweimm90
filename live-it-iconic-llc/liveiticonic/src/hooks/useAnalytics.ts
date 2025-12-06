/**
 * React Hook for Analytics Tracking
 * Provides access to comprehensive analytics functions with automatic page view tracking
 */

import { useEffect, useCallback } from 'react';
import { useLocation } from 'react-router-dom';
import { getAnalytics, AnalyticsProperties, EcommerceProduct } from '@/lib/analytics';
import { getConsentManager } from '@/lib/consentManager';
import { getErrorTracker } from '@/lib/errorTracking';

export const useAnalytics = () => {
  const location = useLocation();
  const analytics = getAnalytics();
  const consentManager = getConsentManager();
  const errorTracker = getErrorTracker();

  // Track page view on route change
  useEffect(() => {
    // Check if user has consent for analytics
    if (consentManager.hasConsent('analytics')) {
      analytics.pageView(location.pathname, document.title);
      errorTracker.addNavigationBreadcrumb(location.pathname, location.pathname);
    }
  }, [location.pathname]);

  // ========================================================================
  // Page Tracking
  // ========================================================================

  const trackPageView = useCallback((path: string, title?: string) => {
    if (consentManager.hasConsent('analytics')) {
      analytics.pageView(path, title);
    }
  }, []);

  // ========================================================================
  // Custom Event Tracking
  // ========================================================================

  const track = useCallback((eventName: string, properties?: AnalyticsProperties) => {
    analytics.track(eventName, properties);
  }, []);

  // ========================================================================
  // E-commerce Events
  // ========================================================================

  const trackProductView = useCallback((product: EcommerceProduct) => {
    if (consentManager.hasConsent('analytics')) {
      analytics.viewItem(product);
      errorTracker.addBreadcrumb(`Viewed product: ${product.name}`, 'ecommerce', {
        productId: product.id,
        price: product.price,
      });
    }
  }, []);

  const trackAddToCart = useCallback((product: EcommerceProduct) => {
    if (consentManager.hasConsent('analytics')) {
      analytics.addToCart(product);
      errorTracker.addBreadcrumb(`Added to cart: ${product.name}`, 'ecommerce', {
        productId: product.id,
        quantity: product.quantity,
      });
    }
  }, []);

  const trackRemoveFromCart = useCallback((product: EcommerceProduct) => {
    if (consentManager.hasConsent('analytics')) {
      analytics.removeFromCart(product);
      errorTracker.addBreadcrumb(`Removed from cart: ${product.name}`, 'ecommerce');
    }
  }, []);

  const trackCheckoutStarted = useCallback((items: any[], value: number) => {
    if (consentManager.hasConsent('analytics')) {
      const convertedItems = items.map(item => ({
        item_id: item.id,
        item_name: item.name,
        price: item.price,
        quantity: item.quantity,
      }));
      analytics.beginCheckout(convertedItems, value);
      errorTracker.addBreadcrumb('Checkout started', 'ecommerce', { itemCount: items.length });
    }
  }, []);

  const trackCheckoutStep = useCallback((step: string, stepNumber: number) => {
    if (consentManager.hasConsent('analytics')) {
      track('checkout_step', { step, step_number: stepNumber });
      errorTracker.addBreadcrumb(`Checkout step: ${step}`, 'ecommerce');
    }
  }, [track]);

  const trackPurchase = useCallback((
    transactionId: string,
    value: number,
    items: any[],
    currency: string = 'USD',
    tax?: number,
    shipping?: number,
    coupon?: string
  ) => {
    if (consentManager.hasConsent('analytics')) {
      const convertedItems = items.map(item => ({
        item_id: item.id,
        item_name: item.name,
        item_brand: item.brand,
        item_category: item.category,
        price: item.price,
        quantity: item.quantity,
      }));

      analytics.purchase({
        transaction_id: transactionId,
        value,
        currency,
        tax,
        shipping,
        coupon,
        items: convertedItems,
      });

      errorTracker.addBreadcrumb('Purchase completed', 'ecommerce', {
        transactionId,
        value,
        itemCount: items.length,
      });
    }
  }, []);

  const trackRefund = useCallback((
    transactionId: string,
    value: number,
    items?: any[],
    currency: string = 'USD'
  ) => {
    if (consentManager.hasConsent('analytics')) {
      const convertedItems = items?.map(item => ({
        item_id: item.id,
        item_name: item.name,
        price: item.price,
        quantity: item.quantity,
      }));

      analytics.refund(transactionId, value, currency, convertedItems);
      errorTracker.addBreadcrumb('Refund processed', 'ecommerce', { transactionId });
    }
  }, []);

  // ========================================================================
  // User Engagement Events
  // ========================================================================

  const trackButtonClick = useCallback((buttonName: string, properties?: AnalyticsProperties) => {
    analytics.buttonClick(buttonName, properties);
    errorTracker.addUserActionBreadcrumb(`Clicked: ${buttonName}`);
  }, []);

  const trackLinkClick = useCallback((linkName: string, url?: string, isExternalLink: boolean = false) => {
    analytics.linkClick(linkName, url, isExternalLink);
    errorTracker.addUserActionBreadcrumb(`Clicked link: ${linkName}`, url);
  }, []);

  const trackSearch = useCallback((query: string) => {
    if (consentManager.hasConsent('analytics')) {
      analytics.search(query);
      errorTracker.addBreadcrumb(`Search: ${query}`, 'search');
    }
  }, []);

  const trackSocialShare = useCallback((platform: string, contentType?: string, contentTitle?: string) => {
    analytics.socialShare(platform, contentType, contentTitle);
    errorTracker.addBreadcrumb(`Shared on ${platform}`, 'social');
  }, []);

  const trackFormSubmit = useCallback((formName: string, fields?: Record<string, string>) => {
    analytics.formSubmit(formName, fields);
    errorTracker.addBreadcrumb(`Form submitted: ${formName}`, 'form', {
      fieldCount: fields ? Object.keys(fields).length : 0,
    });
  }, []);

  const trackEmailSignup = useCallback((email?: string, source?: string) => {
    analytics.emailSignup(email, source);
    errorTracker.addBreadcrumb('Email signup', 'user');
  }, []);

  const trackLogin = useCallback((method: string = 'email') => {
    analytics.login(method);
    errorTracker.addBreadcrumb('User login', 'user');
  }, []);

  const trackLogout = useCallback(() => {
    analytics.logout();
    errorTracker.addBreadcrumb('User logout', 'user');
  }, []);

  const trackFileDownload = useCallback((filename: string, fileType?: string) => {
    analytics.fileDownload(filename, fileType);
    errorTracker.addBreadcrumb(`Downloaded: ${filename}`, 'download');
  }, []);

  // ========================================================================
  // User Identification
  // ========================================================================

  const identify = useCallback((userId: string, userProperties?: Record<string, any>) => {
    analytics.identify(userId, userProperties);
    errorTracker.setUserContext(userId);
  }, []);

  const setUserId = useCallback((userId: string) => {
    analytics.setUserId(userId);
    errorTracker.setUserContext(userId);
  }, []);

  const setUserProperties = useCallback((properties: Record<string, any>) => {
    analytics.setUserProperties(properties);
  }, []);

  // ========================================================================
  // Reset
  // ========================================================================

  const reset = useCallback(() => {
    analytics.reset();
    errorTracker.clearBreadcrumbs();
  }, []);

  return {
    // Page tracking
    trackPageView,

    // Custom events
    track,

    // E-commerce events
    trackProductView,
    trackAddToCart,
    trackRemoveFromCart,
    trackCheckoutStarted,
    trackCheckoutStep,
    trackPurchase,
    trackRefund,

    // User engagement events
    trackButtonClick,
    trackLinkClick,
    trackSearch,
    trackSocialShare,
    trackFormSubmit,
    trackEmailSignup,
    trackLogin,
    trackLogout,
    trackFileDownload,

    // User identification
    identify,
    setUserId,
    setUserProperties,

    // Utilities
    reset,
  };
};
