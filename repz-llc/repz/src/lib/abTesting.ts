/**
 * A/B Testing stub - replace with actual implementation when needed
 */

import { useState, useEffect } from 'react';

export interface ABTestConfig {
  name: string;
  variants: string[];
  weights?: number[];
}

export function useABTest(testName: string, variants: string[] = ['control', 'variant']): string {
  const [variant, setVariant] = useState<string>(variants[0]);

  useEffect(() => {
    // Check localStorage for existing assignment
    const stored = localStorage.getItem(`ab_test_${testName}`);
    if (stored && variants.includes(stored)) {
      setVariant(stored);
      return;
    }

    // Randomly assign variant
    const randomVariant = variants[Math.floor(Math.random() * variants.length)];
    localStorage.setItem(`ab_test_${testName}`, randomVariant);
    setVariant(randomVariant);

    if (import.meta.env.DEV) {
      console.log(`[A/B Test] ${testName}: assigned to ${randomVariant}`);
    }
  }, [testName, variants]);

  return variant;
}

export function trackABTestConversion(testName: string, eventName: string): void {
  const variant = localStorage.getItem(`ab_test_${testName}`);
  if (import.meta.env.DEV) {
    console.log(`[A/B Test] Conversion: ${testName}/${variant} - ${eventName}`);
  }
}

export function getABTestVariant(testName: string): string | null {
  return localStorage.getItem(`ab_test_${testName}`);
}
