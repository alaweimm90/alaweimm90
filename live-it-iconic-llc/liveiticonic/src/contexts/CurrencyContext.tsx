import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  ReactNode,
} from 'react';
import { CurrencyCode, currencies } from '@/lib/currency';

interface CurrencyContextValue {
  currency: CurrencyCode;
  setCurrency: (code: CurrencyCode) => void;
  rates: Record<string, number>;
  isLoading: boolean;
  error: string | null;
  conversionRate: (from: CurrencyCode, to: CurrencyCode) => number;
}

const CurrencyContext = createContext<CurrencyContextValue | undefined>(
  undefined
);

// Mock exchange rates for demonstration
const MOCK_EXCHANGE_RATES: Record<string, number> = {
  'USD_EUR': 0.92,
  'USD_GBP': 0.79,
  'USD_CAD': 1.36,
  'USD_JPY': 150.5,
  'USD_AUD': 1.54,
  'EUR_USD': 1.087,
  'EUR_GBP': 0.86,
  'EUR_CAD': 1.48,
  'EUR_JPY': 163.5,
  'EUR_AUD': 1.67,
  'GBP_USD': 1.266,
  'GBP_EUR': 1.163,
  'GBP_CAD': 1.72,
  'GBP_JPY': 190.1,
  'GBP_AUD': 1.95,
  'CAD_USD': 0.735,
  'CAD_EUR': 0.676,
  'CAD_GBP': 0.581,
  'CAD_JPY': 110.7,
  'CAD_AUD': 1.13,
  'JPY_USD': 0.00664,
  'JPY_EUR': 0.00612,
  'JPY_GBP': 0.00526,
  'JPY_CAD': 0.00903,
  'JPY_AUD': 0.0102,
  'AUD_USD': 0.649,
  'AUD_EUR': 0.599,
  'AUD_GBP': 0.513,
  'AUD_CAD': 0.884,
  'AUD_JPY': 97.8,
};

/**
 * Fetch real exchange rates from an API
 * You can replace this with a real API call
 */
async function fetchExchangeRates(): Promise<Record<string, number>> {
  try {
    // In production, you would fetch from a real exchange rate API
    // Example: https://api.exchangerate-api.com/v4/latest/USD
    // For now, we'll return mock data
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve(MOCK_EXCHANGE_RATES);
      }, 500);
    });
  } catch (error) {
    console.error('Failed to fetch exchange rates:', error);
    return MOCK_EXCHANGE_RATES; // Fallback to mock data
  }
}

/**
 * Currency Provider Component
 * Provides currency context to all child components
 */
export function CurrencyProvider({ children }: { children: ReactNode }) {
  const [currency, setCurrencyState] = useState<CurrencyCode>('USD');
  const [rates, setRates] = useState<Record<string, number>>(
    MOCK_EXCHANGE_RATES
  );
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load initial exchange rates
  useEffect(() => {
    const loadRates = async () => {
      setIsLoading(true);
      try {
        const fetchedRates = await fetchExchangeRates();
        setRates(fetchedRates);
        setError(null);
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : 'Failed to load exchange rates';
        setError(errorMessage);
        console.error('Error loading exchange rates:', err);
      } finally {
        setIsLoading(false);
      }
    };

    loadRates();

    // Optionally refresh rates every hour
    const intervalId = setInterval(loadRates, 3600000);
    return () => clearInterval(intervalId);
  }, []);

  // Save currency preference to localStorage
  const setCurrency = useCallback((code: CurrencyCode) => {
    setCurrencyState(code);
    try {
      localStorage.setItem('preferred-currency', code);
    } catch (err) {
      console.warn('Failed to save currency preference:', err);
    }
  }, []);

  // Get conversion rate between two currencies
  const conversionRate = useCallback(
    (from: CurrencyCode, to: CurrencyCode): number => {
      if (from === to) return 1;
      const rateKey = `${from}_${to}`;
      return rates[rateKey] ?? 1;
    },
    [rates]
  );

  // Load saved currency preference on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem('preferred-currency');
      if (saved && saved in currencies) {
        setCurrencyState(saved as CurrencyCode);
      }
    } catch (err) {
      console.warn('Failed to load currency preference:', err);
    }
  }, []);

  const value: CurrencyContextValue = {
    currency,
    setCurrency,
    rates,
    isLoading,
    error,
    conversionRate,
  };

  return (
    <CurrencyContext.Provider value={value}>
      {children}
    </CurrencyContext.Provider>
  );
}

/**
 * Hook to use currency context
 * @returns Currency context value
 * @throws Error if used outside of CurrencyProvider
 */
export function useCurrency(): CurrencyContextValue {
  const context = useContext(CurrencyContext);
  if (!context) {
    throw new Error('useCurrency must be used within CurrencyProvider');
  }
  return context;
}
