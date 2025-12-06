import { useTranslation } from 'react-i18next';
import { useCurrency } from '@/contexts/CurrencyContext';
import { formatCurrency, convertCurrency, CurrencyCode } from '@/lib/currency';

/**
 * Combined hook for using i18n and currency features together
 * Provides convenient methods for formatting and converting currencies
 */
export function useI18nCurrency() {
  const { i18n } = useTranslation();
  const { currency, rates, conversionRate } = useCurrency();

  /**
   * Format amount in current currency
   */
  const format = (amount: number): string => {
    return formatCurrency(amount, currency);
  };

  /**
   * Format amount in a specific currency
   */
  const formatIn = (amount: number, targetCurrency: CurrencyCode): string => {
    return formatCurrency(amount, targetCurrency);
  };

  /**
   * Convert and format amount from one currency to another
   */
  const convertAndFormat = (
    amount: number,
    fromCurrency: CurrencyCode,
    toCurrency?: CurrencyCode
  ): string => {
    const targetCurrency = toCurrency || currency;
    const converted = convertCurrency(amount, fromCurrency, targetCurrency, rates);
    return formatCurrency(converted, targetCurrency);
  };

  /**
   * Get currency conversion rate
   */
  const getRate = (from: CurrencyCode, to: CurrencyCode): number => {
    return conversionRate(from, to);
  };

  /**
   * Get current language
   */
  const getLanguage = (): string => {
    return i18n.language;
  };

  /**
   * Get current currency
   */
  const getCurrency = (): CurrencyCode => {
    return currency;
  };

  return {
    format,
    formatIn,
    convertAndFormat,
    getRate,
    getLanguage,
    getCurrency,
    currentLanguage: i18n.language,
    currentCurrency: currency,
  };
}
