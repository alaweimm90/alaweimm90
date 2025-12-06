# Live It Iconic - Internationalization (i18n) System

## Overview

This document outlines the comprehensive internationalization and currency system implemented for Live It Iconic. The system supports multiple languages and currencies, providing a seamless experience for global customers.

## System Architecture

### Components

1. **i18n Configuration** (`src/i18n/config.ts`)
   - i18next setup with React integration
   - Language detection via browser locale and localStorage
   - Resource loading for multiple languages

2. **Translation Files** (`src/i18n/locales/`)
   - English (en.json)
   - French (fr.json)
   - Spanish (es.json)

3. **Currency System** (`src/lib/currency.ts`)
   - Currency formatting utilities
   - Currency conversion helpers
   - Support for 6 major currencies (USD, EUR, GBP, CAD, JPY, AUD)

4. **Context Providers**
   - `CurrencyContext`: Manages global currency state and exchange rates
   - `CurrencyProvider`: Wraps application to provide currency functionality

5. **Components**
   - `LanguageSelector`: Dropdown menu for language selection
   - `CurrencySelector`: Dropdown for currency selection
   - `I18nExample`: Example component demonstrating all features

6. **Hooks**
   - `useI18nCurrency`: Combined hook for translations and currency formatting

## Supported Languages

| Code | Name | Flag |
|------|------|------|
| en | English | 🇺🇸 |
| fr | Français | 🇫🇷 |
| es | Español | 🇪🇸 |

## Supported Currencies

| Code | Symbol | Locale |
|------|--------|--------|
| USD | $ | en-US |
| EUR | € | de-DE |
| GBP | £ | en-GB |
| CAD | CA$ | en-CA |
| JPY | ¥ | ja-JP |
| AUD | A$ | en-AU |

## Installation

The following packages were installed:

```bash
npm install i18next react-i18next i18next-browser-languagedetector currency.js
```

## Usage

### Basic Translation

```tsx
import { useTranslation } from 'react-i18next';

export function MyComponent() {
  const { t, i18n } = useTranslation();

  return (
    <div>
      <h1>{t('hero.title')}</h1>
      <p>{t('hero.subtitle')}</p>
      <button onClick={() => i18n.changeLanguage('es')}>
        {t('common.next')}
      </button>
    </div>
  );
}
```

### Currency Formatting

```tsx
import { useCurrency } from '@/contexts/CurrencyContext';
import { formatCurrency, CurrencyCode } from '@/lib/currency';

export function ProductCard() {
  const { currency } = useCurrency();

  // Format using current currency
  const price = formatCurrency(99.99, currency);

  // Format in specific currency
  const priceInEur = formatCurrency(99.99, 'EUR');

  return (
    <div>
      <p>Price: {price}</p>
      <p>Price in EUR: {priceInEur}</p>
    </div>
  );
}
```

### Currency Conversion

```tsx
import { convertCurrency, CurrencyCode } from '@/lib/currency';

export function ConversionExample() {
  const { rates } = useCurrency();

  // Convert $100 USD to EUR
  const convertedAmount = convertCurrency(100, 'USD', 'EUR', rates);

  return <p>$100 USD = {formatCurrency(convertedAmount, 'EUR')}</p>;
}
```

### Combined i18n & Currency Hook

```tsx
import { useI18nCurrency } from '@/hooks/useI18nCurrency';

export function CheckoutSummary() {
  const { format, convertAndFormat, getLanguage, getCurrency } =
    useI18nCurrency();

  const subtotal = 99.99;
  const tax = 10.0;

  return (
    <div>
      <p>Language: {getLanguage()}</p>
      <p>Currency: {getCurrency()}</p>
      <p>Subtotal: {format(subtotal)}</p>
      <p>Tax: {format(tax)}</p>
      <p>Total: {format(subtotal + tax)}</p>
    </div>
  );
}
```

### Language Selector Component

```tsx
import { LanguageSelector } from '@/components/LanguageSelector';

export function Header() {
  return (
    <header>
      <h1>My App</h1>
      <LanguageSelector />
    </header>
  );
}
```

### Currency Selector Component

```tsx
import { CurrencySelector } from '@/components/CurrencySelector';

export function Header() {
  return (
    <header>
      <h1>My App</h1>
      <CurrencySelector />
    </header>
  );
}
```

## Translation Key Structure

All translations are organized by feature/category:

### Common
- `common.loading`
- `common.error`
- `common.success`
- `common.cancel`
- `common.save`
- `common.delete`
- `common.edit`
- `common.back`
- `common.next`

### Navigation
- `navigation.home`
- `navigation.shop`
- `navigation.about`
- `navigation.contact`
- `navigation.cart`
- `navigation.account`

### Product
- `product.addToCart`
- `product.outOfStock`
- `product.inStock`
- `product.price`
- `product.size`
- `product.color`
- `product.quantity`
- `product.description`
- `product.features`
- `product.reviews`
- `product.rating`

### Cart
- `cart.title`
- `cart.empty`
- `cart.subtotal`
- `cart.shipping`
- `cart.total`
- `cart.checkout`
- `cart.continueShopping`
- `cart.removeItem`
- `cart.updateQuantity`

### Checkout
- `checkout.title`
- `checkout.shippingInfo`
- `checkout.paymentInfo`
- `checkout.orderSummary`
- `checkout.placeOrder`
- `checkout.processingOrder`
- Additional fields for shipping form

### Hero
- `hero.title`
- `hero.tagline`
- `hero.subtitle`
- `hero.cta`

### Authentication
- `auth.login`
- `auth.logout`
- `auth.signup`
- `auth.email`
- `auth.password`
- `auth.confirmPassword`
- `auth.forgotPassword`
- `auth.rememberMe`
- `auth.signupNow`

### Errors
- `errors.invalidEmail`
- `errors.passwordTooShort`
- `errors.passwordMismatch`
- `errors.fieldRequired`
- `errors.serverError`
- `errors.notFound`

### Messages
- `messages.addedToCart`
- `messages.removedFromCart`
- `messages.orderPlaced`
- `messages.profileUpdated`

## Currency API Integration

Currently, the system uses mock exchange rates for demonstration purposes. To integrate with a real exchange rate API:

### Example: Using exchangerate-api.com

```typescript
// In src/contexts/CurrencyContext.tsx
async function fetchExchangeRates(): Promise<Record<string, number>> {
  try {
    const response = await fetch(
      `https://api.exchangerate-api.com/v4/latest/USD`
    );
    const data = await response.json();

    const rates: Record<string, number> = {};

    // Convert API format to our format
    for (const [currency, rate] of Object.entries(data.rates)) {
      rates[`USD_${currency}`] = rate as number;
    }

    return rates;
  } catch (error) {
    console.error('Failed to fetch exchange rates:', error);
    return MOCK_EXCHANGE_RATES;
  }
}
```

## Adding New Languages

To add a new language (e.g., German - de):

1. Create a new translation file: `src/i18n/locales/de.json`
2. Copy the structure from `en.json` and translate the values
3. Update `src/i18n/config.ts`:

```typescript
import deTranslations from './locales/de.json';

i18n.init({
  resources: {
    // ... existing resources
    de: { translation: deTranslations },
  },
  // ... rest of config
});
```

4. Update `LanguageSelector.tsx`:

```typescript
const languages: Language[] = [
  { code: 'en', name: 'English', flag: '🇺🇸' },
  { code: 'fr', name: 'Français', flag: '🇫🇷' },
  { code: 'es', name: 'Español', flag: '🇪🇸' },
  { code: 'de', name: 'Deutsch', flag: '🇩🇪' }, // Add this
];
```

## Adding New Currencies

To add a new currency:

1. Update `src/lib/currency.ts`:

```typescript
export const currencies = {
  // ... existing currencies
  CHF: { symbol: 'CHF', code: 'CHF', locale: 'de-CH' },
};
```

2. Add exchange rates in `src/contexts/CurrencyContext.tsx`:

```typescript
const MOCK_EXCHANGE_RATES: Record<string, number> = {
  // ... existing rates
  'USD_CHF': 0.89,
  'CHF_USD': 1.124,
  // ... add rates for all currency pairs
};
```

## Best Practices

1. **Always use translation keys**: Never hardcode user-facing text
2. **Use namespaces wisely**: Keep translations organized by feature
3. **Test all languages**: Ensure text length doesn't break layouts
4. **Currency precision**: Always format currency to 2 decimal places
5. **Cache preferences**: Language and currency preferences are cached in localStorage
6. **Error handling**: Gracefully fall back to English if needed

## Performance Considerations

1. **Lazy loading**: Translation files are loaded once at initialization
2. **Language detection**: Uses browser locale and localStorage caching
3. **Memoization**: The `useI18nCurrency` hook is optimized for performance
4. **Exchange rates**: Cached and refreshed hourly (configurable)

## Testing

When testing components that use i18n:

```tsx
import { useTranslation } from 'react-i18next';

// Mock i18next in your tests
jest.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) => key,
    i18n: {
      changeLanguage: jest.fn(),
      language: 'en',
    },
  }),
}));
```

## Accessibility

1. **Language selector**: Includes proper ARIA labels
2. **Currency selector**: Supports keyboard navigation
3. **RTL support**: Ready for right-to-left languages with minimal changes

## Troubleshooting

### Translations not appearing

Ensure the i18n config is imported at the top of your app:

```typescript
import '@/i18n/config';
```

### Currency context errors

Ensure `CurrencyProvider` wraps your component tree in `App.tsx`

### Stale exchange rates

Exchange rates are refreshed hourly. Manually trigger:

```typescript
const { currency, rates } = useCurrency();
// Rates are automatically updated
```

## Files Structure

```
src/
├── i18n/
│   ├── config.ts                    # i18n configuration
│   └── locales/
│       ├── en.json                  # English translations
│       ├── fr.json                  # French translations
│       └── es.json                  # Spanish translations
├── lib/
│   └── currency.ts                  # Currency utilities
├── contexts/
│   └── CurrencyContext.tsx          # Currency provider
├── components/
│   ├── LanguageSelector.tsx         # Language switcher
│   ├── CurrencySelector.tsx         # Currency switcher
│   └── I18nExample.tsx              # Example component
├── hooks/
│   └── useI18nCurrency.ts           # Combined hook
└── App.tsx                          # Updated with providers
```

## Integration with Existing Features

The i18n system integrates seamlessly with:

- **Cart Context**: Format prices in current currency
- **Checkout**: Multi-language checkout flow
- **Admin Dashboard**: Localized admin interface
- **Email Templates**: Support for multiple language emails
- **Analytics**: Track language and currency preferences

## Future Enhancements

1. Content Management System (CMS) integration for dynamic translations
2. Real-time exchange rate API integration
3. Right-to-left (RTL) language support
4. Pluralization and gender-specific translations
5. Date and number formatting per locale
6. Language-specific product recommendations

## Support

For issues or questions regarding the i18n system, refer to:

- [i18next Documentation](https://www.i18next.com/)
- [react-i18next Documentation](https://react.i18next.com/)
- [Project Documentation](./README.md)
