import { useTranslation } from 'react-i18next';
import { useCurrency } from '@/contexts/CurrencyContext';
import { useI18nCurrency } from '@/hooks/useI18nCurrency';
import { CurrencyCode } from '@/lib/currency';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

/**
 * Example Component for i18n and Currency Usage
 * Demonstrates all major features of the internationalization system
 */
export function I18nExample() {
  const { t, i18n } = useTranslation();
  const { currency, rates } = useCurrency();
  const { format, convertAndFormat, getRate } = useI18nCurrency();

  // Example price
  const basePrice = 99.99;
  const baseUsdPrice = 150.0;

  return (
    <div className="space-y-6">
      {/* Language Information */}
      <Card>
        <CardHeader>
          <CardTitle>{t('common.loading')}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <p className="text-sm text-muted-foreground">Current Language</p>
            <p className="text-lg font-semibold">{i18n.language.toUpperCase()}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Fallback Language</p>
            <p className="text-lg font-semibold">{i18n.options.fallbackLng}</p>
          </div>
        </CardContent>
      </Card>

      {/* Navigation Translations */}
      <Card>
        <CardHeader>
          <CardTitle>Navigation Translations</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-muted-foreground">HOME</p>
              <p className="font-medium">{t('navigation.home')}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">SHOP</p>
              <p className="font-medium">{t('navigation.shop')}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">ABOUT</p>
              <p className="font-medium">{t('navigation.about')}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">CONTACT</p>
              <p className="font-medium">{t('navigation.contact')}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Currency Information */}
      <Card>
        <CardHeader>
          <CardTitle>Currency Information</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <p className="text-sm text-muted-foreground">Current Currency</p>
            <p className="text-lg font-semibold">{currency}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Base Price (USD)</p>
            <p className="text-lg font-semibold">${baseUsdPrice.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">
              Formatted in {currency}
            </p>
            <p className="text-lg font-semibold">{format(baseUsdPrice)}</p>
          </div>
        </CardContent>
      </Card>

      {/* Product Example */}
      <Card>
        <CardHeader>
          <CardTitle>{t('product.addToCart')}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">{t('product.price')}</p>
            <p className="text-2xl font-bold">{format(basePrice)}</p>
          </div>
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">
              Available Stock: <span className="text-green-600">{t('product.inStock')}</span>
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Cart Example */}
      <Card>
        <CardHeader>
          <CardTitle>{t('cart.title')}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-muted-foreground">{t('cart.subtotal')}</p>
              <p className="text-lg font-semibold">{format(basePrice)}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">{t('cart.shipping')}</p>
              <p className="text-lg font-semibold">{format(9.99)}</p>
            </div>
            <div className="col-span-2">
              <p className="text-sm text-muted-foreground">{t('cart.total')}</p>
              <p className="text-2xl font-bold">
                {format(basePrice + 9.99)}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Conversion Examples */}
      <Card>
        <CardHeader>
          <CardTitle>Currency Conversion Example</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <p className="text-sm text-muted-foreground">
              $100 USD to {currency}
            </p>
            <p className="text-lg font-semibold">
              {convertAndFormat(100, 'USD', currency)}
            </p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">
              Exchange Rate (USD to {currency})
            </p>
            <p className="text-lg font-semibold">
              {getRate('USD', currency).toFixed(4)}
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Hero Section Translation */}
      <Card>
        <CardHeader>
          <CardTitle>{t('hero.title')}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <p className="text-lg font-semibold">{t('hero.tagline')}</p>
            <p className="text-sm text-muted-foreground mt-2">
              {t('hero.subtitle')}
            </p>
            <button className="mt-4 px-4 py-2 bg-primary text-primary-foreground rounded">
              {t('hero.cta')}
            </button>
          </div>
        </CardContent>
      </Card>

      {/* Error Messages Example */}
      <Card>
        <CardHeader>
          <CardTitle>Error Messages</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <p className="text-sm">{t('errors.invalidEmail')}</p>
          <p className="text-sm">{t('errors.passwordTooShort')}</p>
          <p className="text-sm">{t('errors.fieldRequired')}</p>
        </CardContent>
      </Card>
    </div>
  );
}

export default I18nExample;
