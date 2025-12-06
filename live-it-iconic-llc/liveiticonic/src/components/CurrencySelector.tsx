import { useCurrency } from '@/contexts/CurrencyContext';
import { currencies, CurrencyCode, getAvailableCurrencies } from '@/lib/currency';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

/**
 * Currency Selector Component
 * Allows users to switch between available currencies
 */
export function CurrencySelector() {
  const { currency, setCurrency } = useCurrency();
  const availableCurrencies = getAvailableCurrencies();

  return (
    <Select
      value={currency}
      onValueChange={(value) => setCurrency(value as CurrencyCode)}
    >
      <SelectTrigger
        className="w-32"
        aria-label="Select currency"
      >
        <SelectValue placeholder="Select currency" />
      </SelectTrigger>
      <SelectContent>
        {availableCurrencies.map((code) => {
          const config = currencies[code];
          return (
            <SelectItem
              key={code}
              value={code}
              aria-label={`${code} - ${config.symbol}`}
            >
              <span className="font-medium">{config.symbol}</span>
              <span className="ml-2 text-muted-foreground">{code}</span>
            </SelectItem>
          );
        })}
      </SelectContent>
    </Select>
  );
}

export default CurrencySelector;
