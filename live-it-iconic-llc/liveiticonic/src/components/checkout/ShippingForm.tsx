import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { ShippingAddress } from '@/types/order';
import { Truck } from 'lucide-react';

interface ShippingFormProps {
  initialData?: Partial<ShippingAddress>;
  onNext: (data: ShippingAddress) => void;
}

/**
 * ShippingForm component collects and validates customer shipping address information
 *
 * Displays a form with fields for customer name, email, phone, and full address details
 * including street, city, state, ZIP code, and country. Validates required fields before
 * allowing progression to the payment step.
 *
 * @component
 * @param {ShippingFormProps} props - Component props
 * @param {Partial<ShippingAddress>} [props.initialData] - Pre-populated shipping data for editing
 * @param {Function} props.onNext - Callback fired when form is submitted with validated ShippingAddress data
 *
 * @example
 * <ShippingForm
 *   initialData={{ firstName: 'John', lastName: 'Doe' }}
 *   onNext={(data) => handleShippingSubmit(data)}
 * />
 */
export const ShippingForm: React.FC<ShippingFormProps> = ({ initialData, onNext }) => {
  const [formData, setFormData] = React.useState<ShippingAddress>({
    firstName: initialData?.firstName || '',
    lastName: initialData?.lastName || '',
    email: initialData?.email || '',
    phone: initialData?.phone || '',
    address: initialData?.address || '',
    city: initialData?.city || '',
    state: initialData?.state || '',
    zipCode: initialData?.zipCode || '',
    country: initialData?.country || 'United States',
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onNext(formData);
  };

  const handleChange = (field: keyof ShippingAddress, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <Card className="bg-lii-ink border-lii-gold/10">
      <CardHeader>
        <CardTitle className="flex items-center gap-3 text-lii-cloud">
          <Truck className="w-5 h-5 text-lii-gold" />
          Shipping Information
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <Label htmlFor="firstName" className="text-lii-ash">
                First Name *
              </Label>
              <Input
                id="firstName"
                required
                value={formData.firstName}
                onChange={e => handleChange('firstName', e.target.value)}
                className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud placeholder:text-lii-ash/50"
              />
            </div>
            <div>
              <Label htmlFor="lastName" className="text-lii-ash">
                Last Name *
              </Label>
              <Input
                id="lastName"
                required
                value={formData.lastName}
                onChange={e => handleChange('lastName', e.target.value)}
                className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud placeholder:text-lii-ash/50"
              />
            </div>
          </div>

          <div>
            <Label htmlFor="email" className="text-lii-ash">
              Email Address *
            </Label>
            <Input
              id="email"
              type="email"
              required
              value={formData.email}
              onChange={e => handleChange('email', e.target.value)}
              className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud placeholder:text-lii-ash/50"
            />
          </div>

          <div>
            <Label htmlFor="phone" className="text-lii-ash">
              Phone Number *
            </Label>
            <Input
              id="phone"
              type="tel"
              required
              value={formData.phone}
              onChange={e => handleChange('phone', e.target.value)}
              className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud placeholder:text-lii-ash/50"
            />
          </div>

          <div>
            <Label htmlFor="address" className="text-lii-ash">
              Street Address *
            </Label>
            <Input
              id="address"
              required
              value={formData.address}
              onChange={e => handleChange('address', e.target.value)}
              className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud placeholder:text-lii-ash/50"
            />
          </div>

          <div className="grid md:grid-cols-3 gap-4">
            <div>
              <Label htmlFor="city" className="text-lii-ash">
                City *
              </Label>
              <Input
                id="city"
                required
                value={formData.city}
                onChange={e => handleChange('city', e.target.value)}
                className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud placeholder:text-lii-ash/50"
              />
            </div>
            <div>
              <Label htmlFor="state" className="text-lii-ash">
                State *
              </Label>
              <Input
                id="state"
                required
                value={formData.state}
                onChange={e => handleChange('state', e.target.value)}
                className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud placeholder:text-lii-ash/50"
              />
            </div>
            <div>
              <Label htmlFor="zipCode" className="text-lii-ash">
                ZIP Code *
              </Label>
              <Input
                id="zipCode"
                required
                value={formData.zipCode}
                onChange={e => handleChange('zipCode', e.target.value)}
                className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud placeholder:text-lii-ash/50"
              />
            </div>
          </div>

          <div>
            <Label htmlFor="country" className="text-lii-ash">
              Country *
            </Label>
            <Input
              id="country"
              required
              value={formData.country}
              onChange={e => handleChange('country', e.target.value)}
              className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud placeholder:text-lii-ash/50"
            />
          </div>

          <Button type="submit" variant="primary" className="w-full font-ui font-medium">
            Continue to Payment
          </Button>
        </form>
      </CardContent>
    </Card>
  );
};

export default ShippingForm;
