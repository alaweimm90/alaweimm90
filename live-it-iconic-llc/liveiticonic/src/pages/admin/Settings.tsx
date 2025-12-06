/**
 * Admin Settings Page
 * Application settings and configuration
 */
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Save, AlertCircle } from 'lucide-react';

export function AdminSettings() {
  const [settings, setSettings] = useState({
    storeName: 'Live It Iconic',
    storeEmail: 'admin@liveitic.com',
    storePhone: '(555) 123-4567',
    supportEmail: 'support@liveitic.com',
    address: '123 Main St, New York, NY 10001',
    description: 'Premium lifestyle merchandise for iconic living',
    currency: 'USD',
    taxRate: 8.5,
    freeShippingThreshold: 100,
  });

  const [saved, setSaved] = useState(false);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setSettings((prev) => ({
      ...prev,
      [name]: name === 'taxRate' || name === 'freeShippingThreshold' ? Number(value) : value,
    }));
    setSaved(false);
  };

  const handleSave = () => {
    console.log('Settings saved:', settings);
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-display text-lii-bg">Settings</h1>
        <p className="text-lii-ash">Configure your store settings</p>
      </div>

      {/* Success Message */}
      {saved && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-center gap-3">
          <AlertCircle className="text-green-600" size={20} />
          <p className="text-green-700">Settings saved successfully!</p>
        </div>
      )}

      {/* Store Information */}
      <Card className="bg-white border-lii-cloud">
        <CardHeader>
          <CardTitle className="text-lii-bg">Store Information</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <Label htmlFor="storeName">Store Name</Label>
              <Input
                id="storeName"
                name="storeName"
                value={settings.storeName}
                onChange={handleChange}
                className="mt-2"
              />
            </div>
            <div>
              <Label htmlFor="storeEmail">Store Email</Label>
              <Input
                id="storeEmail"
                name="storeEmail"
                type="email"
                value={settings.storeEmail}
                onChange={handleChange}
                className="mt-2"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <Label htmlFor="storePhone">Store Phone</Label>
              <Input
                id="storePhone"
                name="storePhone"
                value={settings.storePhone}
                onChange={handleChange}
                className="mt-2"
              />
            </div>
            <div>
              <Label htmlFor="supportEmail">Support Email</Label>
              <Input
                id="supportEmail"
                name="supportEmail"
                type="email"
                value={settings.supportEmail}
                onChange={handleChange}
                className="mt-2"
              />
            </div>
          </div>

          <div>
            <Label htmlFor="address">Store Address</Label>
            <Textarea
              id="address"
              name="address"
              value={settings.address}
              onChange={handleChange}
              rows={3}
              className="mt-2"
            />
          </div>

          <div>
            <Label htmlFor="description">Store Description</Label>
            <Textarea
              id="description"
              name="description"
              value={settings.description}
              onChange={handleChange}
              rows={3}
              className="mt-2"
            />
          </div>
        </CardContent>
      </Card>

      {/* Pricing & Shipping */}
      <Card className="bg-white border-lii-cloud">
        <CardHeader>
          <CardTitle className="text-lii-bg">Pricing & Shipping</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <Label htmlFor="currency">Currency</Label>
              <select
                id="currency"
                name="currency"
                value={settings.currency}
                onChange={handleChange}
                className="w-full mt-2 px-3 py-2 border border-lii-cloud rounded-lg focus:outline-none focus:ring-2 focus:ring-lii-gold"
              >
                <option value="USD">USD</option>
                <option value="EUR">EUR</option>
                <option value="GBP">GBP</option>
                <option value="CAD">CAD</option>
              </select>
            </div>
            <div>
              <Label htmlFor="taxRate">Tax Rate (%)</Label>
              <Input
                id="taxRate"
                name="taxRate"
                type="number"
                step="0.1"
                value={settings.taxRate}
                onChange={handleChange}
                className="mt-2"
              />
            </div>
            <div>
              <Label htmlFor="freeShippingThreshold">Free Shipping Threshold ($)</Label>
              <Input
                id="freeShippingThreshold"
                name="freeShippingThreshold"
                type="number"
                step="0.01"
                value={settings.freeShippingThreshold}
                onChange={handleChange}
                className="mt-2"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Notifications Settings */}
      <Card className="bg-white border-lii-cloud">
        <CardHeader>
          <CardTitle className="text-lii-bg">Notification Settings</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-3">
            {[
              { id: 'new-orders', label: 'Notify on new orders' },
              { id: 'low-stock', label: 'Notify when stock is low' },
              { id: 'customer-messages', label: 'Customer messages' },
              { id: 'reviews', label: 'New customer reviews' },
            ].map((notification) => (
              <div key={notification.id} className="flex items-center gap-3">
                <input
                  type="checkbox"
                  id={notification.id}
                  defaultChecked
                  className="w-4 h-4 rounded border-lii-cloud cursor-pointer"
                />
                <Label htmlFor={notification.id} className="cursor-pointer">
                  {notification.label}
                </Label>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Save Button */}
      <div className="flex justify-end gap-4">
        <Button variant="outline">Cancel</Button>
        <Button onClick={handleSave} className="flex items-center gap-2">
          <Save size={18} />
          Save Changes
        </Button>
      </div>
    </div>
  );
}

export default AdminSettings;
