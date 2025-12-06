import React, { useState } from 'react';
import { ShippingAddress } from '@/types/order';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Trash2, Plus, Edit } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogTrigger,
} from '@/components/ui/dialog';

interface AddressBookProps {
  userId: string;
  addresses?: ShippingAddress[];
  onAddressSelect?: (address: ShippingAddress) => void;
}

/**
 * AddressBook component manages a user's saved shipping addresses
 *
 * Displays saved addresses in a card list with options to add new, edit, and delete addresses.
 * Uses a modal dialog for address entry forms. Allows users to select addresses for checkout.
 * Address data stored in component state (client-side).
 *
 * @component
 * @param {AddressBookProps} props - Component props
 * @param {string} props.userId - User ID for tracking address ownership
 * @param {ShippingAddress[]} [props.addresses] - Initial array of saved addresses
 * @param {Function} [props.onAddressSelect] - Callback fired when address is selected
 *
 * @example
 * <AddressBook
 *   userId="user123"
 *   addresses={savedAddresses}
 *   onAddressSelect={(address) => handleAddressSelect(address)}
 * />
 *
 * @remarks
 * - Add/edit/delete operations stored in component state only
 * - Modal dialog for address form entry
 * - Full address field validation required
 * - Displays trash icon for delete, edit icon for modify
 */
export const AddressBook: React.FC<AddressBookProps> = ({
  userId,
  addresses: initialAddresses = [],
  onAddressSelect,
}) => {
  const [addresses, setAddresses] = useState<ShippingAddress[]>(initialAddresses);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [editingAddress, setEditingAddress] = useState<ShippingAddress | null>(null);
  const [formData, setFormData] = useState<ShippingAddress>({
    firstName: '',
    lastName: '',
    email: '',
    phone: '',
    address: '',
    city: '',
    state: '',
    zipCode: '',
    country: 'United States',
  });

  const handleSave = () => {
    if (editingAddress) {
      // Update existing address
      setAddresses(addresses.map(a => (a === editingAddress ? formData : a)));
    } else {
      // Add new address
      setAddresses([...addresses, formData]);
    }
    setIsDialogOpen(false);
    setEditingAddress(null);
    setFormData({
      firstName: '',
      lastName: '',
      email: '',
      phone: '',
      address: '',
      city: '',
      state: '',
      zipCode: '',
      country: 'United States',
    });
  };

  const handleEdit = (address: ShippingAddress) => {
    setEditingAddress(address);
    setFormData(address);
    setIsDialogOpen(true);
  };

  const handleDelete = (address: ShippingAddress) => {
    setAddresses(addresses.filter(a => a !== address));
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-display text-lii-cloud">Saved Addresses</h3>
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogTrigger asChild>
            <Button variant="outline" size="sm">
              <Plus className="w-4 h-4 mr-2" />
              Add Address
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-lii-ink border-lii-gold/10 max-w-2xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle className="text-lii-cloud">
                {editingAddress ? 'Edit Address' : 'Add New Address'}
              </DialogTitle>
              <DialogDescription className="text-lii-ash">Save shipping details used during checkout.</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 mt-4">
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="firstName" className="text-lii-ash">
                    First Name *
                  </Label>
                  <Input
                    id="firstName"
                    value={formData.firstName}
                    onChange={e => setFormData({ ...formData, firstName: e.target.value })}
                    className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud"
                  />
                </div>
                <div>
                  <Label htmlFor="lastName" className="text-lii-ash">
                    Last Name *
                  </Label>
                  <Input
                    id="lastName"
                    value={formData.lastName}
                    onChange={e => setFormData({ ...formData, lastName: e.target.value })}
                    className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud"
                  />
                </div>
              </div>
              <div>
                <Label htmlFor="address" className="text-lii-ash">
                  Street Address *
                </Label>
                <Input
                  id="address"
                  value={formData.address}
                  onChange={e => setFormData({ ...formData, address: e.target.value })}
                  className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud"
                />
              </div>
              <div className="grid md:grid-cols-3 gap-4">
                <div>
                  <Label htmlFor="city" className="text-lii-ash">
                    City *
                  </Label>
                  <Input
                    id="city"
                    value={formData.city}
                    onChange={e => setFormData({ ...formData, city: e.target.value })}
                    className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud"
                  />
                </div>
                <div>
                  <Label htmlFor="state" className="text-lii-ash">
                    State *
                  </Label>
                  <Input
                    id="state"
                    value={formData.state}
                    onChange={e => setFormData({ ...formData, state: e.target.value })}
                    className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud"
                  />
                </div>
                <div>
                  <Label htmlFor="zipCode" className="text-lii-ash">
                    ZIP Code *
                  </Label>
                  <Input
                    id="zipCode"
                    value={formData.zipCode}
                    onChange={e => setFormData({ ...formData, zipCode: e.target.value })}
                    className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud"
                  />
                </div>
              </div>
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={() => setIsDialogOpen(false)}>
                  Cancel
                </Button>
                <Button variant="primary" onClick={handleSave}>
                  Save Address
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {addresses.length === 0 ? (
        <p className="text-lii-ash font-ui text-sm text-center py-8">
          No saved addresses. Add one to get started.
        </p>
      ) : (
        <div className="grid gap-4">
          {addresses.map((address, index) => (
            <Card key={index} className="bg-lii-ink border-lii-gold/10">
              <CardContent className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <p className="text-lii-cloud font-ui font-medium">
                      {address.firstName} {address.lastName}
                    </p>
                    <p className="text-lii-ash font-ui text-sm mt-1">
                      {address.address}, {address.city}, {address.state} {address.zipCode}
                    </p>
                    <p className="text-lii-ash font-ui text-sm">{address.country}</p>
                  </div>
                  <div className="flex gap-2">
                    {onAddressSelect && (
                      <Button variant="outline" size="sm" onClick={() => onAddressSelect(address)}>
                        Use
                      </Button>
                    )}
                    <Button variant="ghost" size="icon" onClick={() => handleEdit(address)}>
                      <Edit className="w-4 h-4" />
                    </Button>
                    <Button variant="ghost" size="icon" onClick={() => handleDelete(address)}>
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default AddressBook;
