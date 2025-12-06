/**
 * Admin Customers Page
 * View and manage customer information
 */
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Search, Mail, Phone, MapPin, ArrowRight } from 'lucide-react';

// Mock customers data
const mockCustomers = [
  {
    id: '1',
    name: 'John Doe',
    email: 'john@example.com',
    phone: '(555) 123-4567',
    location: 'New York, NY',
    orders: 5,
    totalSpent: 1248.50,
    joinDate: '2024-05-15',
  },
  {
    id: '2',
    name: 'Jane Smith',
    email: 'jane@example.com',
    phone: '(555) 234-5678',
    location: 'Los Angeles, CA',
    orders: 8,
    totalSpent: 2156.75,
    joinDate: '2024-03-22',
  },
  {
    id: '3',
    name: 'Mike Johnson',
    email: 'mike@example.com',
    phone: '(555) 345-6789',
    location: 'Chicago, IL',
    orders: 3,
    totalSpent: 412.00,
    joinDate: '2024-07-10',
  },
  {
    id: '4',
    name: 'Sarah Williams',
    email: 'sarah@example.com',
    phone: '(555) 456-7890',
    location: 'Houston, TX',
    orders: 12,
    totalSpent: 3298.25,
    joinDate: '2024-01-05',
  },
  {
    id: '5',
    name: 'Tom Brown',
    email: 'tom@example.com',
    phone: '(555) 567-8901',
    location: 'Phoenix, AZ',
    orders: 2,
    totalSpent: 325.00,
    joinDate: '2024-09-18',
  },
];

export function AdminCustomers() {
  const [searchQuery, setSearchQuery] = useState('');

  const filteredCustomers = mockCustomers.filter(
    (customer) =>
      customer.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      customer.email.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-display text-lii-bg">Customers</h1>
        <p className="text-lii-ash">Manage your customer base</p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="bg-white border-lii-cloud">
          <CardContent className="p-6">
            <p className="text-sm text-lii-ash mb-2">Total Customers</p>
            <p className="text-3xl font-bold text-lii-bg">
              {mockCustomers.length}
            </p>
            <p className="text-xs text-lii-ash mt-2">Active customers</p>
          </CardContent>
        </Card>

        <Card className="bg-white border-lii-cloud">
          <CardContent className="p-6">
            <p className="text-sm text-lii-ash mb-2">Total Revenue</p>
            <p className="text-3xl font-bold text-lii-gold">
              ${mockCustomers
                .reduce((sum, c) => sum + c.totalSpent, 0)
                .toLocaleString()}
            </p>
            <p className="text-xs text-lii-ash mt-2">From all customers</p>
          </CardContent>
        </Card>

        <Card className="bg-white border-lii-cloud">
          <CardContent className="p-6">
            <p className="text-sm text-lii-ash mb-2">Average Spend</p>
            <p className="text-3xl font-bold text-lii-bg">
              ${(
                mockCustomers.reduce((sum, c) => sum + c.totalSpent, 0) /
                mockCustomers.length
              ).toFixed(2)}
            </p>
            <p className="text-xs text-lii-ash mt-2">Per customer</p>
          </CardContent>
        </Card>
      </div>

      {/* Search */}
      <div className="bg-white rounded-lg border border-lii-cloud p-4">
        <div className="relative">
          <Search className="absolute left-3 top-3 text-lii-ash" size={18} />
          <Input
            placeholder="Search by name or email..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>
      </div>

      {/* Customers Table */}
      <Card className="bg-white border-lii-cloud">
        <CardHeader>
          <CardTitle className="text-lii-bg">
            Customers ({filteredCustomers.length})
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {filteredCustomers.map((customer) => (
              <div
                key={customer.id}
                className="p-4 border border-lii-cloud/20 rounded-lg hover:bg-lii-cloud/5 transition-colors"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <p className="font-semibold text-lii-bg mb-3">{customer.name}</p>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div className="flex items-center gap-2 text-lii-ash">
                        <Mail size={16} />
                        {customer.email}
                      </div>
                      <div className="flex items-center gap-2 text-lii-ash">
                        <Phone size={16} />
                        {customer.phone}
                      </div>
                      <div className="flex items-center gap-2 text-lii-ash">
                        <MapPin size={16} />
                        {customer.location}
                      </div>
                      <div className="text-lii-ash">
                        Joined {customer.joinDate}
                      </div>
                    </div>
                  </div>
                  <div className="text-right ml-4">
                    <p className="font-semibold text-lii-bg mb-2">
                      {customer.orders} orders
                    </p>
                    <p className="text-lii-gold font-bold mb-3">
                      ${customer.totalSpent.toFixed(2)}
                    </p>
                    <Button variant="ghost" size="sm" className="flex items-center gap-2">
                      View <ArrowRight size={14} />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default AdminCustomers;
