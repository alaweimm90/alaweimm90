import React from 'react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent } from '@/components/ui/card';

interface FiltersProps {
  filters: {
    status?: string;
    search?: string;
    dateFrom?: string;
    dateTo?: string;
  };
  onChange: (filters: Record<string, unknown>) => void;
}

export const Filters: React.FC<FiltersProps> = ({ filters, onChange }) => {
  return (
    <Card className="bg-lii-ink border-lii-gold/10 mb-6">
      <CardContent className="p-4">
        <div className="grid md:grid-cols-4 gap-4">
          <div>
            <Label htmlFor="status" className="text-lii-ash text-sm mb-2 block">
              Status
            </Label>
            <Select
              value={filters.status || 'all'}
              onValueChange={value =>
                onChange({ ...filters, status: value === 'all' ? undefined : value })
              }
            >
              <SelectTrigger className="bg-lii-charcoal/20 border-lii-gold/20">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
                <SelectItem value="confirmed">Confirmed</SelectItem>
                <SelectItem value="processing">Processing</SelectItem>
                <SelectItem value="shipped">Shipped</SelectItem>
                <SelectItem value="delivered">Delivered</SelectItem>
                <SelectItem value="cancelled">Cancelled</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label htmlFor="search" className="text-lii-ash text-sm mb-2 block">
              Search
            </Label>
            <Input
              id="search"
              placeholder="Order number, email..."
              value={filters.search || ''}
              onChange={e => onChange({ ...filters, search: e.target.value })}
              className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud"
            />
          </div>
          <div>
            <Label htmlFor="dateFrom" className="text-lii-ash text-sm mb-2 block">
              From Date
            </Label>
            <Input
              id="dateFrom"
              type="date"
              value={filters.dateFrom || ''}
              onChange={e => onChange({ ...filters, dateFrom: e.target.value })}
              className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud"
            />
          </div>
          <div>
            <Label htmlFor="dateTo" className="text-lii-ash text-sm mb-2 block">
              To Date
            </Label>
            <Input
              id="dateTo"
              type="date"
              value={filters.dateTo || ''}
              onChange={e => onChange({ ...filters, dateTo: e.target.value })}
              className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud"
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default Filters;
