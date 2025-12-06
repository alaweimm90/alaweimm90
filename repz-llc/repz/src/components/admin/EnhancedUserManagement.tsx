import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import { Input } from '@/ui/atoms/Input';
import {
  Search,
  Filter,
  MoreVertical,
  UserPlus,
  Download,
  RefreshCw,
  CheckSquare,
  Mail,
  Ban,
  Trash2,
  Edit,
  Eye,
  Calendar,
  TrendingUp,
  Clock,
  Activity
} from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator
} from '@/ui/molecules/DropdownMenu';
import { cn } from '@/lib/utils';
import { TIER_CONFIGS } from '@/constants/tiers';

interface User {
  id: string;
  name: string;
  email: string;
  tier: string;
  status: 'active' | 'inactive' | 'suspended' | 'trial';
  joinDate: string;
  lastActive: string;
  revenue: number;
  activities: number;
}

const EnhancedUserManagement: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTier, setSelectedTier] = useState<string>('all');
  const [selectedStatus, setSelectedStatus] = useState<string>('all');
  const [selectedUsers, setSelectedUsers] = useState<string[]>([]);
  const [sortBy, setSortBy] = useState<'name' | 'joinDate' | 'revenue' | 'lastActive'>('joinDate');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Mock users data
  const [users] = useState<User[]>([
    {
      id: 'USR-001',
      name: 'John Smith',
      email: 'john.smith@example.com',
      tier: 'performance',
      status: 'active',
      joinDate: '2025-09-15',
      lastActive: '2025-11-12',
      revenue: 2290,
      activities: 147
    },
    {
      id: 'USR-002',
      name: 'Sarah Johnson',
      email: 'sarah.j@example.com',
      tier: 'adaptive',
      status: 'active',
      joinDate: '2025-08-20',
      lastActive: '2025-11-11',
      revenue: 1490,
      activities: 98
    },
    {
      id: 'USR-003',
      name: 'Mike Davis',
      email: 'mike.davis@example.com',
      tier: 'core',
      status: 'trial',
      joinDate: '2025-11-10',
      lastActive: '2025-11-12',
      revenue: 0,
      activities: 5
    },
    {
      id: 'USR-004',
      name: 'Emily Brown',
      email: 'emily.brown@example.com',
      tier: 'longevity',
      status: 'active',
      joinDate: '2025-07-01',
      lastActive: '2025-11-12',
      revenue: 3490,
      activities: 256
    },
    {
      id: 'USR-005',
      name: 'Chris Wilson',
      email: 'chris.w@example.com',
      tier: 'adaptive',
      status: 'inactive',
      joinDate: '2025-10-05',
      lastActive: '2025-10-28',
      revenue: 149,
      activities: 12
    },
    {
      id: 'USR-006',
      name: 'Jessica Lee',
      email: 'jessica.lee@example.com',
      tier: 'performance',
      status: 'suspended',
      joinDate: '2025-06-12',
      lastActive: '2025-09-15',
      revenue: 2290,
      activities: 78
    }
  ]);

  // Filter and sort users
  const filteredUsers = useMemo(() => {
    const filtered = users.filter((user) => {
      const matchesSearch =
        user.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        user.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
        user.id.toLowerCase().includes(searchQuery.toLowerCase());

      const matchesTier = selectedTier === 'all' || user.tier === selectedTier;
      const matchesStatus = selectedStatus === 'all' || user.status === selectedStatus;

      return matchesSearch && matchesTier && matchesStatus;
    });

    // Sort
    filtered.sort((a, b) => {
      let compareValue = 0;
      switch (sortBy) {
        case 'name':
          compareValue = a.name.localeCompare(b.name);
          break;
        case 'joinDate':
          compareValue = new Date(a.joinDate).getTime() - new Date(b.joinDate).getTime();
          break;
        case 'revenue':
          compareValue = a.revenue - b.revenue;
          break;
        case 'lastActive':
          compareValue = new Date(a.lastActive).getTime() - new Date(b.lastActive).getTime();
          break;
      }
      return sortOrder === 'asc' ? compareValue : -compareValue;
    });

    return filtered;
  }, [users, searchQuery, selectedTier, selectedStatus, sortBy, sortOrder]);

  const handleSelectAll = () => {
    if (selectedUsers.length === filteredUsers.length) {
      setSelectedUsers([]);
    } else {
      setSelectedUsers(filteredUsers.map((u) => u.id));
    }
  };

  const handleSelectUser = (userId: string) => {
    setSelectedUsers((prev) =>
      prev.includes(userId) ? prev.filter((id) => id !== userId) : [...prev, userId]
    );
  };

  const handleBulkAction = (action: string) => {
    console.log(`Performing bulk action: ${action} on users:`, selectedUsers);
    // Implement bulk actions
    setSelectedUsers([]);
  };

  const getTierDisplayName = (tierId: string) => {
    const tierConfig = TIER_CONFIGS.find((t) => t.id === tierId);
    return tierConfig?.displayName || tierId;
  };

  const getTierColor = (tierId: string) => {
    const tierConfig = TIER_CONFIGS.find((t) => t.id === tierId);
    return tierConfig?.color || '#666';
  };

  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800';
      case 'inactive':
        return 'bg-gray-100 text-gray-800';
      case 'suspended':
        return 'bg-red-100 text-red-800';
      case 'trial':
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  const getRelativeTime = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInMs = now.getTime() - date.getTime();
    const diffInDays = Math.floor(diffInMs / (1000 * 60 * 60 * 24));

    if (diffInDays === 0) return 'Today';
    if (diffInDays === 1) return 'Yesterday';
    if (diffInDays < 7) return `${diffInDays} days ago`;
    if (diffInDays < 30) return `${Math.floor(diffInDays / 7)} weeks ago`;
    return `${Math.floor(diffInDays / 30)} months ago`;
  };

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">User Management</h1>
          <p className="text-gray-600 mt-1">
            Manage {filteredUsers.length} of {users.length} users
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button size="sm">
            <UserPlus className="h-4 w-4 mr-2" />
            Add User
          </Button>
        </div>
      </div>

      {/* Filters and Search */}
      <Card>
        <CardContent className="pt-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {/* Search */}
            <div className="md:col-span-2">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <Input
                  placeholder="Search by name, email, or ID..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>

            {/* Tier Filter */}
            <div>
              <select
                value={selectedTier}
                onChange={(e) => setSelectedTier(e.target.value)}
                className="w-full h-10 px-3 rounded-md border border-gray-300 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Tiers</option>
                {TIER_CONFIGS.map((tier) => (
                  <option key={tier.id} value={tier.id}>
                    {tier.displayName}
                  </option>
                ))}
              </select>
            </div>

            {/* Status Filter */}
            <div>
              <select
                value={selectedStatus}
                onChange={(e) => setSelectedStatus(e.target.value)}
                className="w-full h-10 px-3 rounded-md border border-gray-300 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Statuses</option>
                <option value="active">Active</option>
                <option value="inactive">Inactive</option>
                <option value="suspended">Suspended</option>
                <option value="trial">Trial</option>
              </select>
            </div>
          </div>

          {/* Bulk Actions */}
          {selectedUsers.length > 0 && (
            <div className="mt-4 flex items-center gap-3 p-3 bg-blue-50 rounded-lg">
              <CheckSquare className="h-5 w-5 text-blue-600" />
              <span className="text-sm font-medium text-blue-900">
                {selectedUsers.length} user{selectedUsers.length > 1 ? 's' : ''} selected
              </span>
              <div className="flex items-center gap-2 ml-auto">
                <Button size="sm" variant="outline" onClick={() => handleBulkAction('email')}>
                  <Mail className="h-4 w-4 mr-2" />
                  Email
                </Button>
                <Button size="sm" variant="outline" onClick={() => handleBulkAction('changeTier')}>
                  <TrendingUp className="h-4 w-4 mr-2" />
                  Change Tier
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleBulkAction('suspend')}
                  className="text-red-600 hover:text-red-700"
                >
                  <Ban className="h-4 w-4 mr-2" />
                  Suspend
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Users Table */}
      <Card>
        <CardContent className="pt-6">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-3 px-4">
                    <input
                      type="checkbox"
                      checked={selectedUsers.length === filteredUsers.length && filteredUsers.length > 0}
                      onChange={handleSelectAll}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700 cursor-pointer hover:text-gray-900"
                      onClick={() => {
                        setSortBy('name');
                        setSortOrder(sortBy === 'name' && sortOrder === 'asc' ? 'desc' : 'asc');
                      }}
                  >
                    User
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Tier</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Status</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700 cursor-pointer hover:text-gray-900"
                      onClick={() => {
                        setSortBy('joinDate');
                        setSortOrder(sortBy === 'joinDate' && sortOrder === 'asc' ? 'desc' : 'asc');
                      }}
                  >
                    Join Date
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700 cursor-pointer hover:text-gray-900"
                      onClick={() => {
                        setSortBy('lastActive');
                        setSortOrder(sortBy === 'lastActive' && sortOrder === 'asc' ? 'desc' : 'asc');
                      }}
                  >
                    Last Active
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700 cursor-pointer hover:text-gray-900"
                      onClick={() => {
                        setSortBy('revenue');
                        setSortOrder(sortBy === 'revenue' && sortOrder === 'asc' ? 'desc' : 'asc');
                      }}
                  >
                    Revenue
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Activities</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredUsers.map((user) => (
                  <tr key={user.id} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-3 px-4">
                      <input
                        type="checkbox"
                        checked={selectedUsers.includes(user.id)}
                        onChange={() => handleSelectUser(user.id)}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                    </td>
                    <td className="py-3 px-4">
                      <div>
                        <p className="text-sm font-medium text-gray-900">{user.name}</p>
                        <p className="text-xs text-gray-600">{user.email}</p>
                        <p className="text-xs text-gray-500 font-mono">{user.id}</p>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <Badge
                        className="text-xs"
                        style={{ backgroundColor: getTierColor(user.tier) + '20', color: getTierColor(user.tier) }}
                      >
                        {getTierDisplayName(user.tier)}
                      </Badge>
                    </td>
                    <td className="py-3 px-4">
                      <Badge className={cn('text-xs', getStatusBadgeClass(user.status))}>
                        {user.status.charAt(0).toUpperCase() + user.status.slice(1)}
                      </Badge>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-1 text-sm text-gray-600">
                        <Calendar className="h-3 w-3" />
                        {formatDate(user.joinDate)}
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-1 text-sm text-gray-600">
                        <Clock className="h-3 w-3" />
                        {getRelativeTime(user.lastActive)}
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <p className="text-sm font-medium text-gray-900">{formatCurrency(user.revenue)}</p>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-1 text-sm text-gray-600">
                        <Activity className="h-3 w-3" />
                        {user.activities}
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                            <MoreVertical className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem>
                            <Eye className="h-4 w-4 mr-2" />
                            View Details
                          </DropdownMenuItem>
                          <DropdownMenuItem>
                            <Edit className="h-4 w-4 mr-2" />
                            Edit User
                          </DropdownMenuItem>
                          <DropdownMenuItem>
                            <Mail className="h-4 w-4 mr-2" />
                            Send Email
                          </DropdownMenuItem>
                          <DropdownMenuItem>
                            <TrendingUp className="h-4 w-4 mr-2" />
                            Change Tier
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem className="text-red-600">
                            <Ban className="h-4 w-4 mr-2" />
                            Suspend User
                          </DropdownMenuItem>
                          <DropdownMenuItem className="text-red-600">
                            <Trash2 className="h-4 w-4 mr-2" />
                            Delete User
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {filteredUsers.length === 0 && (
            <div className="text-center py-12">
              <p className="text-gray-500">No users found matching your filters.</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default EnhancedUserManagement;
