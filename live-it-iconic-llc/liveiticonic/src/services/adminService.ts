import { Order } from '@/types/order';

interface DashboardStats {
  totalOrders: number;
  totalRevenue: number;
  totalProducts: number;
  lowStockCount: number;
  recentOrders: Order[];
}

export const adminService = {
  async getDashboardStats(): Promise<DashboardStats> {
    try {
      const response = await fetch('/api/admin/dashboard', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        return {
          totalOrders: 0,
          totalRevenue: 0,
          totalProducts: 0,
          lowStockCount: 0,
          recentOrders: [],
        };
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Failed to get dashboard stats:', error);
      return {
        totalOrders: 0,
        totalRevenue: 0,
        totalProducts: 0,
        lowStockCount: 0,
        recentOrders: [],
      };
    }
  },
};
