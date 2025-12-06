/**
 * Admin Sidebar Component
 * Navigation sidebar for admin dashboard with links to all admin pages
 */
import { Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Package,
  ShoppingCart,
  Users,
  BarChart3,
  Settings,
  Activity,
} from 'lucide-react';

const navItems = [
  { icon: LayoutDashboard, label: 'Dashboard', path: '/admin' },
  { icon: Package, label: 'Products', path: '/admin/products' },
  { icon: ShoppingCart, label: 'Orders', path: '/admin/orders' },
  { icon: Users, label: 'Customers', path: '/admin/customers' },
  { icon: BarChart3, label: 'Analytics', path: '/admin/analytics' },
  { icon: Activity, label: 'Monitoring', path: '/admin/monitoring' },
  { icon: Settings, label: 'Settings', path: '/admin/settings' },
];

export function AdminSidebar() {
  const location = useLocation();

  return (
    <aside className="w-64 bg-lii-ink border-r border-lii-gold/20 flex flex-col">
      <div className="p-6 border-b border-lii-gold/20">
        <h1 className="text-2xl font-display text-lii-gold">Live It Iconic</h1>
        <p className="text-sm text-lii-ash">Admin Dashboard</p>
      </div>

      <nav className="flex-1 px-4 py-6 space-y-2">
        {navItems.map(({ icon: Icon, label, path }) => {
          const isActive = location.pathname === path || (path !== '/admin' && location.pathname.startsWith(path));
          return (
            <Link
              key={path}
              to={path}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                isActive
                  ? 'bg-lii-gold text-lii-bg font-semibold'
                  : 'text-lii-cloud hover:bg-lii-charcoal/30'
              }`}
            >
              <Icon size={20} />
              <span>{label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-lii-gold/20 bg-lii-charcoal/20">
        <p className="text-xs text-lii-ash text-center">v1.0.0</p>
      </div>
    </aside>
  );
}
