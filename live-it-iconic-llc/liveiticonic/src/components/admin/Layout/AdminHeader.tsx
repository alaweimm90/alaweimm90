/**
 * Admin Header Component
 * Top navigation bar with user menu and system info
 */
import { Bell, User, LogOut } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useNavigate } from 'react-router-dom';

export function AdminHeader() {
  const navigate = useNavigate();

  const handleLogout = () => {
    // Clear auth state and redirect
    localStorage.removeItem('adminToken');
    navigate('/');
  };

  return (
    <header className="bg-white border-b border-lii-cloud/20 px-6 py-4 flex items-center justify-between">
      <div className="flex-1">
        <h2 className="text-lg font-semibold text-lii-bg">
          Welcome back, Admin
        </h2>
        <p className="text-sm text-lii-ash">
          {new Date().toLocaleDateString('en-US', {
            weekday: 'long',
            month: 'long',
            day: 'numeric',
          })}
        </p>
      </div>

      <div className="flex items-center gap-4">
        <Button
          variant="ghost"
          size="sm"
          className="relative"
          title="Notifications"
        >
          <Bell size={20} />
          <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full"></span>
        </Button>

        <Button
          variant="ghost"
          size="sm"
          className="flex items-center gap-2"
          title="User Profile"
        >
          <User size={20} />
          <span className="hidden sm:inline text-sm font-medium text-lii-bg">
            Profile
          </span>
        </Button>

        <Button
          variant="ghost"
          size="sm"
          onClick={handleLogout}
          className="flex items-center gap-2"
          title="Logout"
        >
          <LogOut size={20} />
          <span className="hidden sm:inline text-sm font-medium text-lii-bg">
            Logout
          </span>
        </Button>
      </div>
    </header>
  );
}
