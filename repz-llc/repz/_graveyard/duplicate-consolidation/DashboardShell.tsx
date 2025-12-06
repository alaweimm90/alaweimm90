// Dashboard Shell - Centralized Layout with Role-Based Theming
// Industry standard: shared layout, role-specific theme injection

import React from 'react';
import { ThemeProvider, useTheme } from '@/theme/ThemeProvider';
import type { UserRole, TierType } from '@/types/business';

interface DashboardShellProps {
  children: React.ReactNode;
  role: UserRole;
  tier?: TierType;
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
}

function DashboardContent({ children, title, subtitle, actions }: Omit<DashboardShellProps, 'role' | 'tier'>) {
  const { role, tier } = useTheme();

  return (
    <div className="min-h-screen bg-background">
      {/* Dashboard Header */}
      <header className="bg-surface-elevated border-b border-white/10 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h1 className="text-2xl md:text-3xl font-bold text-white font-heading">
                {title}
              </h1>
              
              {subtitle && (
                <p className="text-muted-foreground mt-1">
                  {subtitle}
                </p>
              )}
              
              {/* Role & Tier Indicator */}
              <div className="flex items-center gap-2 mt-2">
                <span className={`
                  px-2 py-1 rounded text-xs font-medium text-white
                  ${role === 'coach' ? 'bg-repz-primary' : `bg-tier-${tier || 'core'}`}
                `}>
                  {role === 'coach' ? 'Coach View' : `${tier?.toUpperCase()} Client`}
                </span>
              </div>
            </div>
            
            {actions && (
              <div className="flex items-center gap-3">
                {actions}
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Dashboard Content */}
      <main className="p-6 space-y-6">
        <div className="max-w-7xl mx-auto">
          {children}
        </div>
      </main>
    </div>
  );
}

export function DashboardShell({ children, role, tier, ...props }: DashboardShellProps) {
  return (
    <ThemeProvider role={role} tier={tier}>
      <DashboardContent {...props}>
        {children}
      </DashboardContent>
    </ThemeProvider>
  );
}