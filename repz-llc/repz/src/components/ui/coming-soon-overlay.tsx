import React from 'react';
import { cn } from '@/lib/utils';
import { Clock, Sparkles } from 'lucide-react';

interface ComingSoonOverlayProps {
  children: React.ReactNode;
  disabled?: boolean;
  message?: string;
  className?: string;
}

export const ComingSoonOverlay: React.FC<ComingSoonOverlayProps> = ({
  children,
  disabled = true,
  message = "New, modern dashboards launching soon for monthly plans.",
  className = "",
}) => {
  if (!disabled) {
    return <>{children}</>;
  }

  return (
    <div className={cn("relative", className)}>
      {/* Dimmed content */}
      <div className="opacity-40 pointer-events-none">
        {children}
      </div>
      
      {/* Coming Soon overlay */}
      <div className="absolute inset-0 flex items-center justify-center backdrop-blur-sm rounded-xl animate-fade-in">
        <div className="relative bg-black/60 backdrop-blur-md border border-white/20 rounded-xl p-6 max-w-sm mx-4 text-center shadow-2xl">
          {/* Decorative elements with hover animation */}
          <div className="absolute -top-2 -right-2 w-4 h-4 bg-gradient-to-r from-orange-500 to-amber-500 rounded-full animate-pulse hover-scale" />
          <div className="absolute -bottom-2 -left-2 w-3 h-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full animate-pulse hover-scale" style={{ animationDelay: '0.5s' }} />
          
          {/* Icon */}
          <div className="flex justify-center mb-4">
            <div className="relative">
              <Clock className="w-8 h-8 text-orange-400 hover-scale" />
              <Sparkles className="w-4 h-4 text-amber-300 absolute -top-1 -right-1 animate-pulse" />
            </div>
          </div>
          
          {/* Subtitle */}
          <p className="text-gray-300 text-sm leading-relaxed">
            {message}
          </p>
          
          {/* Optional decorative glow */}
          <div className="absolute inset-0 bg-gradient-to-r from-orange-500/10 to-amber-500/10 rounded-xl -z-10 blur-xl" />
        </div>
      </div>
    </div>
  );
};