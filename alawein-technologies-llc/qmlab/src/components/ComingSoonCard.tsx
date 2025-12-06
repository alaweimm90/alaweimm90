import React from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Lock, Bell, ArrowRight } from 'lucide-react';
import { logger } from '@/lib/logger';

interface ComingSoonCardProps {
  title: string;
  description: string;
  features: string[];
  eta?: string;
  priority?: 'high' | 'medium' | 'low';
  preview?: React.ReactNode;
}

export const ComingSoonCard: React.FC<ComingSoonCardProps> = ({
  title,
  description,
  features,
  eta = "Q2 2024",
  priority = 'medium',
  preview
}) => {
  const priorityColors = {
    high: 'border-blue-400/40 bg-gradient-to-br from-blue-500/15 via-slate-900/50 to-slate-900/30',
    medium: 'border-purple-400/40 bg-gradient-to-br from-purple-500/15 via-slate-900/50 to-slate-900/30', 
    low: 'border-slate-400/40 bg-gradient-to-br from-slate-500/15 via-slate-900/50 to-slate-900/30'
  };

  const priorityBadges = {
    high: 'bg-blue-400/20 text-blue-400 border-blue-400/50',
    medium: 'bg-purple-400/20 text-purple-400 border-purple-400/50',
    low: 'bg-slate-400/20 text-slate-400 border-slate-400/50'
  };

  return (
    <Card className={`relative rounded-3xl border backdrop-blur-sm shadow-2xl transition-all duration-500 hover:scale-[1.02] hover:shadow-3xl ${priorityColors[priority]}`}>
      {/* Lock overlay with quantum effect */}
      <div className="absolute top-4 right-4 w-10 h-10 bg-slate-800/80 rounded-full flex items-center justify-center backdrop-blur-sm border border-slate-600/50 hover:border-slate-500/70 transition-all duration-300">
        <Lock className="w-5 h-5 text-slate-400 hover:text-slate-300 transition-colors" />
      </div>

      {/* Preview area */}
      {preview && (
        <div className="relative h-32 rounded-t-2xl overflow-hidden bg-surface-2/30 border-b border-border">
          <div className="absolute inset-0 opacity-30">
            {preview}
          </div>
          <div className="absolute inset-0 bg-gradient-to-t from-surface-1/80 to-transparent"></div>
          <div className="absolute bottom-2 left-4">
            <Badge className={`text-xs ${priorityBadges[priority]}`}>
              Preview
            </Badge>
          </div>
        </div>
      )}

      <div className="p-8 space-y-6">
        {/* Header */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-slate-200 to-slate-400">{title}</h3>
            <Badge variant="outline" className="text-xs">
              {eta}
            </Badge>
          </div>
          <p className="text-slate-400 leading-relaxed">{description}</p>
        </div>

        {/* Features list */}
        <div className="space-y-2">
          <h4 className="text-base font-semibold text-slate-300 mb-3">Planned Features:</h4>
          <ul className="space-y-1">
            {features.map((feature, index) => (
              <li key={index} className="flex items-center gap-3 text-slate-400 hover:text-slate-300 transition-colors">
                <div className={`w-2 h-2 rounded-full ${priority === 'high' ? 'bg-blue-400' : priority === 'medium' ? 'bg-purple-400' : 'bg-slate-400'} animate-pulse`}></div>
                <span>{feature}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Action */}
        <div className="pt-2 border-t border-border">
          <Button 
            variant="outline" 
            size="sm" 
            className="w-full gap-2"
            onClick={() => {
              // TODO: Add to wishlist or notification system
              logger.info(`User interested in: ${title}`);
            }}
          >
            <Bell className="w-4 h-4" />
            Notify Me
            <ArrowRight className="w-4 h-4 ml-auto" />
          </Button>
        </div>
      </div>
    </Card>
  );
};