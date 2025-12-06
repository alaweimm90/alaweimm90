import React from 'react';
import { Badge } from '@/ui/atoms/Badge';
import { Activity, Heart, Zap } from 'lucide-react';

interface CoachingCue {
  id: string;
  type: 'form' | 'motivation' | 'pacing' | 'safety';
  message: string;
  priority: 'low' | 'medium' | 'high';
  timestamp: number;
}

interface CoachingCuesDisplayProps {
  cues: CoachingCue[];
  isActive: boolean;
}

export const CoachingCuesDisplay: React.FC<CoachingCuesDisplayProps> = ({ cues, isActive }) => {
  const getCueColor = (type: CoachingCue['type'], priority: CoachingCue['priority']) => {
    if (priority === 'high') return 'border-red-500 bg-red-50';
    if (type === 'form') return 'border-blue-500 bg-blue-50';
    if (type === 'motivation') return 'border-green-500 bg-green-50';
    if (type === 'pacing') return 'border-yellow-500 bg-yellow-50';
    return 'border-gray-500 bg-gray-50';
  };

  const getCueIcon = (type: CoachingCue['type']) => {
    switch (type) {
      case 'form': return Activity;
      case 'motivation': return Zap;
      case 'pacing': return Heart;
      case 'safety': return Heart;
      default: return Activity;
    }
  };

  if (!isActive || cues.length === 0) return null;

  return (
    <div className="space-y-3">
      <h4 className="font-medium">Recent Coaching Cues</h4>
      <div className="space-y-2 max-h-60 overflow-y-auto">
        {cues.map((cue) => {
          const Icon = getCueIcon(cue.type);
          return (
            <div
              key={cue.id}
              className={`p-3 rounded-lg border-l-4 ${getCueColor(cue.type, cue.priority)}`}
            >
              <div className="flex items-start gap-2">
                <Icon className="h-4 w-4 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  <p className="text-sm font-medium">{cue.message}</p>
                  <div className="flex items-center gap-2 mt-1">
                    <Badge variant="outline" className="text-xs">
                      {cue.type}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      {new Date(cue.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};