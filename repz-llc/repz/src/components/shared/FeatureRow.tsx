import React from 'react';
import { Check, X, Clock, Users, Brain, Zap, FlaskConical, Shield, MessageCircle, Dumbbell } from 'lucide-react';

interface FeatureRowProps {
  icon: string;
  label: string;
  enabled: boolean;
  value?: string | boolean | number;
  className?: string;
}

const iconMap = {
  dashboard: Zap,
  qa: MessageCircle,
  clock: Clock,
  checkin: Check,
  form: Dumbbell,
  device: Zap,
  ai: Brain,
  science: FlaskConical,
  supplements: Shield,
  peptides: FlaskConical,
  peds: Shield,
  bio: FlaskConical,
  telegram: MessageCircle,
  training: Dumbbell
};

export const FeatureRow: React.FC<FeatureRowProps> = ({
  icon,
  label,
  enabled,
  value,
  className = ""
}) => {
  const IconComponent = iconMap[icon as keyof typeof iconMap] || Check;

  const displayValue = () => {
    if (!enabled) return null;
    if (value === true || value === undefined) return null;
    if (value === 'included') return 'Included';
    if (typeof value === 'number' && icon === 'clock') return `${value}hr`;
    return String(value);
  };

  const valueDisplay = displayValue();

  return (
    <div className={`flex items-center justify-between py-2 px-1 ${className}`}>
      <div className="flex items-center gap-3">
        <div className="flex-shrink-0">
          <IconComponent className="h-4 w-4 text-gray-500" />
        </div>
        <span className="text-sm text-gray-700 font-medium">{label}</span>
      </div>
      
      <div className="flex items-center gap-2">
        {valueDisplay && (
          <span className="text-xs text-gray-600 bg-gray-100 px-2 py-1 rounded-full">
            {valueDisplay}
          </span>
        )}
        <div className="flex-shrink-0">
          {enabled ? (
            <Check className="h-4 w-4 text-green-500" />
          ) : (
            <X className="h-4 w-4 text-red-400" />
          )}
        </div>
      </div>
    </div>
  );
};