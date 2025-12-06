import React from 'react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { PlayIcon, PauseIcon, RotateCcwIcon, SettingsIcon, InfoIcon } from 'lucide-react';
import { useResponsive } from '@/hooks/use-responsive';
import { AccessibleSlider, AccessibleControlButton } from '@/components/AccessibilityEnhancedComponents';
import { cn } from '@/lib/utils';

interface ControlGroup {
  title: string;
  description?: string;
  variant?: 'primary' | 'secondary' | 'physics';
  children: React.ReactNode;
}

interface SimulationControlsProps {
  isRunning: boolean;
  onToggle: () => void;
  onReset: () => void;
  onStep?: () => void;
  disabled?: boolean;
  variant?: 'quantum' | 'statistical' | 'fields' | 'default';
}

interface PhysicsSliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  description?: string;
  variant?: 'quantum' | 'statistical' | 'fields' | 'energy';
  formatValue?: (value: number) => string;
}

interface PhysicsSelectProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: Array<{ value: string; label: string; description?: string }>;
  description?: string;
  variant?: 'quantum' | 'statistical' | 'fields';
}

export const ControlGroup: React.FC<ControlGroup> = ({ 
  title, 
  description, 
  variant = 'secondary',
  children 
}) => {
  const { isMobile } = useResponsive();
  
  const variantStyles = {
    primary: 'border-accentPhysics/20 bg-surfaceElevated',
    secondary: 'border-muted/20 bg-card',
    physics: 'border-accentQuantum/20 bg-surfaceGlass backdrop-blur-sm'
  };

  return (
    <Card className={cn(
      'transition-all duration-300',
      variantStyles[variant],
      isMobile && 'text-sm'
    )}>
      <CardHeader className={cn('pb-3', isMobile && 'pb-2')}>
        <CardTitle className={cn(
          'text-base font-medium text-textPrimary',
          isMobile && 'text-sm'
        )}>
          {title}
        </CardTitle>
        {description && (
          <p className={cn(
            'text-xs text-textSecondary mt-1',
            isMobile && 'text-xs'
          )}>
            {description}
          </p>
        )}
      </CardHeader>
      <CardContent className={cn('space-y-4', isMobile && 'space-y-3')}>
        {children}
      </CardContent>
    </Card>
  );
};

export const SimulationControls: React.FC<SimulationControlsProps> = ({
  isRunning,
  onToggle,
  onReset,
  onStep,
  disabled = false,
  variant = 'default'
}) => {
  const { isMobile } = useResponsive();
  
  const variantConfig = {
    quantum: { color: 'quantum', icon: 'quantum' },
    statistical: { color: 'statistical', icon: 'statistical' },
    fields: { color: 'fields', icon: 'fields' },
    default: { color: 'physics', icon: 'default' }
  };
  
  const config = variantConfig[variant];
  
  return (
    <div className={cn(
      'flex items-center gap-2',
      isMobile && 'flex-wrap'
    )}>
      <AccessibleControlButton
        isActive={isRunning}
        onToggle={onToggle}
        activeLabel="Pause Simulation"
        inactiveLabel="Start Simulation"
        activeIcon={<PauseIcon className="w-4 h-4" />}
        inactiveIcon={<PlayIcon className="w-4 h-4" />}
        disabled={disabled}
        variant="default"
        size={isMobile ? 'sm' : 'default'}
      />
      
      {onStep && (
        <Button
          variant="outline"
          size={isMobile ? 'sm' : 'default'}
          onClick={onStep}
          disabled={disabled || isRunning}
          aria-label="Step simulation forward"
        >
          Step
        </Button>
      )}
      
      <Button
        variant="outline"
        size={isMobile ? 'sm' : 'default'}
        onClick={onReset}
        disabled={disabled}
        aria-label="Reset simulation"
        className="gap-1"
      >
        <RotateCcwIcon className="w-4 h-4" />
        {!isMobile && 'Reset'}
      </Button>
      
      <Badge variant="outline" className="ml-auto">
        {isRunning ? 'Running' : 'Paused'}
      </Badge>
    </div>
  );
};

export const PhysicsSlider: React.FC<PhysicsSliderProps> = ({
  label,
  value,
  onChange,
  min,
  max,
  step = 0.01,
  unit = '',
  description,
  variant = 'default',
  formatValue
}) => {
  const displayValue = formatValue ? formatValue(value) : value.toFixed(2);
  
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="text-sm font-medium text-textPrimary">
          {label}
        </Label>
        <span className="text-sm font-mono text-textSecondary">
          {displayValue}{unit}
        </span>
      </div>
      {description && (
        <p className="text-xs text-textSecondary">{description}</p>
      )}
      <Slider
        value={[value]}
        onValueChange={([val]) => onChange(val)}
        min={min}
        max={max}
        step={step}
        className={cn(
          'w-full',
          variant === 'quantum' && 'accent-accentQuantum',
          variant === 'statistical' && 'accent-accentStatistical', 
          variant === 'fields' && 'accent-accentField',
          variant === 'energy' && 'accent-accentEnergy'
        )}
        aria-label={label}
      />
    </div>
  );
};

export const PhysicsSelect: React.FC<PhysicsSelectProps> = ({
  label,
  value,
  onChange,
  options,
  description,
  variant = 'default'
}) => {
  const { isMobile } = useResponsive();
  
  return (
    <div className="space-y-2">
      <Label className="text-sm font-medium text-textPrimary">
        {label}
      </Label>
      {description && (
        <p className="text-xs text-textSecondary">
          {description}
        </p>
      )}
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className={cn(
          'w-full transition-colors',
          isMobile && 'text-sm'
        )}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem 
              key={option.value} 
              value={option.value}
              className={isMobile ? 'text-sm' : ''}
            >
              <div>
                <div>{option.label}</div>
                {option.description && (
                  <div className="text-xs text-textSecondary">
                    {option.description}
                  </div>
                )}
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
};

// Physics parameter toggle switch
interface PhysicsToggleProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  description?: string;
  variant?: 'quantum' | 'statistical' | 'fields';
}

export const PhysicsToggle: React.FC<PhysicsToggleProps> = ({
  label,
  checked,
  onChange,
  description,
  variant = 'default'
}) => {
  return (
    <div className="flex items-center justify-between space-x-3">
      <div className="space-y-1">
        <Label className="text-sm font-medium text-textPrimary">
          {label}
        </Label>
        {description && (
          <p className="text-xs text-textSecondary">
            {description}
          </p>
        )}
      </div>
      <Switch
        checked={checked}
        onCheckedChange={onChange}
        className={cn(
          'transition-colors',
          variant === 'quantum' && 'data-[state=checked]:bg-accentQuantum',
          variant === 'statistical' && 'data-[state=checked]:bg-accentStatistical',
          variant === 'fields' && 'data-[state=checked]:bg-accentField'
        )}
        aria-label={label}
      />
    </div>
  );
};

// Control panel layout component
interface ControlPanelProps {
  title: string;
  children: React.ReactNode;
  variant?: 'quantum' | 'statistical' | 'fields' | 'default';
  collapsible?: boolean;
  defaultExpanded?: boolean;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  title,
  children,
  variant = 'default',
  collapsible = false,
  defaultExpanded = true
}) => {
  const [isExpanded, setIsExpanded] = React.useState(defaultExpanded);
  const { isMobile } = useResponsive();
  
  const variantColors = {
    quantum: 'border-accentQuantum/20 bg-surfaceGlass',
    statistical: 'border-accentStatistical/20 bg-surfaceGlass', 
    fields: 'border-accentField/20 bg-surfaceGlass',
    default: 'border-muted/20 bg-card'
  };

  return (
    <Card className={cn(
      'transition-all duration-300',
      variantColors[variant],
      collapsible && !isExpanded && 'hover:shadow-lg'
    )}>
      <CardHeader 
        className={cn(
          'pb-4',
          collapsible && 'cursor-pointer hover:bg-surfaceMuted/50 transition-colors'
        )}
        onClick={collapsible ? () => setIsExpanded(!isExpanded) : undefined}
      >
        <div className="flex items-center justify-between">
          <CardTitle className={cn(
            'text-lg font-semibold text-textPrimary',
            isMobile && 'text-base'
          )}>
            {title}
          </CardTitle>
          {collapsible && (
            <Button
              variant="ghost"
              size="sm"
              aria-label={isExpanded ? 'Collapse panel' : 'Expand panel'}
            >
              <SettingsIcon className={cn(
                'w-4 h-4 transition-transform',
                isExpanded && 'rotate-180'
              )} />
            </Button>
          )}
        </div>
      </CardHeader>
      
      {isExpanded && (
        <CardContent className="space-y-4">
          {children}
        </CardContent>
      )}
    </Card>
  );
};