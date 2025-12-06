/**
 * Accessibility Enhanced Components
 * 
 * WCAG 2.2 AA compliant components for scientific interfaces with
 * proper ARIA support, keyboard navigation, and screen reader optimization.
 */

import { ReactNode, useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Play, Pause, RotateCcw, Info, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface AccessibleSliderProps {
  label: string;
  value: number[];
  onChange: (value: number[]) => void;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  description?: string;
  className?: string;
  disabled?: boolean;
}

export function AccessibleSlider({
  label,
  value,
  onChange,
  min,
  max,
  step = 0.1,
  unit = '',
  description,
  className,
  disabled = false
}: AccessibleSliderProps) {
  const sliderId = `slider-${label.toLowerCase().replace(/\s+/g, '-')}`;
  const descriptionId = description ? `${sliderId}-description` : undefined;
  
  return (
    <div className={cn("space-y-3", className)}>
      <div className="flex items-center justify-between">
        <Label 
          htmlFor={sliderId}
          className="text-sm font-medium text-textPrimary"
        >
          {label}
        </Label>
        <Badge 
          variant="outline" 
          className="text-xs font-mono bg-surfaceMuted text-textSecondary"
          aria-label={`Current value: ${value[0]}${unit}`}
        >
          {value[0].toFixed(2)}{unit}
        </Badge>
      </div>
      
      {description && (
        <p 
          id={descriptionId}
          className="text-xs text-textMuted leading-relaxed"
        >
          {description}
        </p>
      )}
      
      <Slider
        id={sliderId}
        value={value}
        onValueChange={onChange}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        className="w-full"
        aria-describedby={descriptionId}
        aria-label={`${label} slider, current value ${value[0]}${unit}, minimum ${min}, maximum ${max}`}
      />
      
      {/* Screen reader only live region for value changes */}
      <div 
        aria-live="polite" 
        aria-atomic="true"
        className="sr-only"
      >
        {label}: {value[0].toFixed(2)}{unit}
      </div>
    </div>
  );
}

interface AccessibleControlButtonProps {
  isActive: boolean;
  onToggle: () => void;
  activeLabel: string;
  inactiveLabel: string;
  activeIcon?: ReactNode;
  inactiveIcon?: ReactNode;
  disabled?: boolean;
  variant?: 'default' | 'destructive' | 'outline';
  size?: 'sm' | 'default' | 'lg';
  className?: string;
}

export function AccessibleControlButton({
  isActive,
  onToggle,
  activeLabel,
  inactiveLabel,
  activeIcon = <Pause className="w-4 h-4" />,
  inactiveIcon = <Play className="w-4 h-4" />,
  disabled = false,
  variant = 'default',
  size = 'default',
  className
}: AccessibleControlButtonProps) {
  const [announcement, setAnnouncement] = useState('');
  
  const handleToggle = () => {
    onToggle();
    setAnnouncement(isActive ? inactiveLabel : activeLabel);
    // Clear announcement after a short delay
    setTimeout(() => setAnnouncement(''), 1000);
  };
  
  return (
    <>
      <Button
        variant={variant}
        size={size}
        onClick={handleToggle}
        disabled={disabled}
        className={cn(
          "transition-all duration-300 min-w-[44px] min-h-[44px]", // WCAG touch target size
          className
        )}
        aria-label={isActive ? activeLabel : inactiveLabel}
        aria-pressed={isActive}
      >
        <span className="flex items-center gap-2">
          {isActive ? activeIcon : inactiveIcon}
          <span className="hidden sm:inline">
            {isActive ? activeLabel : inactiveLabel}
          </span>
        </span>
      </Button>
      
      {/* Announcement for screen readers */}
      {announcement && (
        <div 
          aria-live="assertive"
          aria-atomic="true"
          className="sr-only"
        >
          {announcement}
        </div>
      )}
    </>
  );
}

interface AccessibleStatusDisplayProps {
  title: string;
  value: string | number;
  unit?: string;
  status?: 'normal' | 'warning' | 'error';
  description?: string;
  className?: string;
}

export function AccessibleStatusDisplay({
  title,
  value,
  unit = '',
  status = 'normal',
  description,
  className
}: AccessibleStatusDisplayProps) {
  const statusColors = {
    normal: 'text-textPrimary border-muted/30',
    warning: 'text-yellow-600 border-yellow-300 bg-yellow-50/50',
    error: 'text-red-600 border-red-300 bg-red-50/50'
  };
  
  const statusIcons = {
    normal: null,
    warning: <AlertTriangle className="w-4 h-4 text-yellow-600" />,
    error: <AlertTriangle className="w-4 h-4 text-red-600" />
  };
  
  const statusId = `status-${title.toLowerCase().replace(/\s+/g, '-')}`;
  
  return (
    <Card className={cn(
      "transition-all duration-300",
      statusColors[status],
      className
    )}>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {statusIcons[status]}
            <Label 
              htmlFor={statusId}
              className="text-sm font-medium"
            >
              {title}
            </Label>
          </div>
          
          <div 
            id={statusId}
            className="text-lg font-mono font-bold"
            aria-label={`${title}: ${value}${unit}`}
          >
            {typeof value === 'number' ? value.toFixed(3) : value}{unit}
          </div>
        </div>
        
        {description && (
          <p className="text-xs text-textMuted mt-2 leading-relaxed">
            {description}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

interface AccessibleSimulationControlsProps {
  isRunning: boolean;
  onToggleSimulation: () => void;
  onReset: () => void;
  onStep?: () => void;
  disabled?: boolean;
  currentStep?: number;
  className?: string;
}

export function AccessibleSimulationControls({
  isRunning,
  onToggleSimulation,
  onReset,
  onStep,
  disabled = false,
  currentStep,
  className
}: AccessibleSimulationControlsProps) {
  return (
    <div 
      className={cn("flex flex-wrap gap-2", className)}
      role="toolbar"
      aria-label="Simulation controls"
    >
      <AccessibleControlButton
        isActive={isRunning}
        onToggle={onToggleSimulation}
        activeLabel="Pause simulation"
        inactiveLabel="Start simulation"
        disabled={disabled}
        variant={isRunning ? "destructive" : "default"}
      />
      
      {onStep && (
        <Button
          variant="outline"
          size="default"
          onClick={onStep}
          disabled={disabled || isRunning}
          className="min-w-[44px] min-h-[44px] transition-all duration-300"
          aria-label="Advance simulation by one step"
        >
          <span className="flex items-center gap-2">
            <span className="text-sm">⏭</span>
            <span className="hidden sm:inline">Step</span>
          </span>
        </Button>
      )}
      
      <Button
        variant="outline"
        size="default"
        onClick={onReset}
        disabled={disabled}
        className="min-w-[44px] min-h-[44px] transition-all duration-300"
        aria-label="Reset simulation to initial state"
      >
        <span className="flex items-center gap-2">
          <RotateCcw className="w-4 h-4" />
          <span className="hidden sm:inline">Reset</span>
        </span>
      </Button>
      
      {currentStep !== undefined && (
        <div 
          className="flex items-center px-3 py-2 bg-surfaceMuted rounded-md"
          aria-label={`Current simulation step: ${currentStep}`}
        >
          <span className="text-sm font-mono text-textSecondary">
            Step: {currentStep}
          </span>
        </div>
      )}
    </div>
  );
}

interface AccessibleAlertProps {
  type: 'info' | 'warning' | 'error' | 'success';
  title?: string;
  children: ReactNode;
  dismissible?: boolean;
  onDismiss?: () => void;
  className?: string;
}

export function AccessibleAlert({
  type,
  title,
  children,
  dismissible = false,
  onDismiss,
  className
}: AccessibleAlertProps) {
  const alertRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    // Focus the alert when it appears for screen readers
    if (alertRef.current) {
      alertRef.current.focus();
    }
  }, []);
  
  const alertTypes = {
    info: { variant: 'default' as const, icon: <Info className="w-4 h-4" /> },
    warning: { variant: 'destructive' as const, icon: <AlertTriangle className="w-4 h-4" /> },
    error: { variant: 'destructive' as const, icon: <AlertTriangle className="w-4 h-4" /> },
    success: { variant: 'default' as const, icon: <Info className="w-4 h-4" /> }
  };
  
  return (
    <Alert 
      ref={alertRef}
      variant={alertTypes[type].variant}
      className={cn(
        "transition-all duration-300",
        className
      )}
      role="alert"
      aria-live="assertive"
      tabIndex={-1}
    >
      <div className="flex items-start gap-3">
        {alertTypes[type].icon}
        <div className="flex-1 space-y-1">
          {title && (
            <h4 className="font-medium">{title}</h4>
          )}
          <AlertDescription className="text-sm leading-relaxed">
            {children}
          </AlertDescription>
        </div>
        
        {dismissible && onDismiss && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onDismiss}
            className="h-6 w-6 p-0"
            aria-label="Dismiss alert"
          >
            ×
          </Button>
        )}
      </div>
    </Alert>
  );
}