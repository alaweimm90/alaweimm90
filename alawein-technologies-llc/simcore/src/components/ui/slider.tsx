// Enhanced accessible slider component with WCAG 2.1 AA compliance
import * as React from "react"
import * as SliderPrimitive from "@radix-ui/react-slider"

import { cn } from "@/lib/utils"

interface SliderProps extends React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root> {
  label?: string
  description?: string
  valueLabel?: (value: number) => string
  showValue?: boolean
  unit?: string
}

const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  SliderProps
>(({ 
  className, 
  label, 
  description, 
  valueLabel, 
  showValue = false, 
  unit = "",
  value,
  defaultValue,
  onValueChange,
  min = 0,
  max = 100,
  step = 1,
  disabled,
  'aria-label': ariaLabel,
  'aria-labelledby': ariaLabelledBy,
  'aria-describedby': ariaDescribedBy,
  ...props 
}, ref) => {
  const sliderId = React.useId()
  const labelId = `${sliderId}-label`
  const descId = `${sliderId}-desc`
  const valueId = `${sliderId}-value`
  
  const currentValue = value || defaultValue || [min]
  const displayValue = Array.isArray(currentValue) ? currentValue[0] : currentValue

  // Enhanced accessibility attributes
  const accessibilityProps = {
    'aria-label': ariaLabel || label,
    'aria-labelledby': ariaLabelledBy || (label ? labelId : undefined),
    'aria-describedby': [
      ariaDescribedBy,
      description ? descId : undefined,
      showValue ? valueId : undefined
    ].filter(Boolean).join(' ') || undefined,
    'aria-valuemin': min,
    'aria-valuemax': max,
    'aria-valuenow': displayValue,
    'aria-valuetext': valueLabel ? valueLabel(displayValue) : `${displayValue}${unit}`,
    role: 'slider'
  }

  return (
    <div className="space-y-2">
      {label && (
        <label 
          id={labelId}
          htmlFor={sliderId}
          className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
        >
          {label}
        </label>
      )}
      
      {description && (
        <p id={descId} className="text-sm text-muted-foreground">
          {description}
        </p>
      )}
      
      <div className="relative">
        <SliderPrimitive.Root
          ref={ref}
          id={sliderId}
          className={cn(
            "relative flex w-full touch-none select-none items-center",
            "focus-within:outline-none", // Remove default outline, rely on thumb focus
            className
          )}
          value={value}
          defaultValue={defaultValue}
          onValueChange={onValueChange}
          min={min}
          max={max}
          step={step}
          disabled={disabled}
          {...accessibilityProps}
          {...props}
        >
          <SliderPrimitive.Track className="relative h-2 w-full grow overflow-hidden rounded-full bg-secondary">
            <SliderPrimitive.Range className="absolute h-full bg-primary" />
          </SliderPrimitive.Track>
          <SliderPrimitive.Thumb 
            className={cn(
              "block h-5 w-5 rounded-full border-2 border-primary bg-background ring-offset-background transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
              "disabled:pointer-events-none disabled:opacity-50",
              "hover:scale-110 active:scale-95 transition-transform",
              "cursor-grab active:cursor-grabbing"
            )}
          />
        </SliderPrimitive.Root>
        
        {showValue && (
          <div 
            id={valueId}
            className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-popover text-popover-foreground px-2 py-1 rounded text-xs font-medium border shadow-md"
            aria-live="polite"
          >
            {valueLabel ? valueLabel(displayValue) : `${displayValue}${unit}`}
          </div>
        )}
      </div>
    </div>
  )
})
Slider.displayName = SliderPrimitive.Root.displayName

export { Slider }