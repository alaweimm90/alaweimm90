import * as React from "react"
import * as SwitchPrimitives from "@radix-ui/react-switch"

import { cn } from "@/lib/utils"

interface SwitchProps extends React.ComponentPropsWithoutRef<typeof SwitchPrimitives.Root> {
  label?: string
  description?: string
  srOnlyLabel?: string
}

const Switch = React.forwardRef<
  React.ElementRef<typeof SwitchPrimitives.Root>,
  SwitchProps
>(({ className, label, description, srOnlyLabel, id, ...props }, ref) => {
  const autoId = React.useId()
  const switchId = id ?? autoId
  const labelId = `${switchId}-label`
  const descId = `${switchId}-desc`
  
  const describedByIds = [
    description ? descId : undefined
  ].filter(Boolean).join(' ') || undefined

  const switchElement = (
    <SwitchPrimitives.Root
      id={switchId}
      className={cn(
        "peer inline-flex h-6 w-11 shrink-0 cursor-pointer items-center rounded-full border-2 border-transparent transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:cursor-not-allowed disabled:opacity-50 data-[state=checked]:bg-primary data-[state=unchecked]:bg-input",
        className
      )}
      {...props}
      ref={ref}
      aria-labelledby={label ? labelId : undefined}
      aria-describedby={describedByIds}
    >
      <SwitchPrimitives.Thumb className={cn(
        "pointer-events-none block h-5 w-5 rounded-full bg-background shadow-lg ring-0 transition-transform data-[state=checked]:translate-x-5 data-[state=unchecked]:translate-x-0"
      )} />
      {srOnlyLabel && <span className="sr-only">{srOnlyLabel}</span>}
    </SwitchPrimitives.Root>
  )

  if (label || description) {
    return (
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0 mt-0.5">
          {switchElement}
        </div>
        <div className="space-y-1">
          {label && (
            <label 
              id={labelId}
              htmlFor={switchId}
              className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
            >
              {label}
            </label>
          )}
          {description && (
            <p id={descId} className="text-sm text-muted-foreground">
              {description}
            </p>
          )}
        </div>
      </div>
    )
  }

  return switchElement
})
Switch.displayName = SwitchPrimitives.Root.displayName

export { Switch }
