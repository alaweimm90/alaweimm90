import * as React from "react"

import { cn } from "@/lib/utils"

export interface InputProps extends React.ComponentProps<"input"> {
  label?: string
  description?: string
  error?: string
  success?: string
  icon?: React.ReactNode
  iconPosition?: 'left' | 'right'
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ 
    className, 
    type, 
    label, 
    description, 
    error, 
    success, 
    icon,
    iconPosition = 'left',
    id,
    'aria-describedby': ariaDescribedBy,
    'aria-invalid': ariaInvalid,
    ...props 
  }, ref) => {
    const autoId = React.useId()
    const inputId = id ?? autoId
    const labelId = `${inputId}-label`
    const descId = `${inputId}-desc`
    const errorId = `${inputId}-error`
    const successId = `${inputId}-success`
    
    // Build describedBy string
    const describedByIds = [
      ariaDescribedBy,
      description ? descId : undefined,
      error ? errorId : undefined,
      success ? successId : undefined
    ].filter(Boolean).join(' ') || undefined

    const hasError = Boolean(error)
    const hasSuccess = Boolean(success) && !hasError
    
    return (
      <div className="space-y-2">
        {label && (
          <label 
            id={labelId}
            htmlFor={inputId}
            className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
          >
            {label}
            {props.required && (
              <span className="text-destructive ml-1" aria-label="required">*</span>
            )}
          </label>
        )}
        
        {description && (
          <p id={descId} className="text-sm text-muted-foreground">
            {description}
          </p>
        )}
        
        <div className="relative">
          {icon && iconPosition === 'left' && (
            <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground">
              {icon}
            </div>
          )}
          
          <input
            id={inputId}
            type={type}
            className={cn(
              "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-base ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 md:text-sm",
              icon && iconPosition === 'left' && "pl-10",
              icon && iconPosition === 'right' && "pr-10",
              hasError && "border-destructive focus-visible:ring-destructive",
              hasSuccess && "border-green-500 focus-visible:ring-green-500",
              className
            )}
            ref={ref}
            aria-describedby={describedByIds}
            aria-invalid={ariaInvalid || hasError}
            aria-labelledby={label ? labelId : undefined}
            {...props}
          />
          
          {icon && iconPosition === 'right' && (
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground">
              {icon}
            </div>
          )}
        </div>
        
        {error && (
          <p 
            id={errorId} 
            className="text-sm text-destructive"
            role="alert"
            aria-live="polite"
          >
            {error}
          </p>
        )}
        
        {success && !error && (
          <p 
            id={successId} 
            className="text-sm text-green-600"
            role="status"
            aria-live="polite"
          >
            {success}
          </p>
        )}
      </div>
    )
  }
)
Input.displayName = "Input"

export { Input }
