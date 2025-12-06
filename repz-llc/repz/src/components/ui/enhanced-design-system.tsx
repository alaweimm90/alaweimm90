import React from 'react'
import { cn } from '@/lib/utils'
import { motion } from 'framer-motion'

// ============================================================================
// PHASE 2: ENHANCED DESIGN SYSTEM COMPONENTS
// Beautiful, Consistent, Professional UI Components
// ============================================================================

// Enhanced Button Variants with Glass Morphism
interface EnhancedButtonProps {
  variant?: 'premium' | 'elegant' | 'ghost' | 'glass' | 'tier'
  size?: 'sm' | 'md' | 'lg' | 'xl'
  glow?: boolean
  animation?: 'float' | 'pulse' | 'shimmer' | 'none'
  tier?: 'core' | 'adaptive' | 'performance' | 'longevity'
  className?: string
  children?: React.ReactNode
  onClick?: () => void
  disabled?: boolean
}

export const EnhancedButton: React.FC<EnhancedButtonProps> = ({
  className,
  variant = 'premium',
  size = 'md',
  glow = false,
  animation = 'none',
  tier,
  children,
  onClick,
  disabled,
  ...props
}) => {
  const baseClasses = "inline-flex items-center justify-center font-montserrat font-semibold transition-all duration-elegant relative overflow-hidden focus-premium"
  
  const variants = {
    premium: "btn-premium",
    elegant: "btn-elegant", 
    ghost: "btn-ghost-premium",
    glass: "bg-surface-glass backdrop-blur-xl border border-white/20 text-white hover:bg-surface-frost hover:border-white/30",
    tier: tier ? `bg-tier-${tier}/20 border border-tier-${tier}/30 text-tier-${tier} hover:bg-tier-${tier}/30 hover:border-tier-${tier}/50` : "bg-primary/20 border border-primary/30 text-primary"
  }
  
  const sizes = {
    sm: "px-3 py-1.5 text-sm rounded-lg",
    md: "px-6 py-3 text-base rounded-xl", 
    lg: "px-8 py-4 text-lg rounded-xl",
    xl: "px-12 py-6 text-xl rounded-2xl"
  }
  
  const animations = {
    float: "animate-elegant-float",
    pulse: "animate-sophisticated-pulse", 
    shimmer: "animate-subtle-shimmer",
    none: ""
  }
  
  const glowEffect = glow ? "glow-orange hover:glow-gold" : ""
  
  return (
    <motion.button
      className={cn(
        baseClasses,
        variants[variant],
        sizes[size],
        animations[animation],
        glowEffect,
        className
      )}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </motion.button>
  )
}

// Enhanced Card System with Advanced Glass Morphism
interface EnhancedCardProps {
  variant?: 'glass' | 'frost' | 'velvet' | 'hero' | 'interactive'
  glow?: boolean
  animation?: 'float' | 'pulse' | 'none'
  tier?: 'core' | 'adaptive' | 'performance' | 'longevity'
  blur?: 'sm' | 'md' | 'lg' | 'xl'
  className?: string
  children?: React.ReactNode
}

export const EnhancedCard: React.FC<EnhancedCardProps> = ({
  className,
  variant = 'glass',
  glow = false,
  animation = 'none',
  tier,
  blur = 'md',
  children,
  ...props
}) => {
  const variants = {
    glass: "panel-glass",
    frost: "panel-frost", 
    velvet: "panel-velvet",
    hero: "panel-hero",
    interactive: "card-interactive"
  }
  
  const animations = {
    float: "animate-elegant-float",
    pulse: "animate-sophisticated-pulse",
    none: ""
  }
  
  const blurLevels = {
    sm: "backdrop-blur-sm",
    md: "backdrop-blur-md", 
    lg: "backdrop-blur-lg",
    xl: "backdrop-blur-xl"
  }
  
  const tierAccent = tier ? `border-tier-${tier}/30 hover:border-tier-${tier}/50` : ""
  const glowEffect = glow ? "glow-orange hover:glow-gold" : ""
  
  return (
    <motion.div
      className={cn(
        variants[variant],
        animations[animation],
        blurLevels[blur],
        tierAccent,
        glowEffect,
        "will-change-transform",
        className
      )}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      whileHover={{ scale: 1.02, y: -2 }}
    >
      {children}
    </motion.div>
  )
}

// Enhanced Typography System
interface EnhancedHeadingProps extends React.HTMLAttributes<HTMLHeadingElement> {
  variant?: 'hero' | 'section' | 'subtitle' | 'caption'
  gradient?: boolean
  glow?: boolean
  tier?: 'core' | 'adaptive' | 'performance' | 'longevity'
  as?: 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6'
}

export const EnhancedHeading: React.FC<EnhancedHeadingProps> = ({
  className,
  variant = 'section',
  gradient = false,
  glow = false,
  tier,
  as: Component = 'h2',
  children,
  ...props
}) => {
  const variants = {
    hero: "heading-hero",
    section: "heading-section", 
    subtitle: "text-xl md:text-2xl font-montserrat font-semibold text-white/90",
    caption: "text-sm md:text-base font-inter font-medium text-white/70"
  }
  
  const gradientClass = gradient 
    ? tier 
      ? `bg-gradient-to-r from-white to-tier-${tier} bg-clip-text text-transparent`
      : "bg-gradient-to-r from-white to-repz-orange bg-clip-text text-transparent"
    : ""
    
  const glowClass = glow ? "drop-shadow-lg" : ""
  
  return (
    <Component
      className={cn(
        variants[variant],
        gradientClass,
        glowClass,
        className
      )}
      {...props}
    >
      {children}
    </Component>
  )
}

// Enhanced Text Component
interface EnhancedTextProps extends React.HTMLAttributes<HTMLParagraphElement> {
  variant?: 'elegant' | 'premium' | 'muted' | 'caption'
  size?: 'sm' | 'md' | 'lg' | 'xl'
  weight?: 'light' | 'normal' | 'medium' | 'semibold' | 'bold'
}

export const EnhancedText: React.FC<EnhancedTextProps> = ({
  className,
  variant = 'elegant',
  size = 'md',
  weight = 'normal',
  children,
  ...props
}) => {
  const variants = {
    elegant: "text-elegant",
    premium: "text-premium",
    muted: "text-white/60",
    caption: "text-white/50 text-sm"
  }
  
  const sizes = {
    sm: "text-sm",
    md: "text-base", 
    lg: "text-lg",
    xl: "text-xl"
  }
  
  const weights = {
    light: "font-light",
    normal: "font-normal",
    medium: "font-medium", 
    semibold: "font-semibold",
    bold: "font-bold"
  }
  
  return (
    <p
      className={cn(
        variants[variant],
        sizes[size],
        weights[weight],
        className
      )}
      {...props}
    >
      {children}
    </p>
  )
}

// Enhanced Badge System
interface EnhancedBadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'tier' | 'status' | 'feature' | 'metric'
  tier?: 'core' | 'adaptive' | 'performance' | 'longevity'
  status?: 'active' | 'inactive' | 'pending' | 'complete'
  glow?: boolean
  pulse?: boolean
}

export const EnhancedBadge: React.FC<EnhancedBadgeProps> = ({
  className,
  variant = 'feature',
  tier,
  status,
  glow = false,
  pulse = false,
  children,
  ...props
}) => {
  const baseClasses = "inline-flex items-center px-3 py-1 rounded-full text-xs font-montserrat font-semibold backdrop-blur-sm"
  
  const variants = {
    tier: tier ? `bg-tier-${tier}/20 text-tier-${tier} border border-tier-${tier}/30` : "bg-primary/20 text-primary border border-primary/30",
    status: status ? getStatusClasses(status) : "bg-surface-glass text-white border border-white/20",
    feature: "bg-surface-frost text-white/90 border border-white/15",
    metric: "bg-gradient-to-r from-repz-orange/20 to-amber-elegant/20 text-repz-orange border border-repz-orange/30"
  }
  
  const effects = cn(
    glow && "glow-orange",
    pulse && "animate-sophisticated-pulse"
  )
  
  return (
    <span
      className={cn(
        baseClasses,
        variants[variant],
        effects,
        className
      )}
      {...props}
    >
      {children}
    </span>
  )
}

function getStatusClasses(status: string) {
  switch(status) {
    case 'active': return "bg-green-500/20 text-green-400 border border-green-500/30"
    case 'inactive': return "bg-gray-500/20 text-gray-400 border border-gray-500/30"
    case 'pending': return "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30"
    case 'complete': return "bg-blue-500/20 text-blue-400 border border-blue-500/30"
    default: return "bg-surface-glass text-white border border-white/20"
  }
}

// Enhanced Container System
interface EnhancedContainerProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'page' | 'section' | 'content' | 'sidebar'
  padding?: 'none' | 'sm' | 'md' | 'lg' | 'xl'
  background?: 'none' | 'primary' | 'secondary' | 'hero'
}

export const EnhancedContainer: React.FC<EnhancedContainerProps> = ({
  className,
  variant = 'content',
  padding = 'md',
  background = 'none',
  children,
  ...props
}) => {
  const variants = {
    page: "min-h-screen w-full",
    section: "w-full",
    content: "container-premium",
    sidebar: "w-72 h-screen"
  }
  
  const paddings = {
    none: "",
    sm: "p-4",
    md: "section-padding",
    lg: "py-16 md:py-20 lg:py-24 px-4 md:px-6 lg:px-8", 
    xl: "py-20 md:py-24 lg:py-32 px-4 md:px-6 lg:px-8"
  }
  
  const backgrounds = {
    none: "",
    primary: "bg-primary-elegant",
    secondary: "bg-secondary-elegant",
    hero: "bg-hero-elegant"
  }
  
  return (
    <div
      className={cn(
        variants[variant],
        paddings[padding],
        backgrounds[background],
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}

// Enhanced Grid System
interface EnhancedGridProps extends React.HTMLAttributes<HTMLDivElement> {
  cols?: 1 | 2 | 3 | 4 | 6 | 12
  gap?: 'sm' | 'md' | 'lg' | 'xl'
  responsive?: boolean
}

export const EnhancedGrid: React.FC<EnhancedGridProps> = ({
  className,
  cols = 3,
  gap = 'md',
  responsive = true,
  children,
  ...props
}) => {
  const baseClasses = "grid"
  
  const colClasses = responsive 
    ? `grid-cols-1 ${cols >= 2 ? 'md:grid-cols-2' : ''} ${cols >= 3 ? 'lg:grid-cols-3' : ''} ${cols >= 4 ? 'xl:grid-cols-4' : ''}`
    : `grid-cols-${cols}`
    
  const gaps = {
    sm: "gap-4",
    md: "gap-6",
    lg: "gap-8", 
    xl: "gap-12"
  }
  
  return (
    <div
      className={cn(
        baseClasses,
        colClasses,
        gaps[gap],
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}

// Enhanced Loading States
interface EnhancedLoadingProps {
  variant?: 'spinner' | 'dots' | 'pulse' | 'skeleton'
  size?: 'sm' | 'md' | 'lg'
  message?: string
  className?: string
}

export const EnhancedLoading: React.FC<EnhancedLoadingProps> = ({
  variant = 'spinner',
  size = 'md',
  message,
  className
}) => {
  const sizes = {
    sm: "w-4 h-4",
    md: "w-8 h-8",
    lg: "w-12 h-12"
  }
  
  const variants = {
    spinner: (
      <div className={cn("animate-spin rounded-full border-2 border-muted border-t-primary", sizes[size])} />
    ),
    dots: (
      <div className="flex space-x-1">
        {[0, 1, 2].map(i => (
          <div 
            key={i}
            className={cn("bg-primary rounded-full animate-pulse", size === 'sm' ? 'w-1 h-1' : size === 'md' ? 'w-2 h-2' : 'w-3 h-3')}
            style={{ animationDelay: `${i * 0.1}s` }}
          />
        ))}
      </div>
    ),
    pulse: (
      <div className={cn("bg-surface-glass rounded-xl animate-sophisticated-pulse", sizes[size])} />
    ),
    skeleton: (
      <div className="space-y-3">
        <div className="h-4 bg-surface-glass rounded animate-pulse"></div>
        <div className="h-4 bg-surface-glass rounded animate-pulse w-3/4"></div>
        <div className="h-4 bg-surface-glass rounded animate-pulse w-1/2"></div>
      </div>
    )
  }
  
  return (
    <div className={cn("flex flex-col items-center justify-center space-y-2", className)}>
      {variants[variant]}
      {message && <EnhancedText variant="caption">{message}</EnhancedText>}
    </div>
  )
}