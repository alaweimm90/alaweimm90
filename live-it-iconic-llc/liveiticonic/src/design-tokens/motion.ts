// ============================================================================
// LIVE IT ICONIC - MOTION DESIGN TOKENS
// Animation Duration & Easing Functions
// ============================================================================

export const motion = {
  // ========================================================================
  // DURATION - Animation Timing Scale
  // ========================================================================
  durations: {
    instant: '0ms', // No delay
    fast: '120ms', // Micro-interactions
    normal: '160ms', // Standard micro-interactions
    slow: '240ms', // Entrance animations
    slower: '320ms', // Entrance animations
    slowest: '500ms', // Page transitions

    // ====================================================================
    // CANONICAL DURATIONS - Preferred timing values
    // ====================================================================
    // micro: 140ms - Quick UI feedback
    // enter: 280ms - Element entrance
    // standard: 300ms - General transitions
  },

  // ========================================================================
  // EASING - Timing Functions
  // ========================================================================
  easings: {
    linear: 'linear',

    // Standard Easing
    in: 'ease-in',
    out: 'ease-out',
    inOut: 'ease-in-out',

    // Cubic Bezier Functions
    easeInQuad: 'cubic-bezier(0.11, 0, 0.5, 0)',
    easeInCubic: 'cubic-bezier(0.32, 0, 0.67, 0)',
    easeInQuart: 'cubic-bezier(0.5, 0, 0.75, 0)',
    easeInQuint: 'cubic-bezier(0.64, 0, 0.78, 0)',
    easeInExpo: 'cubic-bezier(0.7, 0, 0.84, 0)',
    easeInCirc: 'cubic-bezier(0.6, 0.04, 0.98, 0.335)',

    easeOutQuad: 'cubic-bezier(0.25, 0.46, 0.45, 0.94)',
    easeOutCubic: 'cubic-bezier(0.215, 0.61, 0.355, 1)',
    easeOutQuart: 'cubic-bezier(0.165, 0.84, 0.44, 1)',
    easeOutQuint: 'cubic-bezier(0.22, 1, 0.36, 1)',
    easeOutExpo: 'cubic-bezier(0.16, 1, 0.3, 1)',
    easeOutCirc: 'cubic-bezier(0.075, 0.82, 0.165, 1)',

    easeInOutQuad: 'cubic-bezier(0.455, 0.03, 0.515, 0.955)',
    easeInOutCubic: 'cubic-bezier(0.645, 0.045, 0.355, 1)',
    easeInOutQuart: 'cubic-bezier(0.77, 0, 0.175, 1)',
    easeInOutQuint: 'cubic-bezier(0.86, 0, 0.07, 1)',
    easeInOutExpo: 'cubic-bezier(1, 0, 0, 1)',
    easeInOutCirc: 'cubic-bezier(0.785, 0.135, 0.15, 0.86)',

    // Brand Easing
    brand: 'cubic-bezier(0.16, 1, 0.3, 1)', // Custom premium easing
    bounce: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)', // Bouncy easing

    // Back easing (overshoot effect)
    easeInBack: 'cubic-bezier(0.36, 0, 0.66, -0.56)',
    easeOutBack: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
    easeInOutBack: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
  },

  // ========================================================================
  // TRANSITIONS - Common Transition Presets
  // ========================================================================
  transitions: {
    colors: 'color 300ms ease-out, background-color 300ms ease-out, border-color 300ms ease-out',
    transform: 'transform 300ms cubic-bezier(0.16, 1, 0.3, 1)',
    opacity: 'opacity 300ms ease-out',
    all: 'all 300ms cubic-bezier(0.16, 1, 0.3, 1)',
    slow: 'all 500ms cubic-bezier(0.16, 1, 0.3, 1)',
    fast: 'all 120ms ease-out',
  },

  // ========================================================================
  // KEYFRAME ANIMATIONS - Reusable animations
  // ========================================================================
  keyframes: {
    fadeIn: {
      '0%': { opacity: '0' },
      '100%': { opacity: '1' },
    },

    fadeOut: {
      '0%': { opacity: '1' },
      '100%': { opacity: '0' },
    },

    slideInUp: {
      '0%': { transform: 'translateY(16px)', opacity: '0' },
      '100%': { transform: 'translateY(0)', opacity: '1' },
    },

    slideInDown: {
      '0%': { transform: 'translateY(-16px)', opacity: '0' },
      '100%': { transform: 'translateY(0)', opacity: '1' },
    },

    slideInLeft: {
      '0%': { transform: 'translateX(-16px)', opacity: '0' },
      '100%': { transform: 'translateX(0)', opacity: '1' },
    },

    slideInRight: {
      '0%': { transform: 'translateX(16px)', opacity: '0' },
      '100%': { transform: 'translateX(0)', opacity: '1' },
    },

    scaleIn: {
      '0%': { transform: 'scale(0.95)', opacity: '0' },
      '100%': { transform: 'scale(1)', opacity: '1' },
    },

    scaleOut: {
      '0%': { transform: 'scale(1)', opacity: '1' },
      '100%': { transform: 'scale(0.95)', opacity: '0' },
    },

    spin: {
      '0%': { transform: 'rotate(0deg)' },
      '100%': { transform: 'rotate(360deg)' },
    },

    pulse: {
      '0%, 100%': { opacity: '1' },
      '50%': { opacity: '0.5' },
    },

    bounce: {
      '0%, 100%': { transform: 'translateY(0)', opacity: '1' },
      '50%': { transform: 'translateY(-8px)' },
    },

    shimmer: {
      '0%': { backgroundPosition: '-1000px 0' },
      '100%': { backgroundPosition: '1000px 0' },
    },
  },
} as const;

// ============================================================================
// MOTION PRESETS - Common Animation Combinations
// ============================================================================
export const motionPresets = {
  // Entrance animations
  enterQuick: {
    duration: motion.durations.fast,
    easing: motion.easings.easeOutCubic,
  },

  enterNormal: {
    duration: motion.durations.normal,
    easing: motion.easings.easeOutCubic,
  },

  enterSlow: {
    duration: motion.durations.slow,
    easing: motion.easings.easeOutCubic,
  },

  // Exit animations
  exitQuick: {
    duration: motion.durations.fast,
    easing: motion.easings.easeInCubic,
  },

  exitNormal: {
    duration: motion.durations.normal,
    easing: motion.easings.easeInCubic,
  },

  exitSlow: {
    duration: motion.durations.slow,
    easing: motion.easings.easeInCubic,
  },

  // Interaction animations
  interaction: {
    duration: motion.durations.normal,
    easing: motion.easings.brand,
  },

  hover: {
    duration: motion.durations.fast,
    easing: motion.easings.easeOutQuad,
  },

  // Stagger animation (for sequential animations)
  stagger: {
    delayBetween: '50ms',
  },
} as const;

// ============================================================================
// TYPE EXPORTS
// ============================================================================
export type Motion = typeof motion;
export type Duration = keyof typeof motion.durations;
export type Easing = keyof typeof motion.easings;
export type Transition = keyof typeof motion.transitions;
export type Keyframe = keyof typeof motion.keyframes;
export type MotionPreset = keyof typeof motionPresets;

// ============================================================================
// MOTION UTILITY FUNCTIONS
// ============================================================================
export const getDuration = (key: Duration): string => {
  return motion.durations[key];
};

export const getEasing = (key: Easing): string => {
  return motion.easings[key];
};

export const getTransition = (key: Transition): string => {
  return motion.transitions[key];
};

/**
 * Creates a CSS transition string
 */
export const createTransition = (property: string, duration: Duration = 'normal', easing: Easing = 'brand'): string => {
  return `${property} ${getDuration(duration)} ${getEasing(easing)}`;
};

/**
 * Creates a CSS animation string
 */
export const createAnimation = (
  name: string,
  duration: Duration = 'normal',
  easing: Easing = 'linear',
  iterationCount: string | number = 1
): string => {
  return `${name} ${getDuration(duration)} ${getEasing(easing)} ${iterationCount}`;
};

/**
 * Creates a staggered animation delay for sequential animations
 */
export const createStaggerDelay = (index: number, delayPerItem: number = 50): number => {
  return index * delayPerItem;
};

/**
 * CSS-in-JS helper for common transitions
 */
export const transitionStyles = {
  smooth: {
    transition: motion.transitions.all,
  },

  smoothColors: {
    transition: motion.transitions.colors,
  },

  smoothTransform: {
    transition: motion.transitions.transform,
  },

  smoothOpacity: {
    transition: motion.transitions.opacity,
  },
} as const;

// ============================================================================
// ANIMATION HELPERS
// ============================================================================
export const animateIn = (duration: Duration = 'normal', easing: Easing = 'easeOutCubic'): object => ({
  animation: `fadeIn ${getDuration(duration)} ${getEasing(easing)}`,
});

export const animateOut = (duration: Duration = 'normal', easing: Easing = 'easeInCubic'): object => ({
  animation: `fadeOut ${getDuration(duration)} ${getEasing(easing)}`,
});
