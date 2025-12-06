/**
 * Animation Library
 * Pre-built Framer Motion animation presets
 */

import { Variants, Transition, AnimationControls } from 'framer-motion';

// Animation Durations
export const duration = {
  instant: 0,
  fast: 0.1,
  normal: 0.3,
  slow: 0.5,
  slower: 0.7,
  slowest: 1,
} as const;

// Easing Functions
export const easing = {
  linear: [0, 0, 1, 1],
  easeIn: [0.4, 0, 1, 1],
  easeOut: [0, 0, 0.2, 1],
  easeInOut: [0.4, 0, 0.2, 1],
  spring: [0.175, 0.885, 0.32, 1.275],
  bounce: [0.68, -0.55, 0.265, 1.55],
} as const;

// Spring Configurations
export const spring = {
  default: {
    type: 'spring',
    stiffness: 100,
    damping: 10,
  },
  gentle: {
    type: 'spring',
    stiffness: 50,
    damping: 20,
  },
  wobbly: {
    type: 'spring',
    stiffness: 180,
    damping: 12,
  },
  stiff: {
    type: 'spring',
    stiffness: 300,
    damping: 20,
  },
  slow: {
    type: 'spring',
    stiffness: 40,
    damping: 30,
  },
} as const;

// Fade Animations
export const fade: Variants = {
  initial: {
    opacity: 0,
  },
  animate: {
    opacity: 1,
    transition: {
      duration: duration.normal,
      ease: easing.easeOut,
    },
  },
  exit: {
    opacity: 0,
    transition: {
      duration: duration.fast,
      ease: easing.easeIn,
    },
  },
};

export const fadeIn: Variants = {
  initial: {
    opacity: 0,
  },
  animate: {
    opacity: 1,
    transition: {
      duration: duration.normal,
      ease: easing.easeOut,
    },
  },
};

export const fadeOut: Variants = {
  initial: {
    opacity: 1,
  },
  animate: {
    opacity: 0,
    transition: {
      duration: duration.normal,
      ease: easing.easeIn,
    },
  },
};

// Scale Animations
export const scale: Variants = {
  initial: {
    scale: 0,
    opacity: 0,
  },
  animate: {
    scale: 1,
    opacity: 1,
    transition: spring.default,
  },
  exit: {
    scale: 0,
    opacity: 0,
    transition: {
      duration: duration.fast,
      ease: easing.easeIn,
    },
  },
};

export const scaleUp: Variants = {
  initial: {
    scale: 0.95,
    opacity: 0,
  },
  animate: {
    scale: 1,
    opacity: 1,
    transition: {
      duration: duration.normal,
      ease: easing.easeOut,
    },
  },
  exit: {
    scale: 0.95,
    opacity: 0,
    transition: {
      duration: duration.fast,
      ease: easing.easeIn,
    },
  },
};

export const scaleDown: Variants = {
  initial: {
    scale: 1.05,
    opacity: 0,
  },
  animate: {
    scale: 1,
    opacity: 1,
    transition: {
      duration: duration.normal,
      ease: easing.easeOut,
    },
  },
  exit: {
    scale: 1.05,
    opacity: 0,
    transition: {
      duration: duration.fast,
      ease: easing.easeIn,
    },
  },
};

// Slide Animations
export const slideUp: Variants = {
  initial: {
    y: 20,
    opacity: 0,
  },
  animate: {
    y: 0,
    opacity: 1,
    transition: {
      duration: duration.normal,
      ease: easing.easeOut,
    },
  },
  exit: {
    y: -20,
    opacity: 0,
    transition: {
      duration: duration.fast,
      ease: easing.easeIn,
    },
  },
};

export const slideDown: Variants = {
  initial: {
    y: -20,
    opacity: 0,
  },
  animate: {
    y: 0,
    opacity: 1,
    transition: {
      duration: duration.normal,
      ease: easing.easeOut,
    },
  },
  exit: {
    y: 20,
    opacity: 0,
    transition: {
      duration: duration.fast,
      ease: easing.easeIn,
    },
  },
};

export const slideLeft: Variants = {
  initial: {
    x: 20,
    opacity: 0,
  },
  animate: {
    x: 0,
    opacity: 1,
    transition: {
      duration: duration.normal,
      ease: easing.easeOut,
    },
  },
  exit: {
    x: -20,
    opacity: 0,
    transition: {
      duration: duration.fast,
      ease: easing.easeIn,
    },
  },
};

export const slideRight: Variants = {
  initial: {
    x: -20,
    opacity: 0,
  },
  animate: {
    x: 0,
    opacity: 1,
    transition: {
      duration: duration.normal,
      ease: easing.easeOut,
    },
  },
  exit: {
    x: 20,
    opacity: 0,
    transition: {
      duration: duration.fast,
      ease: easing.easeIn,
    },
  },
};

// Slide Full Screen Animations
export const slideInFromBottom: Variants = {
  initial: {
    y: '100%',
  },
  animate: {
    y: 0,
    transition: {
      duration: duration.slow,
      ease: easing.easeOut,
    },
  },
  exit: {
    y: '100%',
    transition: {
      duration: duration.normal,
      ease: easing.easeIn,
    },
  },
};

export const slideInFromTop: Variants = {
  initial: {
    y: '-100%',
  },
  animate: {
    y: 0,
    transition: {
      duration: duration.slow,
      ease: easing.easeOut,
    },
  },
  exit: {
    y: '-100%',
    transition: {
      duration: duration.normal,
      ease: easing.easeIn,
    },
  },
};

export const slideInFromLeft: Variants = {
  initial: {
    x: '-100%',
  },
  animate: {
    x: 0,
    transition: {
      duration: duration.slow,
      ease: easing.easeOut,
    },
  },
  exit: {
    x: '-100%',
    transition: {
      duration: duration.normal,
      ease: easing.easeIn,
    },
  },
};

export const slideInFromRight: Variants = {
  initial: {
    x: '100%',
  },
  animate: {
    x: 0,
    transition: {
      duration: duration.slow,
      ease: easing.easeOut,
    },
  },
  exit: {
    x: '100%',
    transition: {
      duration: duration.normal,
      ease: easing.easeIn,
    },
  },
};

// Rotate Animations
export const rotate: Variants = {
  initial: {
    rotate: -180,
    opacity: 0,
  },
  animate: {
    rotate: 0,
    opacity: 1,
    transition: {
      duration: duration.slow,
      ease: easing.easeOut,
    },
  },
  exit: {
    rotate: 180,
    opacity: 0,
    transition: {
      duration: duration.normal,
      ease: easing.easeIn,
    },
  },
};

export const rotate360: Variants = {
  initial: {
    rotate: 0,
  },
  animate: {
    rotate: 360,
    transition: {
      duration: duration.slow,
      ease: easing.linear,
      repeat: Infinity,
    },
  },
};

// Flip Animations
export const flipX: Variants = {
  initial: {
    rotateX: 90,
    opacity: 0,
  },
  animate: {
    rotateX: 0,
    opacity: 1,
    transition: {
      duration: duration.normal,
      ease: easing.easeOut,
    },
  },
  exit: {
    rotateX: 90,
    opacity: 0,
    transition: {
      duration: duration.fast,
      ease: easing.easeIn,
    },
  },
};

export const flipY: Variants = {
  initial: {
    rotateY: 90,
    opacity: 0,
  },
  animate: {
    rotateY: 0,
    opacity: 1,
    transition: {
      duration: duration.normal,
      ease: easing.easeOut,
    },
  },
  exit: {
    rotateY: 90,
    opacity: 0,
    transition: {
      duration: duration.fast,
      ease: easing.easeIn,
    },
  },
};

// Stagger Animations
export const staggerContainer: Variants = {
  initial: {},
  animate: {
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2,
    },
  },
  exit: {
    transition: {
      staggerChildren: 0.05,
      staggerDirection: -1,
    },
  },
};

export const staggerItem: Variants = {
  initial: {
    y: 20,
    opacity: 0,
  },
  animate: {
    y: 0,
    opacity: 1,
    transition: {
      duration: duration.normal,
      ease: easing.easeOut,
    },
  },
  exit: {
    y: 20,
    opacity: 0,
    transition: {
      duration: duration.fast,
      ease: easing.easeIn,
    },
  },
};

// Page Transition Animations
export const pageTransition: Variants = {
  initial: {
    opacity: 0,
    y: 20,
  },
  animate: {
    opacity: 1,
    y: 0,
    transition: {
      duration: duration.slow,
      ease: easing.easeOut,
    },
  },
  exit: {
    opacity: 0,
    y: -20,
    transition: {
      duration: duration.normal,
      ease: easing.easeIn,
    },
  },
};

// Modal Animations
export const modalOverlay: Variants = {
  initial: {
    opacity: 0,
  },
  animate: {
    opacity: 1,
    transition: {
      duration: duration.normal,
      ease: easing.easeOut,
    },
  },
  exit: {
    opacity: 0,
    transition: {
      duration: duration.fast,
      ease: easing.easeIn,
    },
  },
};

export const modalContent: Variants = {
  initial: {
    scale: 0.9,
    opacity: 0,
    y: 20,
  },
  animate: {
    scale: 1,
    opacity: 1,
    y: 0,
    transition: spring.default,
  },
  exit: {
    scale: 0.9,
    opacity: 0,
    y: 20,
    transition: {
      duration: duration.fast,
      ease: easing.easeIn,
    },
  },
};

// Toast/Notification Animations
export const toast: Variants = {
  initial: {
    x: '100%',
    opacity: 0,
  },
  animate: {
    x: 0,
    opacity: 1,
    transition: spring.wobbly,
  },
  exit: {
    x: '100%',
    opacity: 0,
    transition: {
      duration: duration.normal,
      ease: easing.easeIn,
    },
  },
};

// Skeleton Loading Animation
export const skeleton: Variants = {
  initial: {
    backgroundPosition: '-200% 0',
  },
  animate: {
    backgroundPosition: '200% 0',
    transition: {
      duration: 1.5,
      ease: easing.linear,
      repeat: Infinity,
    },
  },
};

// Pulse Animation
export const pulse: Variants = {
  initial: {
    scale: 1,
  },
  animate: {
    scale: [1, 1.05, 1],
    transition: {
      duration: duration.slow,
      ease: easing.easeInOut,
      repeat: Infinity,
    },
  },
};

// Bounce Animation
export const bounce: Variants = {
  initial: {
    y: 0,
  },
  animate: {
    y: [0, -20, 0],
    transition: {
      duration: duration.slow,
      ease: easing.bounce,
      repeat: Infinity,
    },
  },
};

// Shake Animation
export const shake: Variants = {
  initial: {
    x: 0,
  },
  animate: {
    x: [-10, 10, -10, 10, 0],
    transition: {
      duration: duration.normal,
      ease: easing.linear,
    },
  },
};

// Hover Animations
export const hoverScale = {
  whileHover: {
    scale: 1.05,
    transition: {
      duration: duration.fast,
      ease: easing.easeOut,
    },
  },
  whileTap: {
    scale: 0.95,
    transition: {
      duration: duration.instant,
    },
  },
};

export const hoverLift = {
  whileHover: {
    y: -4,
    transition: {
      duration: duration.fast,
      ease: easing.easeOut,
    },
  },
  whileTap: {
    y: 0,
    transition: {
      duration: duration.instant,
    },
  },
};

export const hoverGlow = {
  whileHover: {
    boxShadow: '0 0 20px rgba(99, 102, 241, 0.4)',
    transition: {
      duration: duration.fast,
      ease: easing.easeOut,
    },
  },
};

// Complex Animations
export const morphing: Variants = {
  initial: {
    borderRadius: '50%',
    width: '100px',
    height: '100px',
  },
  animate: {
    borderRadius: ['50%', '10%', '50%'],
    width: ['100px', '200px', '100px'],
    height: ['100px', '200px', '100px'],
    transition: {
      duration: 2,
      ease: easing.easeInOut,
      repeat: Infinity,
    },
  },
};

export const typewriter = {
  initial: {
    width: 0,
    opacity: 0,
  },
  animate: {
    width: '100%',
    opacity: 1,
    transition: {
      duration: 2,
      ease: easing.linear,
    },
  },
};

// Utility function to create custom stagger
export const createStagger = (staggerTime: number = 0.1, delayTime: number = 0) => ({
  animate: {
    transition: {
      staggerChildren: staggerTime,
      delayChildren: delayTime,
    },
  },
});

// Utility function to create custom spring
export const createSpring = (stiffness: number = 100, damping: number = 10) => ({
  type: 'spring',
  stiffness,
  damping,
});

// Animation Composer
// Allows combining multiple animation variants
export const composeAnimations = (...animations: Variants[]): Variants => {
  return animations.reduce((acc, curr) => ({
    initial: { ...acc.initial, ...curr.initial },
    animate: { ...acc.animate, ...curr.animate },
    exit: { ...acc.exit, ...curr.exit },
  }), { initial: {}, animate: {}, exit: {} });
};

// Gesture Animations
export const dragConstraints = {
  top: -100,
  left: -100,
  right: 100,
  bottom: 100,
};

export const dragTransition = {
  bounceStiffness: 600,
  bounceDamping: 20,
};

// Parallax Effect
export const parallax = (offset: number = 100) => ({
  initial: {
    y: -offset,
  },
  animate: {
    y: offset,
    transition: {
      duration: duration.slowest,
      ease: easing.linear,
    },
  },
});

// 3D Animations
export const rotate3D: Variants = {
  initial: {
    rotateX: 0,
    rotateY: 0,
  },
  animate: {
    rotateX: 360,
    rotateY: 360,
    transition: {
      duration: 20,
      ease: easing.linear,
      repeat: Infinity,
    },
  },
};

// Reveal Animations
export const reveal: Variants = {
  initial: {
    clipPath: 'polygon(0 0, 0 0, 0 100%, 0 100%)',
  },
  animate: {
    clipPath: 'polygon(0 0, 100% 0, 100% 100%, 0 100%)',
    transition: {
      duration: duration.slow,
      ease: easing.easeOut,
    },
  },
};

// Text Animations
export const textReveal: Variants = {
  initial: {
    opacity: 0,
    y: 20,
  },
  animate: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: {
      delay: i * 0.05,
      duration: duration.normal,
      ease: easing.easeOut,
    },
  }),
};

// Export all animations as a single object for easy import
export const animations = {
  // Basic
  fade,
  fadeIn,
  fadeOut,
  scale,
  scaleUp,
  scaleDown,

  // Slides
  slideUp,
  slideDown,
  slideLeft,
  slideRight,
  slideInFromBottom,
  slideInFromTop,
  slideInFromLeft,
  slideInFromRight,

  // Rotations
  rotate,
  rotate360,
  flipX,
  flipY,
  rotate3D,

  // Stagger
  staggerContainer,
  staggerItem,

  // Page & Modal
  pageTransition,
  modalOverlay,
  modalContent,

  // Feedback
  toast,
  skeleton,
  pulse,
  bounce,
  shake,

  // Hover
  hoverScale,
  hoverLift,
  hoverGlow,

  // Complex
  morphing,
  typewriter,
  parallax,
  reveal,
  textReveal,

  // Utilities
  spring,
  easing,
  duration,
  createStagger,
  createSpring,
  composeAnimations,
};