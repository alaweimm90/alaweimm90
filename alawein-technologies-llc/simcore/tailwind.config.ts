import type { Config } from "tailwindcss";

export default {
	darkMode: ["class"],
	content: [
		"./pages/**/*.{ts,tsx}",
		"./components/**/*.{ts,tsx}",
		"./app/**/*.{ts,tsx}",
		"./src/**/*.{ts,tsx}",
	],
	prefix: "",
	theme: {
		container: {
			center: true,
			padding: {
				DEFAULT: '1rem',
				sm: '1.5rem',
				lg: '2rem',
			},
			screens: {
				sm: '640px',
				md: '768px',
				lg: '1024px',
				xl: '1280px',
				'2xl': '1400px'
			}
		},
			extend: {
				fontFamily: {
					'sans': ['Inter', 'system-ui', 'sans-serif'],
					'mono': ['JetBrains Mono', 'Monaco', 'Cascadia Code', 'monospace'],
					'serif': ['Crimson Text', 'Times New Roman', 'serif'],
					'display': ['Playfair Display', 'serif'],
				},
			colors: {
				border: 'hsl(var(--border))',
				input: 'hsl(var(--input))',
				ring: 'hsl(var(--ring))',
				background: 'hsl(var(--background))',
				foreground: 'hsl(var(--foreground))',
				primary: {
					DEFAULT: 'hsl(var(--primary))',
					foreground: 'hsl(var(--primary-foreground))',
					glow: 'hsl(var(--primary-glow))'
				},
				secondary: {
					DEFAULT: 'hsl(var(--secondary))',
					foreground: 'hsl(var(--secondary-foreground))'
				},
				destructive: {
					DEFAULT: 'hsl(var(--destructive))',
					foreground: 'hsl(var(--destructive-foreground))'
				},
				muted: {
					DEFAULT: 'hsl(var(--muted))',
					foreground: 'hsl(var(--muted-foreground))'
				},
				accent: {
					DEFAULT: 'hsl(var(--accent))',
					foreground: 'hsl(var(--accent-foreground))'
				},
				popover: {
					DEFAULT: 'hsl(var(--popover))',
					foreground: 'hsl(var(--popover-foreground))'
				},
				card: {
					DEFAULT: 'hsl(var(--card))',
					foreground: 'hsl(var(--card-foreground))'
				}
			},
			backgroundImage: {
				'gradient-quantum': 'var(--gradient-quantum)',
				'gradient-field': 'var(--gradient-field)',
				'gradient-cosmic': 'var(--gradient-cosmic)',
				'gradient-wave': 'var(--gradient-wave)'
			},
			boxShadow: {
				'quantum': 'var(--shadow-quantum)',
				'field': 'var(--shadow-field)',
				'depth': 'var(--shadow-depth)'
			},
			transitionTimingFunction: {
				'quantum': 'cubic-bezier(0.4, 0, 0.2, 1)',
				'field': 'cubic-bezier(0.25, 0.46, 0.45, 0.94)'
			},
			borderRadius: {
				lg: 'var(--radius)',
				md: 'calc(var(--radius) - 2px)',
				sm: 'calc(var(--radius) - 4px)'
			},
			keyframes: {
				// Accordion Animations
				'accordion-down': {
					from: { height: '0', opacity: '0' },
					to: { height: 'var(--radix-accordion-content-height)', opacity: '1' }
				},
				'accordion-up': {
					from: { height: 'var(--radix-accordion-content-height)', opacity: '1' },
					to: { height: '0', opacity: '0' }
				},

				// Fade Animations
				'fade-in': {
					'0%': { opacity: '0', transform: 'translateY(10px)' },
					'100%': { opacity: '1', transform: 'translateY(0)' }
				},
				'fade-out': {
					'0%': { opacity: '1', transform: 'translateY(0)' },
					'100%': { opacity: '0', transform: 'translateY(10px)' }
				},

				// Scale Animations  
				'scale-in': {
					'0%': { transform: 'scale(0.95)', opacity: '0' },
					'100%': { transform: 'scale(1)', opacity: '1' }
				},
				'scale-out': {
					from: { transform: 'scale(1)', opacity: '1' },
					to: { transform: 'scale(0.95)', opacity: '0' }
				},

				// Slide Animations
				'slide-in-right': {
					'0%': { transform: 'translateX(100%)' },
					'100%': { transform: 'translateX(0)' }
				},
				'slide-out-right': {
					'0%': { transform: 'translateX(0)' },
					'100%': { transform: 'translateX(100%)' }
				},
				'slide-up': {
					'0%': { transform: 'translateY(20px)', opacity: '0' },
					'100%': { transform: 'translateY(0)', opacity: '1' }
				},
				'slide-down': {
					'0%': { transform: 'translateY(-20px)', opacity: '0' },
					'100%': { transform: 'translateY(0)', opacity: '1' }
				},
				'slide-left': {
					'0%': { transform: 'translateX(20px)', opacity: '0' },
					'100%': { transform: 'translateX(0)', opacity: '1' }
				},

				// Enhanced Animations
				'glow': {
					'0%, 100%': { boxShadow: '0 0 20px hsl(var(--primary) / 0.3)' },
					'50%': { boxShadow: '0 0 40px hsl(var(--primary) / 0.6)' }
				},
				'wiggle': {
					'0%, 100%': { transform: 'rotate(-3deg)' },
					'50%': { transform: 'rotate(3deg)' }
				},

				// Physics-specific animations
				'quantum-pulse': {
					'0%, 100%': { 
						boxShadow: '0 0 20px hsl(240 100% 65% / 0.3)',
						transform: 'scale(1)'
					},
					'50%': { 
						boxShadow: '0 0 40px hsl(240 100% 65% / 0.6)',
						transform: 'scale(1.02)'
					}
				},
				'wave-flow': {
					'0%': { transform: 'translateX(-100%)' },
					'100%': { transform: 'translateX(100%)' }
				},
				'orbital-rotation': {
					'0%': { transform: 'rotate(0deg) translateX(100px) rotate(0deg)' },
					'100%': { transform: 'rotate(360deg) translateX(100px) rotate(-360deg)' }
				},
				'field-shimmer': {
					'0%': { opacity: '0.3', transform: 'translateX(-100px)' },
					'50%': { opacity: '0.8' },
					'100%': { opacity: '0.3', transform: 'translateX(100px)' }
				},
				'scroll-horizontal': {
					'0%': { transform: 'translateX(0%)' },
					'100%': { transform: 'translateX(-50%)' }
				},
				'float-slow': {
					'0%, 100%': { transform: 'translateY(0px) rotate(0deg)', opacity: '0.6' },
					'33%': { transform: 'translateY(-8px) rotate(1deg)', opacity: '0.8' },
					'66%': { transform: 'translateY(4px) rotate(-1deg)', opacity: '0.7' }
				},
				'float-medium': {
					'0%, 100%': { transform: 'translateY(0px) rotate(0deg)', opacity: '0.5' },
					'50%': { transform: 'translateY(-12px) rotate(2deg)', opacity: '0.9' }
				},
				'float-fast': {
					'0%, 100%': { transform: 'translateY(0px) rotate(0deg)', opacity: '0.4' },
					'25%': { transform: 'translateY(-6px) rotate(-1deg)', opacity: '0.7' },
					'75%': { transform: 'translateY(6px) rotate(1deg)', opacity: '0.6' }
				},
				'quantum-drift': {
					'0%': { transform: 'translate(0, 0) scale(1)', opacity: '0.3' },
					'33%': { transform: 'translate(20px, -15px) scale(1.1)', opacity: '0.7' },
					'66%': { transform: 'translate(-15px, 10px) scale(0.9)', opacity: '0.5' },
					'100%': { transform: 'translate(0, 0) scale(1)', opacity: '0.3' }
				}
			},
			animation: {
				// Basic Animations
				'accordion-down': 'accordion-down 0.2s ease-out',
				'accordion-up': 'accordion-up 0.2s ease-out',
				'fade-in': 'fade-in 0.3s ease-out',
				'fade-out': 'fade-out 0.3s ease-out',
				'scale-in': 'scale-in 0.2s ease-out',
				'scale-out': 'scale-out 0.2s ease-out',
				'slide-in-right': 'slide-in-right 0.3s ease-out',
				'slide-out-right': 'slide-out-right 0.3s ease-out',
				'slide-up': 'slide-up 0.3s ease-out',
				'slide-down': 'slide-down 0.3s ease-out',
				'slide-left': 'slide-left 0.3s ease-out',
				
				// Enhanced Animations
				'glow': 'glow 2s ease-in-out infinite alternate',
				'wiggle': 'wiggle 1s ease-in-out infinite',
				
				// Combined Animations
				'enter': 'fade-in 0.3s ease-out, scale-in 0.2s ease-out',
				'exit': 'fade-out 0.3s ease-out, scale-out 0.2s ease-out',

				// Physics Animations
				'quantum-pulse': 'quantum-pulse 3s ease-in-out infinite',
				'wave-flow': 'wave-flow 4s linear infinite',
				'orbital-rotation': 'orbital-rotation 8s linear infinite',
				'field-shimmer': 'field-shimmer 2s ease-in-out infinite',
				'scroll-horizontal': 'scroll-horizontal 20s linear infinite',
				'float-slow': 'float-slow 6s ease-in-out infinite',
				'float-medium': 'float-medium 4s ease-in-out infinite',
				'float-fast': 'float-fast 3s ease-in-out infinite',
				'quantum-drift': 'quantum-drift 8s ease-in-out infinite'
			}
		}
	},
	plugins: [
		require("tailwindcss-animate"),
		function({ addUtilities }: any) {
			addUtilities({
				'.text-balance': {
					'text-wrap': 'balance'
				},
				'.hover-scale': {
					'@apply transition-transform duration-200 hover:scale-105 cursor-pointer': {}
				}
			})
		}
	],
} satisfies Config;
