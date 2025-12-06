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
			padding: '2rem',
			screens: {
				'2xl': '1400px'
			}
		},
		extend: {
			fontFamily: {
				sans: ["IBM Plex Sans", "Inter", "sans-serif"],
				mono: ["Fira Code", "JetBrains Mono", "monospace"],
				code: ["Fira Code", "monospace"],
			},
			fontFeatureSettings: {
				'tnum': '"tnum"',
				'lnum': '"lnum"',
			},
			colors: {
				border: 'hsl(var(--border))',
				input: 'hsl(var(--input))',
				ring: 'hsl(var(--ring))',
				background: 'hsl(var(--background))',
				foreground: 'hsl(var(--foreground))',
				
				// AI Detection specific colors
				'ai-high': 'hsl(var(--ai-high))',
				'ai-medium': 'hsl(var(--ai-medium))',
				'ai-low': 'hsl(var(--ai-low))',
				'ai-neutral': 'hsl(var(--ai-neutral))',
				
				// Confidence colors
				'confidence-high': 'hsl(var(--confidence-high))',
				'confidence-medium': 'hsl(var(--confidence-medium))',
				'confidence-low': 'hsl(var(--confidence-low))',
				
				// GLTR colors
				'gltr-green': 'hsl(var(--gltr-green))',
				'gltr-yellow': 'hsl(var(--gltr-yellow))',
				'gltr-red': 'hsl(var(--gltr-red))',
				'gltr-purple': 'hsl(var(--gltr-purple))',
				
				// Status colors
				success: 'hsl(var(--success))',
				warning: 'hsl(var(--warning))',
				error: 'hsl(var(--error))',
				info: 'hsl(var(--info))',
				primary: {
					DEFAULT: 'hsl(var(--primary))',
					foreground: 'hsl(var(--primary-foreground))'
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
				},
				sidebar: {
					DEFAULT: 'hsl(var(--sidebar-background))',
					foreground: 'hsl(var(--sidebar-foreground))',
					primary: 'hsl(var(--sidebar-primary))',
					'primary-foreground': 'hsl(var(--sidebar-primary-foreground))',
					accent: 'hsl(var(--sidebar-accent))',
					'accent-foreground': 'hsl(var(--sidebar-accent-foreground))',
					border: 'hsl(var(--sidebar-border))',
					ring: 'hsl(var(--sidebar-ring))'
				}
			},
			borderRadius: {
				lg: 'var(--radius)',
				md: 'calc(var(--radius) - 2px)',
				sm: 'calc(var(--radius) - 4px)'
			},
			keyframes: {
				'accordion-down': {
					from: { height: '0', opacity: '0' },
					to: { height: 'var(--radix-accordion-content-height)', opacity: '1' }
				},
				'accordion-up': {
					from: { height: 'var(--radix-accordion-content-height)', opacity: '1' },
					to: { height: '0', opacity: '0' }
				},
				'fade-in': {
					'0%': { opacity: '0', transform: 'translateY(10px)' },
					'100%': { opacity: '1', transform: 'translateY(0)' }
				},
				'fade-out': {
					'0%': { opacity: '1', transform: 'translateY(0)' },
					'100%': { opacity: '0', transform: 'translateY(10px)' }
				},
				'scale-in': {
					'0%': { transform: 'scale(0.95)', opacity: '0' },
					'100%': { transform: 'scale(1)', opacity: '1' }
				},
				'slide-up': {
					'0%': { transform: 'translateY(100%)', opacity: '0' },
					'100%': { transform: 'translateY(0)', opacity: '1' }
				},
				'neural-pulse': {
					'0%, 100%': { opacity: '0.3' },
					'50%': { opacity: '0.8' }
				},
				'count-up': {
					'0%': { transform: 'scale(0.8)', opacity: '0' },
					'100%': { transform: 'scale(1)', opacity: '1' }
				},
				'hover-lift': {
					'0%': { transform: 'translateY(0) scale(1)', boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)' },
					'100%': { transform: 'translateY(-2px) scale(1.02)', boxShadow: '0 10px 25px 0 rgba(59, 130, 246, 0.15)' }
				},
				'gradient-shift': {
					'0%': { backgroundPosition: '0% 50%' },
					'50%': { backgroundPosition: '100% 50%' },
					'100%': { backgroundPosition: '0% 50%' }
				},
				'ripple': {
					'0%': { transform: 'scale(0)', opacity: '1' },
					'100%': { transform: 'scale(4)', opacity: '0' }
				},
				'shimmer': {
					'0%': { transform: 'translateX(-100%)' },
					'100%': { transform: 'translateX(100%)' }
				}
			},
			animation: {
				'accordion-down': 'accordion-down 0.2s ease-out',
				'accordion-up': 'accordion-up 0.2s ease-out',
				'fade-in': 'fade-in 0.3s ease-out',
				'fade-out': 'fade-out 0.3s ease-out', 
				'scale-in': 'scale-in 0.2s ease-out',
				'slide-up': 'slide-up 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
				'neural-pulse': 'neural-pulse 2s ease-in-out infinite',
				'count-up': 'count-up 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275)',
				'hover-lift': 'hover-lift 0.2s ease-out forwards',
				'gradient-shift': 'gradient-shift 3s ease-in-out infinite',
				'ripple': 'ripple 0.6s ease-out',
				'shimmer': 'shimmer 2s infinite'
			}
		}
	},
	// eslint-disable-next-line @typescript-eslint/no-require-imports
	plugins: [require("tailwindcss-animate")],
} satisfies Config;
