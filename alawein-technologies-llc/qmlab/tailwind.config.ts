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
			colors: {
				border: 'hsl(var(--border))',
				input: 'hsl(var(--input))',
				ring: 'hsl(var(--ring))',
				background: 'hsl(var(--background))',
				foreground: 'hsl(var(--foreground))',
				
				/* Semantic color system */
				surface: {
					1: 'hsl(var(--surface-1))',
					2: 'hsl(var(--surface-2))'
				},
				text: 'hsl(var(--text))',
				
				primary: {
					DEFAULT: 'hsl(var(--primary))',
					foreground: 'hsl(var(--primary-foreground))',
					glow: 'hsl(var(--primary-glow))'
				},
				secondary: {
					DEFAULT: 'hsl(var(--secondary))',
					foreground: 'hsl(var(--secondary-foreground))'
				},
				accent: {
					DEFAULT: 'hsl(var(--accent))',
					foreground: 'hsl(var(--accent-foreground))'
				},
				
				/* Status colors */
				success: 'hsl(var(--success))',
				warn: 'hsl(var(--warn))',
				error: 'hsl(var(--error))',
				info: 'hsl(var(--info))',
				
				/* Quantum state colors */
				state: {
					pure: 'hsl(var(--state-pure))',
					mixed: 'hsl(var(--state-mixed))',
					entangled: 'hsl(var(--state-entangled))'
				},
				
				/* Legacy compatibility */
				destructive: {
					DEFAULT: 'hsl(var(--destructive))',
					foreground: 'hsl(var(--destructive-foreground))'
				},
				muted: {
					DEFAULT: 'hsl(var(--muted))',
					foreground: 'hsl(var(--muted-foreground))',
					light: 'hsl(var(--muted-light))',
					'foreground-light': 'hsl(var(--muted-foreground-light))'
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
			fontFamily: {
				sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
				display: ['Space Grotesk', 'Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
				mono: ['JetBrains Mono', 'Fira Code', 'ui-monospace', 'SFMono-Regular', 'Consolas', 'monospace']
			},
			fontSize: {
				'display': ['3.2rem', { lineHeight: '1.05' }],
				'h1': ['2.4rem', { lineHeight: '1.1' }],
				'h2': ['1.92rem', { lineHeight: '1.2' }],
				'h3': ['1.54rem', { lineHeight: '1.3' }],
				'h4': ['1.23rem', { lineHeight: '1.35' }],
				'body': ['1rem', { lineHeight: '1.6' }],
				'small': ['0.875rem', { lineHeight: '1.5' }]
			},
			backgroundImage: {
				'gradient-quantum': 'var(--gradient-quantum)',
				'gradient-primary': 'var(--gradient-primary)',
				'gradient-secondary': 'var(--gradient-secondary)',
				'gradient-accent': 'var(--gradient-accent)'
			},
			boxShadow: {
				'glow-primary': 'var(--glow-primary)',
				'glow-secondary': 'var(--glow-secondary)',
				'glow-accent': 'var(--glow-accent)',
				'low': 'var(--shadow-low)',
				'high': 'var(--shadow-high)'
			},
			borderRadius: {
				lg: 'var(--radius)',
				md: 'calc(var(--radius) - 2px)',
				sm: 'calc(var(--radius) - 4px)'
			},
			keyframes: {
				'accordion-down': {
					from: { height: '0' },
					to: { height: 'var(--radix-accordion-content-height)' }
				},
				'accordion-up': {
					from: { height: 'var(--radix-accordion-content-height)' },
					to: { height: '0' }
				},
				'quantum-pulse': {
					'0%, 100%': { opacity: '1', transform: 'scale(1)' },
					'50%': { opacity: '0.8', transform: 'scale(1.05)' }
				},
				'quantum-glow': {
					'0%, 100%': { boxShadow: 'var(--glow-primary)' },
					'50%': { boxShadow: '0 0 30px hsl(208 100% 55% / 0.5), 0 0 60px hsl(208 100% 55% / 0.2)' }
				},
				'float': {
					'0%, 100%': { transform: 'translateY(0px)' },
					'50%': { transform: 'translateY(-10px)' }
				},
				'quantum-glow-subtle': {
					'0%, 100%': { 
						boxShadow: '0 0 20px hsl(var(--primary) / 0.2), 0 0 40px hsl(var(--primary) / 0.1)',
						transform: 'scale(1)'
					},
					'50%': { 
						boxShadow: '0 0 30px hsl(var(--primary) / 0.4), 0 0 60px hsl(var(--primary) / 0.2)',
						transform: 'scale(1.02)'
					}
				},
				'scale-102': {
					'0%': { transform: 'scale(1)' },
					'100%': { transform: 'scale(1.02)' }
				}
			},
			animation: {
				'accordion-down': 'accordion-down 0.2s ease-out',
				'accordion-up': 'accordion-up 0.2s ease-out',
				'quantum-pulse': 'quantum-pulse 2s ease-in-out infinite',
				'quantum-glow': 'quantum-glow 3s ease-in-out infinite',
				'quantum-glow-subtle': 'quantum-glow-subtle 4s ease-in-out infinite',
				'float': 'float 6s ease-in-out infinite',
				'scale-102': 'scale-102 0.2s ease-out forwards'
			}
		}
	},
	plugins: [require("tailwindcss-animate")],
} satisfies Config;
