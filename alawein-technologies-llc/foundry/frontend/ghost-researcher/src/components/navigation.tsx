'use client'

import { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import { Menu, X, Search, User, Moon, Sun, LogIn } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useTheme } from 'next-themes'
import { cn } from '@/lib/utils'

const navItems = [
  { href: '/features', label: 'Features' },
  { href: '/pricing', label: 'Pricing' },
  { href: '/research', label: 'Research' },
  { href: '/about', label: 'About' },
  { href: '/blog', label: 'Blog' },
]

export function Navigation() {
  const [isOpen, setIsOpen] = useState(false)
  const pathname = usePathname()
  const { theme, setTheme } = useTheme()

  return (
    <nav className="fixed top-0 z-50 w-full bg-background/80 backdrop-blur-lg border-b">
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
              <Search className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="font-bold text-xl">Ghost Researcher</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  'text-sm font-medium transition-colors hover:text-primary',
                  pathname === item.href
                    ? 'text-primary'
                    : 'text-muted-foreground'
                )}
              >
                {item.label}
              </Link>
            ))}
          </div>

          {/* Desktop Actions */}
          <div className="hidden md:flex items-center space-x-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className="rounded-full"
            >
              {theme === 'dark' ? (
                <Sun className="h-5 w-5" />
              ) : (
                <Moon className="h-5 w-5" />
              )}
            </Button>

            <Link href="/login">
              <Button variant="ghost" className="gap-2">
                <LogIn className="w-4 h-4" />
                Login
              </Button>
            </Link>

            <Link href="/signup">
              <Button>Get Started</Button>
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <button
            className="md:hidden"
            onClick={() => setIsOpen(!isOpen)}
            aria-label="Toggle menu"
          >
            {isOpen ? (
              <X className="h-6 w-6" />
            ) : (
              <Menu className="h-6 w-6" />
            )}
          </button>
        </div>
      </div>

      {/* Mobile Navigation */}
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          className="md:hidden bg-background border-b"
        >
          <div className="container mx-auto px-6 py-4">
            <div className="flex flex-col space-y-4">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    'text-sm font-medium transition-colors hover:text-primary',
                    pathname === item.href
                      ? 'text-primary'
                      : 'text-muted-foreground'
                  )}
                  onClick={() => setIsOpen(false)}
                >
                  {item.label}
                </Link>
              ))}

              <div className="pt-4 border-t flex items-center justify-between">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
                  className="rounded-full"
                >
                  {theme === 'dark' ? (
                    <Sun className="h-5 w-5" />
                  ) : (
                    <Moon className="h-5 w-5" />
                  )}
                </Button>

                <div className="flex gap-2">
                  <Link href="/login">
                    <Button variant="ghost" size="sm">
                      Login
                    </Button>
                  </Link>

                  <Link href="/signup">
                    <Button size="sm">Get Started</Button>
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </nav>
  )
}