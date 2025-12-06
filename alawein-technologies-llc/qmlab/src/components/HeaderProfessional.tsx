import React from 'react';
import { Button } from '@/components/ui/button';
import { IconButton } from '@/components/ui/icon-button';
import { Search, Menu, X } from 'lucide-react';
import { useState } from 'react';

export const HeaderProfessional: React.FC = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    {
      label: 'Laboratory',
      href: '#lab-section',
      ariaLabel: 'Navigate to Interactive Quantum Laboratory section'
    },
    {
      label: 'Learning',
      href: '#learning-resources',
      ariaLabel: 'Navigate to Quantum Learning Resources section'
    },
    {
      label: 'Features',
      href: '#advanced-features',
      ariaLabel: 'Navigate to Advanced Features section'
    },
    {
      label: 'Progress',
      href: '#learning-journey',
      ariaLabel: 'Navigate to Learning Journey section'
    },
  ];

  return (
    <header className="fixed top-0 w-full z-50 bg-slate-950/80 backdrop-blur-md border-b border-slate-800">
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-8">
            <a href="/" className="flex items-center gap-3">
              <div className="w-8 h-8 rounded bg-white flex items-center justify-center">
                <span className="text-slate-900 font-bold text-sm">Q</span>
              </div>
              <span className="text-white font-semibold text-lg">QMLab</span>
            </a>

            {/* Desktop Navigation */}
            <nav className="hidden md:flex items-center gap-6" aria-label="Main navigation">
              {navItems.map((item) => (
                <a
                  key={item.label}
                  href={item.href}
                  className="text-sm text-slate-400 hover:text-white transition-colors focus-visible:outline-2 focus-visible:outline-primary focus-visible:outline-offset-2 rounded px-2 py-1"
                  aria-label={item.ariaLabel}
                >
                  {item.label}
                </a>
              ))}
            </nav>
          </div>

          {/* Right Actions */}
          <div className="flex items-center gap-4">
            {/* Search Button */}
            <button
              className="hidden md:flex items-center gap-2 px-4 py-2 text-sm text-slate-400 hover:text-white border border-slate-700 rounded-lg hover:border-slate-600 transition-all hover:bg-slate-800/50 min-h-[44px] focus-visible:outline-2 focus-visible:outline-primary focus-visible:outline-offset-2 rounded"
              onClick={() => {
                const event = new KeyboardEvent('keydown', {
                  key: 'k',
                  metaKey: true,
                });
                document.dispatchEvent(event);
              }}
              aria-label="Search quantum concepts and tutorials (Press Command+K or Control+K)"
            >
              <Search className="w-4 h-4" aria-hidden="true" />
              <span className="hidden sm:inline">Search</span>
              <kbd className="hidden md:inline-flex items-center gap-1 px-1.5 py-0.5 text-xs bg-slate-800 rounded border border-slate-700 text-slate-400">
                <span className="text-[10px]">⌘</span>K
              </kbd>
            </button>

            {/* GitHub Link */}
            <a
              href="https://github.com/alaweimm90/qml-playground"
              target="_blank"
              rel="noopener noreferrer"
              className="hidden md:block text-sm text-slate-400 hover:text-white transition-colors min-h-[44px] flex items-center"
              aria-label="View source code on GitHub (opens in new tab)"
            >
              GitHub
            </a>

            {/* Mobile Menu Toggle */}
            <IconButton
              label={mobileMenuOpen ? "Close navigation menu" : "Open navigation menu"}
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden"
              aria-expanded={mobileMenuOpen}
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </IconButton>
          </div>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <nav className="md:hidden py-4 border-t border-slate-800">
            {navItems.map((item) => (
              <a
                key={item.label}
                href={item.href}
                className="block py-2 text-sm text-slate-400 hover:text-white transition-colors"
                onClick={() => setMobileMenuOpen(false)}
              >
                {item.label}
              </a>
            ))}
            <div className="mt-4 pt-4 border-t border-slate-800">
              <a
                href="https://github.com/alaweimm90/qml-playground"
                target="_blank"
                rel="noopener noreferrer"
                className="block py-2 text-sm text-slate-400 hover:text-white transition-colors"
              >
                GitHub
              </a>
            </div>
          </nav>
        )}
      </div>
    </header>
  );
};
