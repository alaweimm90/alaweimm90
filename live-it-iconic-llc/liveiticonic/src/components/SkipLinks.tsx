import { useEffect, useState } from 'react';

const SkipLinks = () => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Show skip links when Tab is pressed from the beginning
      if (e.key === 'Tab' && !isVisible) {
        setIsVisible(true);
      }
    };

    const handleFocus = () => {
      setIsVisible(true);
    };

    const handleBlur = () => {
      // Hide after a short delay to allow focus to move
      setTimeout(() => setIsVisible(false), 100);
    };

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('focusin', handleFocus);
    document.addEventListener('focusout', handleBlur);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('focusin', handleFocus);
      document.removeEventListener('focusout', handleBlur);
    };
  }, [isVisible]);

  const skipLinks = [
    { href: '#main-content', label: 'Skip to main content' },
    { href: '#navigation', label: 'Skip to navigation' },
    { href: '#collection', label: 'Skip to collection' },
    { href: '#connect', label: 'Skip to contact' },
  ];

  return (
    <div
      className={`fixed top-0 left-0 z-[9999] transition-transform duration-200 ${
        isVisible ? 'translate-y-0' : '-translate-y-full'
      }`}
      role="navigation"
      aria-label="Skip links"
    >
      <div className="flex gap-2 p-2 bg-lii-bg/95 backdrop-blur-sm border-b border-lii-gold/20">
        {skipLinks.map(link => (
          <a
            key={link.href}
            href={link.href}
            className="px-4 py-2 text-sm font-ui text-lii-cloud bg-lii-gold hover:bg-lii-gold-press focus:bg-lii-gold-press rounded-lg transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-lii-gold focus:ring-offset-2 focus:ring-offset-lii-bg"
            onFocus={() => setIsVisible(true)}
          >
            {link.label}
          </a>
        ))}
      </div>
    </div>
  );
};

export default SkipLinks;
