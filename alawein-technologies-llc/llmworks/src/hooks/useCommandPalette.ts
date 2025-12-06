import { useState, useEffect, useCallback } from 'react';

export const useCommandPalette = () => {
  const [isOpen, setIsOpen] = useState(false);

  const openPalette = useCallback(() => {
    setIsOpen(true);
  }, []);

  const closePalette = useCallback(() => {
    setIsOpen(false);
  }, []);

  const togglePalette = useCallback(() => {
    setIsOpen(prev => !prev);
  }, []);

  // Global keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd+K (Mac) or Ctrl+K (Windows/Linux)
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        togglePalette();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [togglePalette]);

  return {
    isOpen,
    openPalette,
    closePalette,
    togglePalette,
    setIsOpen
  };
};