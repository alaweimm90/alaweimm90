import { useEffect, useState } from 'react';

const ScrollProgress = () => {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const updateScrollProgress = () => {
      const scrollPx = document.documentElement.scrollTop;
      const winHeightPx =
        document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const scrolled = (scrollPx / winHeightPx) * 100;

      setScrollProgress(scrolled);
      setIsVisible(scrolled > 5); // Show after 5% scroll
    };

    window.addEventListener('scroll', updateScrollProgress);
    return () => window.removeEventListener('scroll', updateScrollProgress);
  }, []);

  return (
    <div
      className={`scroll-indicator ${isVisible ? 'visible' : ''}`}
      style={{
        background: `linear-gradient(90deg, 
          transparent 0%, 
          hsl(var(--lii-gold)) ${Math.max(0, scrollProgress - 10)}%, 
          hsl(var(--lii-champagne)) ${scrollProgress}%,
          hsl(var(--lii-gold)) ${Math.min(100, scrollProgress + 10)}%, 
          transparent 100%
        )`,
      }}
    />
  );
};

export default ScrollProgress;
