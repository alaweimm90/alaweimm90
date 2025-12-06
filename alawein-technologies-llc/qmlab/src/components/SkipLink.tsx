import React from 'react';

interface SkipLinkProps {
  href: string;
  children: React.ReactNode;
  className?: string;
}

export const SkipLink: React.FC<SkipLinkProps> = ({ 
  href, 
  children, 
  className = "" 
}) => {
  return (
    <a
      href={href}
      className={`
        skip-to-main
        transition-all
        ${className}
      `}
      onClick={(e) => {
        e.preventDefault();
        const target = document.querySelector(href);
        if (target) {
          // Set tabindex to make element focusable if it isn't already
          if (!target.hasAttribute('tabindex')) {
            target.setAttribute('tabindex', '-1');
          }
          (target as HTMLElement).focus();
          target.scrollIntoView({ behavior: 'smooth' });
        }
      }}
    >
      {children}
    </a>
  );
};