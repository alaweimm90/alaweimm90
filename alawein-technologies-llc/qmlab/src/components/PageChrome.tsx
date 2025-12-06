import * as React from "react";

export interface PageChromeProps {
  /**
   * Page title for the H1 - should match document title
   */
  title: string;
  /**
   * Optional header content
   */
  header?: React.ReactNode;
  /**
   * Optional navigation content
   */
  navigation?: React.ReactNode;
  /**
   * Main page content
   */
  children: React.ReactNode;
  /**
   * Optional footer content
   */
  footer?: React.ReactNode;
  /**
   * Whether to show the H1 visually or screen-reader only
   */
  showTitle?: boolean;
}

export function PageChrome({ 
  title, 
  header, 
  navigation, 
  children, 
  footer, 
  showTitle = false 
}: PageChromeProps) {
  return (
    <>
      {/* Skip Link - WCAG 2.1 SC 2.4.1 */}
      <a 
        href="#main-content" 
        className="skip-to-main focus:not-sr-only"
      >
        Skip to main content
      </a>
      
      {/* Header Landmark */}
      {header && (
        <header role="banner">
          {header}
        </header>
      )}
      
      {/* Primary Navigation Landmark */}
      {navigation && (
        <nav aria-label="Primary navigation" role="navigation">
          {navigation}
        </nav>
      )}
      
      {/* Main Content Landmark */}
      <main 
        id="main-content" 
        tabIndex={-1}
        role="main"
        aria-label="Main content"
      >
        {/* Required H1 for page hierarchy - WCAG 1.3.1 */}
        <h1 className={showTitle ? "quantum-display text-4xl font-bold text-white mb-8" : "sr-only"}>
          {title}
        </h1>
        {children}
      </main>
      
      {/* Footer Landmark */}
      {footer && (
        <footer role="contentinfo">
          {footer}
        </footer>
      )}
    </>
  );
}