/**
 * Skip-to-content link for keyboard users
 * Add to the top of your layout component
 */
export function SkipToContent() {
  return (
    <a
      href="#main-content"
      className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-primary focus:text-white focus:rounded"
    >
      Skip to main content
    </a>
  );
}

// Usage in layout:
// <body>
//   <SkipToContent />
//   <nav>...</nav>
//   <main id="main-content">...</main>
// </body>
