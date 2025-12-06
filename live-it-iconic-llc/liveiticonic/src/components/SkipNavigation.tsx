/**
 * Skip Navigation Component
 * WCAG 2.4.1: Allows keyboard users to skip repetitive navigation
 * Appears on focus and links to main content
 */

const SkipNavigation = () => {
  return (
    <a
      href="#main-content"
      className={`
        sr-only
        focus:not-sr-only
        focus:absolute
        focus:top-0
        focus:left-0
        focus:z-50
        focus:bg-lii-gold
        focus:text-lii-bg
        focus:px-4
        focus:py-2
        focus:font-bold
        focus:outline-none
        focus:rounded-br-lg
        transition-all
        duration-300
      `}
    >
      Skip to main content
    </a>
  );
};

export default SkipNavigation;
