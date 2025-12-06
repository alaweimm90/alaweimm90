import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@/test/test-utils';
import userEvent from '@testing-library/user-event';
import Navigation from './Navigation';

// Mock child components
vi.mock('./CartIcon', () => ({
  default: ({ onClick }: any) => (
    <button onClick={onClick} aria-label="Open cart">
      Cart Icon
    </button>
  ),
}));

vi.mock('./CartDrawer', () => ({
  default: ({ isOpen, onClose }: any) => (
    isOpen ? <div role="dialog">Cart Drawer</div> : null
  ),
}));

vi.mock('./BirdLogoShowcase', () => ({
  default: () => <div>Bird Showcase</div>,
}));

vi.mock('./icons/DiamondLogo', () => ({
  default: () => <div data-testid="diamond-logo">Diamond Logo</div>,
}));

// Mock the router
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    Link: ({ to, children, ...props }: any) => (
      <a href={to} {...props}>
        {children}
      </a>
    ),
  };
});

describe('Navigation Component', () => {
  beforeEach(() => {
    // Reset window scroll position
    window.scrollY = 0;
  });

  it('renders navigation bar with logo', () => {
    render(<Navigation />);

    const logo = screen.getByTestId('diamond-logo');
    expect(logo).toBeInTheDocument();
  });

  it('displays all navigation links on desktop', () => {
    render(<Navigation />);

    const navItems = ['About', 'Lifestyle', 'Collection', 'Brand', 'Launch', 'Contact'];

    navItems.forEach(item => {
      const link = screen.getAllByText(item).find(el => el.tagName === 'A');
      expect(link).toBeInTheDocument();
    });
  });

  it('renders cart icon button', () => {
    render(<Navigation />);

    const cartButton = screen.getByLabelText('Open cart');
    expect(cartButton).toBeInTheDocument();
  });

  it('opens mobile menu when hamburger button is clicked', async () => {
    const user = userEvent.setup();
    render(<Navigation />);

    const menuButton = screen.getByLabelText('Open navigation menu');
    await user.click(menuButton);

    // Check if menu is opened by looking for aria-expanded
    expect(menuButton).toHaveAttribute('aria-expanded', 'true');
  });

  it('closes mobile menu when close button is clicked', async () => {
    const user = userEvent.setup();
    render(<Navigation />);

    const menuButton = screen.getByLabelText('Open navigation menu');
    await user.click(menuButton);
    expect(menuButton).toHaveAttribute('aria-expanded', 'true');

    const closeButton = screen.getByLabelText('Close navigation menu');
    await user.click(closeButton);
    expect(closeButton).toHaveAttribute('aria-expanded', 'false');
  });

  it('closes mobile menu when Escape key is pressed', async () => {
    const user = userEvent.setup();
    render(<Navigation />);

    const menuButton = screen.getByLabelText('Open navigation menu');
    await user.click(menuButton);
    expect(menuButton).toHaveAttribute('aria-expanded', 'true');

    fireEvent.keyDown(window, { key: 'Escape', code: 'Escape' });

    expect(menuButton).toHaveAttribute('aria-expanded', 'false');
  });

  it('opens cart drawer when cart icon is clicked', async () => {
    const user = userEvent.setup();
    render(<Navigation />);

    const cartButton = screen.getByLabelText('Open cart');
    await user.click(cartButton);

    await waitFor(() => {
      const cartDrawer = screen.getByRole('dialog');
      expect(cartDrawer).toBeInTheDocument();
    });
  });

  it('updates navigation styling when scrolled', () => {
    const { container } = render(<Navigation />);

    const nav = container.querySelector('nav#navigation');
    expect(nav).toHaveClass('backdrop-blur-xl');

    // Simulate scroll
    fireEvent.scroll(window, { y: 50 });

    expect(nav).toHaveClass('backdrop-blur-2xl');
  });

  it('renders logo link with correct href', () => {
    render(<Navigation />);

    const logoLink = screen.getByLabelText('Live It Iconic Home');
    expect(logoLink).toHaveAttribute('href', '/');
  });

  it('focuses first navigation link when mobile menu opens', async () => {
    const user = userEvent.setup();
    render(<Navigation />);

    const menuButton = screen.getByLabelText('Open navigation menu');
    await user.click(menuButton);

    // The first nav link should be focused
    const firstLink = screen.getAllByText('About')[0].closest('a');
    await waitFor(() => {
      expect(firstLink).toHaveFocus();
    });
  });

  it('closes mobile menu when a navigation link is clicked', async () => {
    const user = userEvent.setup();
    render(<Navigation />);

    const menuButton = screen.getByLabelText('Open navigation menu');
    await user.click(menuButton);

    const aboutLink = screen.getAllByText('About')[0].closest('a') as HTMLAnchorElement;
    await user.click(aboutLink);

    expect(menuButton).toHaveAttribute('aria-expanded', 'false');
  });

  it('has proper accessibility attributes', () => {
    render(<Navigation />);

    const nav = screen.getByRole('navigation', { name: /Primary navigation/i });
    expect(nav).toBeInTheDocument();
  });

  it('renders navigation with id attribute for testing', () => {
    const { container } = render(<Navigation />);
    const nav = container.querySelector('nav#navigation');
    expect(nav).toBeInTheDocument();
  });

  it('has min-h of 44px for touch targets', () => {
    render(<Navigation />);

    const cartButton = screen.getByLabelText('Open cart');
    expect(cartButton).toHaveClass('min-w-[44px]', 'min-h-[44px]');
  });
});
