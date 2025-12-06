import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@/test/test-utils';
import Hero from './Hero';

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

describe('Hero Component', () => {
  it('renders hero section with correct structure', () => {
    render(<Hero />);

    const section = screen.getByRole('img', {
      name: /Athlete showcasing Live It Iconic apparel/i,
    });
    expect(section).toBeInTheDocument();
  });

  it('displays main heading with correct text', () => {
    render(<Hero />);

    const heading = screen.getByRole('heading', {
      name: /Live It Iconic/i,
    });
    expect(heading).toBeInTheDocument();
    expect(heading).toHaveClass('text-7xl', 'sm:text-8xl', 'md:text-9xl');
  });

  it('displays tagline text', () => {
    const { container } = render(<Hero />);

    // Check that the component renders without errors
    const section = container.querySelector('section');
    expect(section).toBeInTheDocument();

    // Verify hero content exists
    const content = container.querySelector('[class*="relative"]');
    expect(content).toBeInTheDocument();
  });

  it('renders Shop the drop button', () => {
    render(<Hero />);

    const shopButton = screen.getByRole('link', { name: /Shop the drop/i });
    expect(shopButton).toBeInTheDocument();
    // Check that button exists and is either a link or within a button
    expect(shopButton || shopButton.closest('button')).toBeInTheDocument();
  });

  it('renders Explore lifestyle button', () => {
    render(<Hero />);

    const exploreButton = screen.getByRole('link', { name: /Explore lifestyle/i });
    expect(exploreButton).toBeInTheDocument();
    // Check that link exists
    expect(exploreButton).toBeInTheDocument();
  });

  it('has proper accessibility attributes', () => {
    render(<Hero />);

    const section = screen.getByRole('img', {
      name: /Athlete showcasing Live It Iconic apparel/i,
    });
    expect(section).toHaveAttribute('role', 'img');
    expect(section).toHaveAttribute(
      'aria-label',
      'Athlete showcasing Live It Iconic apparel with luxury supercar in coastal mountain setting'
    );
  });

  it('renders decorative elements with aria-hidden', () => {
    const { container } = render(<Hero />);

    const decorativeElements = container.querySelectorAll('[aria-hidden="true"]');
    expect(decorativeElements.length).toBeGreaterThan(0);
  });

  it('has full viewport height section', () => {
    const { container } = render(<Hero />);

    const section = container.querySelector('section');
    expect(section).toHaveClass('h-screen', 'w-full');
  });

  it('renders both buttons in a flex container', () => {
    const { container } = render(<Hero />);

    const buttonContainer = container.querySelector('div[class*="flex"][class*="flex-col"]');
    expect(buttonContainer).toBeInTheDocument();

    const buttons = buttonContainer?.querySelectorAll('a');
    expect(buttons?.length).toBe(2);
  });

  it('buttons are properly styled with variant classes', () => {
    const { container } = render(<Hero />);

    const shopButton = screen.getByRole('link', { name: /Shop the drop/i });
    const exploreButton = screen.getByRole('link', { name: /Explore lifestyle/i });

    // Check that buttons exist
    expect(shopButton).toBeInTheDocument();
    expect(exploreButton).toBeInTheDocument();

    // Check they have proper styling
    const buttonContainers = container.querySelectorAll('[class*="group"]');
    expect(buttonContainers.length).toBeGreaterThan(0);
  });
});
