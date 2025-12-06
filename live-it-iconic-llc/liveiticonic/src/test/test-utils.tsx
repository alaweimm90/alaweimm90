import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { CartProvider } from '@/contexts/CartContext';
import { HelmetProvider } from 'react-helmet-async';

/**
 * Custom render function that wraps components with necessary providers
 * @param ui Component to render
 * @param options Additional render options
 */
const AllTheProviders = ({ children }: { children: React.ReactNode }) => {
  return (
    <HelmetProvider>
      <BrowserRouter>
        <CartProvider>{children}</CartProvider>
      </BrowserRouter>
    </HelmetProvider>
  );
};

const customRender = (ui: ReactElement, options?: Omit<RenderOptions, 'wrapper'>) =>
  render(ui, { wrapper: AllTheProviders, ...options });

export * from '@testing-library/react';
export { customRender as render };
