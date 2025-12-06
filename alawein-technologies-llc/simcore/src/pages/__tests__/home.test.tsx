import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import Index from '../Index'
import { DomainThemeProvider } from '../../components/DomainThemeProvider'

describe('Home page', () => {
  it('renders hero content', () => {
    // Ensure matchMedia is available in jsdom
    if (!('matchMedia' in window)) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).matchMedia = () => ({
        matches: false,
        media: '',
        onchange: null,
        addListener: () => {},
        removeListener: () => {},
        addEventListener: () => {},
        removeEventListener: () => {},
        dispatchEvent: () => false,
      });
    }
    render(
      <MemoryRouter>
        <DomainThemeProvider defaultDomain="quantum">
          <Index />
        </DomainThemeProvider>
      </MemoryRouter>
    )
    // The hero now renders a progressive loader; assert visible UI instead
    expect(screen.getByText(/Ready/i)).toBeTruthy()
  })
})
