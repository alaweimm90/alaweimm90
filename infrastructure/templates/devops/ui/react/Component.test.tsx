import { render, screen } from '@testing-library/react';
import { {{COMPONENT_NAME}} } from './Component';

describe('{{COMPONENT_NAME}}', () => {
  it('renders without crashing', () => {
    render(<{{COMPONENT_NAME}} />);
  });

  it('displays title when provided', () => {
    render(<{{COMPONENT_NAME}} title="Test Title" />);
    expect(screen.getByText('Test Title')).toBeInTheDocument();
  });

  it('renders children', () => {
    render(<{{COMPONENT_NAME}}>Child Content</{{COMPONENT_NAME}}>);
    expect(screen.getByText('Child Content')).toBeInTheDocument();
  });
});
