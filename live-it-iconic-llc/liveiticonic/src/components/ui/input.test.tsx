import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@/test/test-utils';
import userEvent from '@testing-library/user-event';
import { Input } from './input';

describe('Input Component', () => {
  it('renders input element', () => {
    render(<Input placeholder="Enter text" />);

    const input = screen.getByPlaceholderText('Enter text');
    expect(input).toBeInTheDocument();
    expect(input).toBeInstanceOf(HTMLInputElement);
  });

  it('handles text input', async () => {
    const user = userEvent.setup();
    render(<Input placeholder="Enter text" />);

    const input = screen.getByPlaceholderText('Enter text') as HTMLInputElement;
    await user.type(input, 'Hello World');

    expect(input.value).toBe('Hello World');
  });

  it('supports type attribute', () => {
    render(<Input type="email" placeholder="Enter email" />);

    const input = screen.getByPlaceholderText('Enter email') as HTMLInputElement;
    expect(input.type).toBe('email');
  });

  it('supports password type', () => {
    render(<Input type="password" placeholder="Enter password" />);

    const input = screen.getByPlaceholderText('Enter password') as HTMLInputElement;
    expect(input.type).toBe('password');
  });

  it('supports number type', () => {
    render(<Input type="number" placeholder="Enter number" />);

    const input = screen.getByPlaceholderText('Enter number') as HTMLInputElement;
    expect(input.type).toBe('number');
  });

  it('renders disabled input', () => {
    render(<Input placeholder="Disabled" disabled />);

    const input = screen.getByPlaceholderText('Disabled') as HTMLInputElement;
    expect(input).toBeDisabled();
  });

  it('renders readonly input', () => {
    render(<Input value="Read only" readOnly />);

    const input = screen.getByDisplayValue('Read only') as HTMLInputElement;
    expect(input).toHaveAttribute('readonly');
  });

  it('handles change events', async () => {
    const handleChange = vi.fn();
    const user = userEvent.setup();

    render(<Input placeholder="Change test" onChange={handleChange} />);

    const input = screen.getByPlaceholderText('Change test');
    await user.type(input, 'test');

    expect(handleChange).toHaveBeenCalled();
  });

  it('handles focus events', async () => {
    const handleFocus = vi.fn();
    const user = userEvent.setup();

    render(<Input placeholder="Focus test" onFocus={handleFocus} />);

    const input = screen.getByPlaceholderText('Focus test');
    await user.click(input);

    expect(handleFocus).toHaveBeenCalled();
  });

  it('handles blur events', async () => {
    const handleBlur = vi.fn();
    const user = userEvent.setup();

    render(
      <>
        <Input placeholder="Blur test" onBlur={handleBlur} />
        <button>Next</button>
      </>
    );

    const input = screen.getByPlaceholderText('Blur test');
    const button = screen.getByRole('button');

    await user.click(input);
    await user.click(button);

    expect(handleBlur).toHaveBeenCalled();
  });

  it('supports className prop', () => {
    const { container } = render(
      <Input placeholder="Custom class" className="custom-input" />
    );

    const input = screen.getByPlaceholderText('Custom class');
    expect(input).toHaveClass('custom-input');
  });

  it('supports aria attributes', () => {
    render(
      <Input
        placeholder="Aria test"
        aria-label="Username"
        aria-required="true"
      />
    );

    const input = screen.getByLabelText('Username');
    expect(input).toHaveAttribute('aria-required', 'true');
  });

  it('displays validation error style', () => {
    const { container } = render(
      <Input placeholder="Error" aria-invalid="true" />
    );

    const input = screen.getByPlaceholderText('Error');
    expect(input).toHaveAttribute('aria-invalid', 'true');
  });

  it('handles maxLength attribute', async () => {
    const user = userEvent.setup();
    render(<Input placeholder="Max length" maxLength={5} />);

    const input = screen.getByPlaceholderText('Max length') as HTMLInputElement;
    await user.type(input, 'Hello World');

    // The input should only accept up to maxLength characters
    expect(input.value.length).toBeLessThanOrEqual(5);
  });

  it('handles minLength attribute', () => {
    render(<Input placeholder="Min length" minLength={3} />);

    const input = screen.getByPlaceholderText('Min length') as HTMLInputElement;
    expect(input).toHaveAttribute('minlength', '3');
  });

  it('supports required attribute', () => {
    render(<Input placeholder="Required" required />);

    const input = screen.getByPlaceholderText('Required') as HTMLInputElement;
    expect(input).toBeRequired();
  });

  it('handles placeholder attribute', () => {
    render(<Input placeholder="Enter your name" />);

    const input = screen.getByPlaceholderText('Enter your name');
    expect(input).toHaveAttribute('placeholder', 'Enter your name');
  });

  it('handles default value', () => {
    render(<Input defaultValue="Default text" />);

    const input = screen.getByDisplayValue('Default text') as HTMLInputElement;
    expect(input.value).toBe('Default text');
  });

  it('supports pattern validation', () => {
    render(<Input placeholder="Pattern" pattern="[0-9]{3}" />);

    const input = screen.getByPlaceholderText('Pattern');
    expect(input).toHaveAttribute('pattern', '[0-9]{3}');
  });

  it('renders with focus state', async () => {
    const user = userEvent.setup();
    const { container } = render(<Input placeholder="Focus state" />);

    const input = screen.getByPlaceholderText('Focus state');
    await user.click(input);

    expect(input).toHaveFocus();
  });

  it('has proper touch target size', () => {
    const { container } = render(<Input placeholder="Touch target" />);

    const input = screen.getByPlaceholderText('Touch target') as HTMLInputElement;
    // Check that input has height class
    expect(input.className).toMatch(/h-\d+/);
    expect(input.className).toMatch(/px-\d+/);
  });

  it('has focus ring styles', () => {
    const { container } = render(<Input placeholder="Focus ring" />);

    const input = screen.getByPlaceholderText('Focus ring');
    // Check that it has focus styling in className
    expect(input.className).toMatch(/focus/);
  });

  it('supports autocomplete attribute', () => {
    render(<Input type="email" placeholder="Email" autoComplete="email" />);

    const input = screen.getByPlaceholderText('Email');
    expect(input).toHaveAttribute('autocomplete', 'email');
  });

  it('handles multiple input types', () => {
    const { container } = render(
      <>
        <Input type="text" placeholder="Text" />
        <Input type="email" placeholder="Email" />
        <Input type="password" placeholder="Password" />
        <Input type="number" placeholder="Number" />
      </>
    );

    expect(screen.getByPlaceholderText('Text')).toHaveAttribute('type', 'text');
    expect(screen.getByPlaceholderText('Email')).toHaveAttribute('type', 'email');
    expect(screen.getByPlaceholderText('Password')).toHaveAttribute('type', 'password');
    expect(screen.getByPlaceholderText('Number')).toHaveAttribute('type', 'number');
  });
});
