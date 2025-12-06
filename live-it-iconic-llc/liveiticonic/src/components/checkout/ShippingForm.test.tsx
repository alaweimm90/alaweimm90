import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@/test/test-utils';
import userEvent from '@testing-library/user-event';
import ShippingForm from './ShippingForm';

describe('ShippingForm Component', () => {
  const mockOnNext = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders shipping form with all fields', () => {
    render(<ShippingForm onNext={mockOnNext} />);

    expect(screen.getByLabelText(/First Name/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Last Name/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Email Address/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Phone Number/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Street Address/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/^City/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/^State/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/ZIP Code/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Country/i)).toBeInTheDocument();
  });

  it('renders form title with truck icon', () => {
    const { container } = render(<ShippingForm onNext={mockOnNext} />);

    const title = screen.getByText('Shipping Information');
    expect(title).toBeInTheDocument();

    const icon = container.querySelector('svg');
    expect(icon).toBeInTheDocument();
  });

  it('displays Continue to Payment button', () => {
    render(<ShippingForm onNext={mockOnNext} />);

    const button = screen.getByRole('button', { name: /Continue to Payment/i });
    expect(button).toBeInTheDocument();
  });

  it('pre-fills form with initialData', () => {
    const initialData = {
      firstName: 'John',
      lastName: 'Doe',
      email: 'john@example.com',
      phone: '555-1234',
      address: '123 Main St',
      city: 'San Francisco',
      state: 'CA',
      zipCode: '94105',
      country: 'United States',
    };

    render(<ShippingForm initialData={initialData} onNext={mockOnNext} />);

    expect(screen.getByDisplayValue('John')).toBeInTheDocument();
    expect(screen.getByDisplayValue('Doe')).toBeInTheDocument();
    expect(screen.getByDisplayValue('john@example.com')).toBeInTheDocument();
    expect(screen.getByDisplayValue('555-1234')).toBeInTheDocument();
    expect(screen.getByDisplayValue('123 Main St')).toBeInTheDocument();
    expect(screen.getByDisplayValue('San Francisco')).toBeInTheDocument();
    expect(screen.getByDisplayValue('CA')).toBeInTheDocument();
    expect(screen.getByDisplayValue('94105')).toBeInTheDocument();
  });

  it('updates form fields on input change', async () => {
    const user = userEvent.setup();
    render(<ShippingForm onNext={mockOnNext} />);

    const firstNameInput = screen.getByLabelText(/First Name/i) as HTMLInputElement;
    const lastNameInput = screen.getByLabelText(/Last Name/i) as HTMLInputElement;

    await user.type(firstNameInput, 'Jane');
    await user.type(lastNameInput, 'Smith');

    expect(firstNameInput.value).toBe('Jane');
    expect(lastNameInput.value).toBe('Smith');
  });

  it('calls onNext with form data when submitted', async () => {
    const user = userEvent.setup();
    render(<ShippingForm onNext={mockOnNext} />);

    const formData = {
      firstName: 'John',
      lastName: 'Doe',
      email: 'john@example.com',
      phone: '555-1234',
      address: '123 Main St',
      city: 'San Francisco',
      state: 'CA',
      zipCode: '94105',
    };

    await user.type(screen.getByLabelText(/First Name/i), formData.firstName);
    await user.type(screen.getByLabelText(/Last Name/i), formData.lastName);
    await user.type(screen.getByLabelText(/Email Address/i), formData.email);
    await user.type(screen.getByLabelText(/Phone Number/i), formData.phone);
    await user.type(screen.getByLabelText(/Street Address/i), formData.address);
    await user.type(screen.getByLabelText(/^City/i), formData.city);
    await user.type(screen.getByLabelText(/^State/i), formData.state);
    await user.type(screen.getByLabelText(/ZIP Code/i), formData.zipCode);

    const submitButton = screen.getByRole('button', { name: /Continue to Payment/i });
    await user.click(submitButton);

    expect(mockOnNext).toHaveBeenCalledTimes(1);
    expect(mockOnNext).toHaveBeenCalledWith(
      expect.objectContaining({
        firstName: formData.firstName,
        lastName: formData.lastName,
        email: formData.email,
        phone: formData.phone,
        address: formData.address,
        city: formData.city,
        state: formData.state,
        zipCode: formData.zipCode,
        country: 'United States',
      })
    );
  });

  it('sets default country to United States', () => {
    render(<ShippingForm onNext={mockOnNext} />);

    const countryInput = screen.getByDisplayValue('United States') as HTMLInputElement;
    expect(countryInput.value).toBe('United States');
  });

  it('allows changing country field', async () => {
    const user = userEvent.setup();
    render(<ShippingForm onNext={mockOnNext} />);

    const countryInput = screen.getByLabelText(/Country/i) as HTMLInputElement;
    await user.clear(countryInput);
    await user.type(countryInput, 'Canada');

    expect(countryInput.value).toBe('Canada');
  });

  it('marks all fields as required', () => {
    render(<ShippingForm onNext={mockOnNext} />);

    const firstNameInput = screen.getByLabelText(/First Name/i);
    const lastNameInput = screen.getByLabelText(/Last Name/i);
    const emailInput = screen.getByLabelText(/Email Address/i);
    const phoneInput = screen.getByLabelText(/Phone Number/i);
    const addressInput = screen.getByLabelText(/Street Address/i);
    const cityInput = screen.getByLabelText(/^City/i);
    const stateInput = screen.getByLabelText(/^State/i);
    const zipCodeInput = screen.getByLabelText(/ZIP Code/i);
    const countryInput = screen.getByLabelText(/Country/i);

    expect(firstNameInput).toBeRequired();
    expect(lastNameInput).toBeRequired();
    expect(emailInput).toBeRequired();
    expect(phoneInput).toBeRequired();
    expect(addressInput).toBeRequired();
    expect(cityInput).toBeRequired();
    expect(stateInput).toBeRequired();
    expect(zipCodeInput).toBeRequired();
    expect(countryInput).toBeRequired();
  });

  it('validates email field type', () => {
    render(<ShippingForm onNext={mockOnNext} />);

    const emailInput = screen.getByLabelText(/Email Address/i) as HTMLInputElement;
    expect(emailInput.type).toBe('email');
  });

  it('validates phone field type', () => {
    render(<ShippingForm onNext={mockOnNext} />);

    const phoneInput = screen.getByLabelText(/Phone Number/i) as HTMLInputElement;
    expect(phoneInput.type).toBe('tel');
  });

  it('has proper accessibility labels for all fields', () => {
    render(<ShippingForm onNext={mockOnNext} />);

    expect(screen.getByLabelText(/First Name/i)).toHaveAttribute('id', 'firstName');
    expect(screen.getByLabelText(/Last Name/i)).toHaveAttribute('id', 'lastName');
    expect(screen.getByLabelText(/Email Address/i)).toHaveAttribute('id', 'email');
    expect(screen.getByLabelText(/Phone Number/i)).toHaveAttribute('id', 'phone');
    expect(screen.getByLabelText(/Street Address/i)).toHaveAttribute('id', 'address');
    expect(screen.getByLabelText(/^City/i)).toHaveAttribute('id', 'city');
    expect(screen.getByLabelText(/^State/i)).toHaveAttribute('id', 'state');
    expect(screen.getByLabelText(/ZIP Code/i)).toHaveAttribute('id', 'zipCode');
    expect(screen.getByLabelText(/Country/i)).toHaveAttribute('id', 'country');
  });

  it('renders in a card container with proper styling', () => {
    const { container } = render(<ShippingForm onNext={mockOnNext} />);

    const card = container.querySelector('[class*="bg-lii-ink"]');
    expect(card).toBeInTheDocument();
  });

  it('handles rapid field updates', async () => {
    const user = userEvent.setup();
    render(<ShippingForm onNext={mockOnNext} />);

    const firstNameInput = screen.getByLabelText(/First Name/i) as HTMLInputElement;

    await user.type(firstNameInput, 'A');
    await user.type(firstNameInput, 'B');
    await user.type(firstNameInput, 'C');

    expect(firstNameInput.value).toBe('ABC');
  });

  it('submit button has proper styling', () => {
    render(<ShippingForm onNext={mockOnNext} />);

    const submitButton = screen.getByRole('button', { name: /Continue to Payment/i });
    expect(submitButton).toHaveClass('w-full', 'font-medium');
  });

  it('maintains state across multiple input changes', async () => {
    const user = userEvent.setup();
    render(<ShippingForm onNext={mockOnNext} />);

    // Fill in all fields
    await user.type(screen.getByLabelText(/First Name/i), 'John');
    await user.type(screen.getByLabelText(/Last Name/i), 'Doe');
    await user.type(screen.getByLabelText(/Email Address/i), 'john@test.com');
    await user.type(screen.getByLabelText(/Phone Number/i), '555-1234');

    // Verify first field still has value
    const firstNameInput = screen.getByLabelText(/First Name/i) as HTMLInputElement;
    expect(firstNameInput.value).toBe('John');
  });
});
