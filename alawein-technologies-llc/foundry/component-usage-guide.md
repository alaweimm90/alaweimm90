# Component Usage Guide

## Complete guide for using the Foundry Design System components

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Button Component](#button-component)
3. [Input Component](#input-component)
4. [Select Component](#select-component)
5. [Card Component](#card-component)
6. [Modal Component](#modal-component)
7. [Table Component](#table-component)
8. [Toast Component](#toast-component)
9. [Badge Component](#badge-component)
10. [Spinner Component](#spinner-component)
11. [Theme Configuration](#theme-configuration)
12. [Animation Usage](#animation-usage)

---

## Getting Started

### Installation

```bash
# Install required dependencies
npm install react react-dom framer-motion lucide-react tailwindcss

# Install the design system (if published as package)
npm install @Foundry/design-system
```

### Basic Setup

```tsx
// Import styles and components
import '@Foundry/design-system/styles.css';
import { Button, Card, Input, ToastProvider } from '@Foundry/design-system';

// Wrap your app with providers
function App() {
  return (
    <ToastProvider position="top-right">
      <YourApp />
    </ToastProvider>
  );
}
```

---

## Button Component

### Basic Usage

```tsx
import { Button } from '@Foundry/design-system';

// Simple button
<Button>Click me</Button>

// With variant
<Button variant="primary">Primary</Button>
<Button variant="secondary">Secondary</Button>
<Button variant="success">Success</Button>
<Button variant="danger">Danger</Button>
<Button variant="warning">Warning</Button>
<Button variant="ghost">Ghost</Button>
<Button variant="outline">Outline</Button>
<Button variant="link">Link</Button>
```

### Sizes

```tsx
<Button size="xs">Extra Small</Button>
<Button size="sm">Small</Button>
<Button size="md">Medium (Default)</Button>
<Button size="lg">Large</Button>
<Button size="xl">Extra Large</Button>
```

### With Icons

```tsx
import { Download, ArrowRight } from 'lucide-react';

<Button leftIcon={<Download className="w-4 h-4" />}>
  Download
</Button>

<Button rightIcon={<ArrowRight className="w-4 h-4" />}>
  Continue
</Button>
```

### Loading State

```tsx
<Button loading loadingText="Processing...">
  Submit
</Button>

// Loading without text change
<Button loading>
  Submit
</Button>
```

### Button Group

```tsx
import { ButtonGroup } from '@Foundry/design-system';

<ButtonGroup>
  <Button>One</Button>
  <Button>Two</Button>
  <Button>Three</Button>
</ButtonGroup>

// Vertical group
<ButtonGroup direction="vertical">
  <Button>Top</Button>
  <Button>Middle</Button>
  <Button>Bottom</Button>
</ButtonGroup>
```

### Advanced Examples

```tsx
// Full width button
<Button fullWidth>Full Width Button</Button>

// Disabled state
<Button disabled>Disabled</Button>

// Custom styling
<Button className="custom-class" rounded="full">
  Rounded Button
</Button>

// With onClick handler
<Button
  onClick={() => console.log('Clicked!')}
  variant="primary"
  size="lg"
>
  Click Me
</Button>
```

---

## Input Component

### Basic Usage

```tsx
import { Input } from '@Foundry/design-system';

// Simple input
<Input placeholder="Enter text..." />

// With label
<Input label="Email Address" type="email" />

// Required field
<Input label="Username" required />
```

### Variants

```tsx
<Input variant="outline" placeholder="Outline (Default)" />
<Input variant="filled" placeholder="Filled" />
<Input variant="flushed" placeholder="Flushed" />
<Input variant="unstyled" placeholder="Unstyled" />
```

### States

```tsx
// Error state
<Input
  error="Please enter a valid email"
  value={email}
  onChange={(e) => setEmail(e.target.value)}
/>

// Success state
<Input
  success="Email available!"
  value={email}
/>

// Warning state
<Input
  warning="This username is already popular"
  value={username}
/>

// Info state
<Input
  info="Use 8+ characters for security"
  type="password"
/>

// Helper text
<Input
  helperText="We'll never share your email"
  type="email"
/>
```

### With Icons

```tsx
import { Search, Mail } from 'lucide-react';

// Left icon
<Input
  leftIcon={<Mail className="w-5 h-5" />}
  placeholder="Email"
/>

// Right icon
<Input
  rightIcon={<Search className="w-5 h-5" />}
  placeholder="Search..."
/>
```

### Special Features

```tsx
// Password input with show/hide toggle
<Input type="password" label="Password" />

// Clearable input
<Input
  clearable
  value={value}
  onChange={(e) => setValue(e.target.value)}
  onClear={() => setValue('')}
/>

// Loading state
<Input loading placeholder="Loading..." />

// Different sizes
<Input size="xs" placeholder="Extra Small" />
<Input size="sm" placeholder="Small" />
<Input size="md" placeholder="Medium" />
<Input size="lg" placeholder="Large" />
<Input size="xl" placeholder="Extra Large" />
```

---

## Select Component

### Basic Usage

```tsx
import { Select } from '@Foundry/design-system';

const options = [
  { value: '1', label: 'Option 1' },
  { value: '2', label: 'Option 2' },
  { value: '3', label: 'Option 3' },
];

<Select
  options={options}
  placeholder="Select an option"
  onChange={(value) => console.log(value)}
/>
```

### Multiple Selection

```tsx
<Select
  multiple
  options={options}
  placeholder="Select multiple options"
  onChange={(values) => console.log(values)}
/>
```

### Searchable Select

```tsx
<Select
  searchable
  options={options}
  placeholder="Type to search..."
/>
```

### Grouped Options

```tsx
const groupedOptions = [
  { value: 'apple', label: 'Apple', group: 'Fruits' },
  { value: 'banana', label: 'Banana', group: 'Fruits' },
  { value: 'carrot', label: 'Carrot', group: 'Vegetables' },
  { value: 'broccoli', label: 'Broccoli', group: 'Vegetables' },
];

<Select
  options={groupedOptions}
  placeholder="Select item"
/>
```

### With Icons

```tsx
import { User, Settings, Bell } from 'lucide-react';

const iconOptions = [
  { value: 'user', label: 'Profile', icon: <User className="w-4 h-4" /> },
  { value: 'settings', label: 'Settings', icon: <Settings className="w-4 h-4" /> },
  { value: 'notifications', label: 'Notifications', icon: <Bell className="w-4 h-4" /> },
];

<Select options={iconOptions} placeholder="Select action" />
```

### Advanced Features

```tsx
// Clearable select
<Select
  clearable
  options={options}
  value={selectedValue}
  onChange={setSelectedValue}
/>

// Custom rendering
<Select
  options={options}
  renderOption={(option, isSelected) => (
    <div className="flex items-center justify-between">
      <span>{option.label}</span>
      {isSelected && <Check className="w-4 h-4" />}
    </div>
  )}
  renderValue={(value, options) => {
    const selected = options.find(o => o.value === value);
    return <strong>{selected?.label}</strong>;
  }}
/>

// With validation
<Select
  options={options}
  error="Please select an option"
  required
  label="Category"
/>
```

---

## Card Component

### Basic Usage

```tsx
import { Card, CardHeader, CardBody, CardFooter } from '@Foundry/design-system';

<Card>
  <CardHeader title="Card Title" subtitle="Card subtitle" />
  <CardBody>
    <p>Card content goes here</p>
  </CardBody>
  <CardFooter>
    <Button>Action</Button>
  </CardFooter>
</Card>
```

### Card Variants

```tsx
// Elevated (default)
<Card variant="elevated">
  <CardBody>Elevated card with shadow</CardBody>
</Card>

// Outlined
<Card variant="outlined">
  <CardBody>Outlined card with border</CardBody>
</Card>

// Filled
<Card variant="filled" bgColor="gray">
  <CardBody>Filled card with background</CardBody>
</Card>

// Ghost
<Card variant="ghost">
  <CardBody>Transparent ghost card</CardBody>
</Card>
```

### Interactive Cards

```tsx
// Hoverable card
<Card hoverable>
  <CardBody>Hover me for effect</CardBody>
</Card>

// Clickable card
<Card
  clickable
  onClick={() => console.log('Card clicked')}
>
  <CardBody>Click me</CardBody>
</Card>
```

### Advanced Layouts

```tsx
// Card with custom header action
<Card>
  <CardHeader
    title="Users"
    subtitle="Manage your team"
    action={
      <Button size="sm" variant="ghost">
        Add User
      </Button>
    }
  />
  <CardBody>
    <UserList />
  </CardBody>
</Card>

// Card with bordered sections
<Card>
  <CardHeader title="Settings" bordered />
  <CardBody>
    <SettingsForm />
  </CardBody>
  <CardFooter bordered align="between">
    <Button variant="ghost">Cancel</Button>
    <Button variant="primary">Save Changes</Button>
  </CardFooter>
</Card>
```

---

## Modal Component

### Basic Usage

```tsx
import { Modal } from '@Foundry/design-system';

const [isOpen, setIsOpen] = useState(false);

<>
  <Button onClick={() => setIsOpen(true)}>Open Modal</Button>

  <Modal
    isOpen={isOpen}
    onClose={() => setIsOpen(false)}
    title="Modal Title"
    description="Modal description text"
  >
    <p>Modal content goes here</p>
  </Modal>
</>
```

### Modal Sizes

```tsx
<Modal size="xs">Extra small modal</Modal>
<Modal size="sm">Small modal</Modal>
<Modal size="md">Medium modal (default)</Modal>
<Modal size="lg">Large modal</Modal>
<Modal size="xl">Extra large modal</Modal>
<Modal size="full">Full screen modal</Modal>
```

### With Footer Actions

```tsx
<Modal
  isOpen={isOpen}
  onClose={() => setIsOpen(false)}
  title="Confirm Action"
  footer={
    <div className="flex gap-3 justify-end">
      <Button variant="ghost" onClick={() => setIsOpen(false)}>
        Cancel
      </Button>
      <Button variant="primary" onClick={handleConfirm}>
        Confirm
      </Button>
    </div>
  }
>
  <p>Are you sure you want to continue?</p>
</Modal>
```

### Confirm Modal

```tsx
import { ConfirmModal } from '@Foundry/design-system';

<ConfirmModal
  isOpen={isOpen}
  onClose={() => setIsOpen(false)}
  title="Delete Item"
  message="Are you sure you want to delete this item? This action cannot be undone."
  confirmText="Delete"
  cancelText="Cancel"
  confirmVariant="danger"
  onConfirm={handleDelete}
  loading={isDeleting}
/>
```

### Advanced Options

```tsx
<Modal
  isOpen={isOpen}
  onClose={handleClose}
  title="Advanced Modal"
  // Positioning
  centered={false}  // Top-aligned instead of centered

  // Behavior
  closeOnOverlayClick={false}  // Prevent closing on overlay click
  closeOnEsc={false}  // Prevent closing on Escape key
  showCloseButton={false}  // Hide X button

  // Styling
  animation="slide"  // Animation type: fade, slide, scale, none
  overlayBlur={true}  // Blur background
  zIndex={2000}  // Custom z-index

  // Scrolling
  preventScroll={true}  // Prevent body scroll when open
>
  <LongContent />
</Modal>
```

---

## Table Component

### Basic Usage

```tsx
import { Table } from '@Foundry/design-system';

const columns = [
  { key: 'id', label: 'ID', sortable: true },
  { key: 'name', label: 'Name', sortable: true },
  { key: 'email', label: 'Email' },
  { key: 'role', label: 'Role' },
  { key: 'status', label: 'Status' },
];

const data = [
  { id: 1, name: 'John Doe', email: 'john@example.com', role: 'Admin', status: 'Active' },
  { id: 2, name: 'Jane Smith', email: 'jane@example.com', role: 'User', status: 'Active' },
  // ... more data
];

<Table columns={columns} data={data} />
```

### With Pagination

```tsx
<Table
  columns={columns}
  data={data}
  paginated
  pageSize={10}
  pageSizeOptions={[5, 10, 25, 50]}
/>
```

### With Selection

```tsx
const [selectedRows, setSelectedRows] = useState(new Set());

<Table
  columns={columns}
  data={data}
  selectable
  selectedRows={selectedRows}
  onSelectionChange={setSelectedRows}
/>
```

### Custom Rendering

```tsx
const columns = [
  {
    key: 'name',
    label: 'Name',
    render: (value, row) => (
      <div className="flex items-center gap-2">
        <Avatar src={row.avatar} />
        <span className="font-medium">{value}</span>
      </div>
    ),
  },
  {
    key: 'status',
    label: 'Status',
    render: (value) => (
      <Badge variant={value === 'Active' ? 'success' : 'danger'}>
        {value}
      </Badge>
    ),
  },
  {
    key: 'actions',
    label: 'Actions',
    align: 'right',
    render: (_, row) => (
      <div className="flex gap-2">
        <Button size="sm" variant="ghost" onClick={() => handleEdit(row)}>
          Edit
        </Button>
        <Button size="sm" variant="ghost" onClick={() => handleDelete(row)}>
          Delete
        </Button>
      </div>
    ),
  },
];
```

### Advanced Features

```tsx
<Table
  columns={columns}
  data={data}

  // Styling
  striped  // Alternating row colors
  hoverable  // Hover effect on rows
  bordered  // Add borders
  compact  // Reduce padding

  // Behavior
  sortable  // Enable sorting
  filterable  // Enable filtering
  exportable  // Enable CSV export
  exportFilename="users-data"

  // Layout
  stickyHeader  // Sticky header on scroll
  maxHeight="500px"  // Scrollable table

  // Events
  onRowClick={(row, index) => console.log('Clicked row:', row)}

  // Loading & Empty states
  loading={isLoading}
  emptyMessage="No users found"
/>
```

---

## Toast Component

### Setup

```tsx
// Wrap your app with ToastProvider
import { ToastProvider } from '@Foundry/design-system';

function App() {
  return (
    <ToastProvider position="top-right" maxToasts={5}>
      <YourApp />
    </ToastProvider>
  );
}
```

### Using the Hook

```tsx
import { useToast } from '@Foundry/design-system';

function MyComponent() {
  const { showToast, success, error, warning, info } = useToast();

  // Basic toast
  showToast({
    message: 'Operation completed',
    variant: 'default',
    duration: 5000,
  });

  // Convenience methods
  success('Changes saved successfully!');
  error('Failed to save changes');
  warning('Your session will expire soon');
  info('New update available');

  // With title
  showToast({
    title: 'Success',
    message: 'Your profile has been updated',
    variant: 'success',
  });

  // With action
  showToast({
    message: 'File uploaded',
    action: {
      label: 'View',
      onClick: () => openFile(),
    },
  });

  // Persistent toast (won't auto-dismiss)
  showToast({
    message: 'Download complete',
    duration: 0,  // Never auto-dismiss
    closable: true,
  });
}
```

### Toast Positions

```tsx
// Available positions
type ToastPosition =
  | 'top-left'
  | 'top-center'
  | 'top-right'
  | 'bottom-left'
  | 'bottom-center'
  | 'bottom-right';

// Change position dynamically
const { setPosition } = useToast();
setPosition('bottom-center');
```

---

## Badge Component

### Basic Usage

```tsx
import { Badge } from '@Foundry/design-system';

<Badge>Default</Badge>
<Badge variant="primary">Primary</Badge>
<Badge variant="secondary">Secondary</Badge>
<Badge variant="success">Success</Badge>
<Badge variant="warning">Warning</Badge>
<Badge variant="danger">Danger</Badge>
<Badge variant="info">Info</Badge>
```

### Styles

```tsx
// Solid style
<Badge style="solid" variant="primary">Solid</Badge>

// Subtle style (default)
<Badge style="subtle" variant="primary">Subtle</Badge>

// Outline style
<Badge style="outline" variant="primary">Outline</Badge>
```

### With Features

```tsx
// With dot indicator
<Badge dot>Online</Badge>
<Badge dot dotColor="#10b981">Available</Badge>

// With icon
<Badge icon={<CheckIcon className="w-3 h-3" />}>
  Verified
</Badge>

// Removable badge
<Badge
  removable
  onRemove={() => console.log('Removed')}
>
  Tag
</Badge>

// Different sizes
<Badge size="xs">XS</Badge>
<Badge size="sm">SM</Badge>
<Badge size="md">MD</Badge>
<Badge size="lg">LG</Badge>

// Different rounded styles
<Badge rounded="sm">Small</Badge>
<Badge rounded="md">Medium</Badge>
<Badge rounded="lg">Large</Badge>
<Badge rounded="full">Full</Badge>
```

---

## Spinner Component

### Basic Usage

```tsx
import { Spinner } from '@Foundry/design-system';

<Spinner />
<Spinner size="sm" />
<Spinner size="lg" />
<Spinner color="primary" />
```

### Spinner Variants

```tsx
<Spinner variant="circle" />  // Default spinning circle
<Spinner variant="dots" />     // Three bouncing dots
<Spinner variant="pulse" />    // Pulsing circle
<Spinner variant="wave" />     // Wave bars
<Spinner variant="bounce" />   // Bouncing dots
```

### With Label

```tsx
<Spinner label="Loading..." />
<Spinner size="lg" label="Please wait" />
```

### Full Screen Loading

```tsx
<Spinner fullScreen label="Loading application..." />
```

### Skeleton Loading

```tsx
import { Skeleton } from '@Foundry/design-system';

// Text skeleton
<Skeleton variant="text" />
<Skeleton variant="text" lines={3} />

// Circular skeleton (avatar)
<Skeleton variant="circular" width={40} height={40} />

// Rectangular skeleton (image/card)
<Skeleton variant="rectangular" height={200} />

// Rounded skeleton
<Skeleton variant="rounded" height={100} />

// Custom dimensions
<Skeleton width="80%" height={20} />

// Animation types
<Skeleton animation="pulse" />  // Default
<Skeleton animation="wave" />   // Shimmer effect
<Skeleton animation="none" />   // No animation
```

---

## Theme Configuration

### Using the Theme

```tsx
import { themeManager, useTheme } from '@Foundry/design-system';

// In a component
function ThemeToggle() {
  const { mode, setTheme, toggleTheme, colors } = useTheme();

  return (
    <div>
      <Button onClick={toggleTheme}>
        Current theme: {mode}
      </Button>

      <Button onClick={() => setTheme('light')}>Light</Button>
      <Button onClick={() => setTheme('dark')}>Dark</Button>
      <Button onClick={() => setTheme('system')}>System</Button>
    </div>
  );
}

// Direct access
themeManager.setTheme('dark');
themeManager.toggleTheme();
const currentTheme = themeManager.getTheme();
```

### Custom Theme Colors

```tsx
// Override theme colors
const customColors = {
  ...lightTheme,
  primary: '#your-color',
  primaryHover: '#your-hover-color',
};

// Apply custom colors
Object.entries(customColors).forEach(([key, value]) => {
  document.documentElement.style.setProperty(`--color-${key}`, value);
});
```

---

## Animation Usage

### Using Animation Presets

```tsx
import { animations } from '@Foundry/design-system';
import { motion } from 'framer-motion';

// Fade animation
<motion.div {...animations.fade}>
  Content
</motion.div>

// Slide animations
<motion.div variants={animations.slideUp}>
  Slides up
</motion.div>

// Scale animation
<motion.div variants={animations.scaleUp}>
  Scales up
</motion.div>

// Stagger children
<motion.div variants={animations.staggerContainer}>
  {items.map(item => (
    <motion.div key={item.id} variants={animations.staggerItem}>
      {item.content}
    </motion.div>
  ))}
</motion.div>
```

### Hover Animations

```tsx
<motion.button {...animations.hoverScale}>
  Hover to scale
</motion.button>

<motion.div {...animations.hoverLift}>
  Hover to lift
</motion.div>

<motion.div {...animations.hoverGlow}>
  Hover for glow
</motion.div>
```

### Custom Animations

```tsx
import { duration, easing, createSpring } from '@Foundry/design-system';

// Use animation utilities
const customAnimation = {
  initial: { opacity: 0, y: 20 },
  animate: {
    opacity: 1,
    y: 0,
    transition: {
      duration: duration.normal,
      ease: easing.easeOut,
    },
  },
};

// Custom spring
const customSpring = createSpring(120, 15);
```

---

## Best Practices

### Performance

1. **Lazy Load Components**: Import components only when needed
2. **Memoize Callbacks**: Use useCallback for event handlers
3. **Virtualize Long Lists**: Use virtualization for tables with many rows
4. **Optimize Animations**: Disable animations on low-end devices

### Accessibility

1. **Always Include Labels**: Every input should have a label
2. **Keyboard Navigation**: Ensure all interactions work with keyboard
3. **Focus Management**: Manage focus for modals and dynamic content
4. **ARIA Labels**: Add aria-labels for icon-only buttons

### Styling

1. **Use Design Tokens**: Use predefined colors, spacing, and typography
2. **Responsive Design**: Test components on all screen sizes
3. **Dark Mode**: Always test in both light and dark modes
4. **Consistent Spacing**: Use the spacing scale for margins and padding

### Testing

1. **Unit Tests**: Test component logic and rendering
2. **Integration Tests**: Test component interactions
3. **Accessibility Tests**: Use jest-axe for automated a11y testing
4. **Visual Tests**: Use Storybook for visual regression testing

---

## Troubleshooting

### Common Issues

**Issue**: Styles not applying
```tsx
// Solution: Import Tailwind CSS
import 'tailwindcss/tailwind.css';
// Or include in your CSS
@import 'tailwindcss/base';
@import 'tailwindcss/components';
@import 'tailwindcss/utilities';
```

**Issue**: Dark mode not working
```tsx
// Solution: Add dark class to html element
document.documentElement.classList.add('dark');
// Or use theme manager
themeManager.setTheme('dark');
```

**Issue**: Animations not working
```tsx
// Solution: Install and import framer-motion
npm install framer-motion
import { motion } from 'framer-motion';
```

**Issue**: TypeScript errors
```tsx
// Solution: Install type definitions
npm install --save-dev @types/react @types/react-dom
```

---

*For more examples and live demos, check out the Storybook documentation.*