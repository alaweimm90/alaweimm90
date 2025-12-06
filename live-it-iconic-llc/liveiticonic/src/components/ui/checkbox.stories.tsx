import type { Meta, StoryObj } from '@storybook/react';
import { Checkbox } from './checkbox';
import { Label } from './label';
import { useState } from 'react';

const meta: Meta<typeof Checkbox> = {
  title: 'UI/Checkbox',
  component: Checkbox,
  tags: ['autodocs'],
  parameters: {
    backgrounds: {
      default: 'dark',
    },
  },
};

export default meta;
type Story = StoryObj<typeof Checkbox>;

export const Default: Story = {
  args: {
    id: 'checkbox-default',
  },
};

export const Checked: Story = {
  args: {
    id: 'checkbox-checked',
    defaultChecked: true,
  },
};

export const Disabled: Story = {
  args: {
    id: 'checkbox-disabled',
    disabled: true,
  },
};

export const WithLabel: Story = {
  render: () => (
    <div className="flex items-center space-x-2">
      <Checkbox id="agree" />
      <Label htmlFor="agree" className="font-normal">
        I agree to the terms and conditions
      </Label>
    </div>
  ),
};

export const CheckboxGroup: Story = {
  render: () => {
    const [checked, setChecked] = useState([false, false, false]);
    
    return (
      <div className="space-y-3">
        <div className="flex items-center space-x-2">
          <Checkbox
            id="option1"
            checked={checked[0]}
            onCheckedChange={(value) => {
              const newChecked = [...checked];
              newChecked[0] = value;
              setChecked(newChecked);
            }}
          />
          <Label htmlFor="option1" className="font-normal">Performance Collection</Label>
        </div>
        <div className="flex items-center space-x-2">
          <Checkbox
            id="option2"
            checked={checked[1]}
            onCheckedChange={(value) => {
              const newChecked = [...checked];
              newChecked[1] = value;
              setChecked(newChecked);
            }}
          />
          <Label htmlFor="option2" className="font-normal">Heritage Collection</Label>
        </div>
        <div className="flex items-center space-x-2">
          <Checkbox
            id="option3"
            checked={checked[2]}
            onCheckedChange={(value) => {
              const newChecked = [...checked];
              newChecked[2] = value;
              setChecked(newChecked);
            }}
          />
          <Label htmlFor="option3" className="font-normal">Urban Collection</Label>
        </div>
      </div>
    );
  },
};
