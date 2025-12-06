import type { Preview } from '@storybook/react';
import '../src/index.css';

const preview: Preview = {
  parameters: {
    actions: { argTypesRegex: '^on[A-Z].*' },
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/,
      },
    },
    backgrounds: {
      default: 'dark',
      values: [
        {
          name: 'dark',
          value: '#0B0B0C',
        },
        {
          name: 'light',
          value: '#E6E9EF',
        },
        {
          name: 'white',
          value: '#FFFFFF',
        },
      ],
    },
  },
};

export default preview;
