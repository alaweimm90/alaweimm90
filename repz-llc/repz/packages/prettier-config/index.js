/**
 * REPZ Shared Prettier Configuration
 * Enterprise-grade code formatting standards
 * Used across all REPZ platform packages
 */

module.exports = {
  // Core Formatting
  printWidth: 80,
  tabWidth: 2,
  useTabs: false,
  semi: true,
  singleQuote: true,
  quoteProps: 'as-needed',
  trailingComma: 'es5',
  bracketSpacing: true,
  bracketSameLine: false,
  arrowParens: 'avoid',
  endOfLine: 'lf',

  // Language-Specific Overrides
  overrides: [
    // JavaScript and TypeScript
    {
      files: ['*.js', '*.jsx', '*.ts', '*.tsx'],
      options: {
        singleQuote: true,
        semi: true,
        trailingComma: 'es5',
        printWidth: 80,
        tabWidth: 2,
        useTabs: false,
        bracketSpacing: true,
        bracketSameLine: false,
        arrowParens: 'avoid'
      }
    },

    // JSON Files
    {
      files: ['*.json', '*.jsonc'],
      options: {
        printWidth: 120,
        tabWidth: 2,
        useTabs: false,
        semi: false,
        singleQuote: false,
        trailingComma: 'none'
      }
    },

    // Package.json (special formatting)
    {
      files: ['package.json', 'package-lock.json'],
      options: {
        printWidth: 120,
        tabWidth: 2,
        useTabs: false,
        semi: false,
        singleQuote: false,
        trailingComma: 'none'
      }
    },

    // Markdown
    {
      files: ['*.md', '*.mdx'],
      options: {
        printWidth: 100,
        tabWidth: 2,
        useTabs: false,
        semi: false,
        singleQuote: false,
        trailingComma: 'none',
        proseWrap: 'always',
        embeddedLanguageFormatting: 'auto'
      }
    },

    // CSS, SCSS, Less
    {
      files: ['*.css', '*.scss', '*.less'],
      options: {
        printWidth: 100,
        tabWidth: 2,
        useTabs: false,
        semi: true,
        singleQuote: true,
        trailingComma: 'none'
      }
    },

    // HTML and Templates
    {
      files: ['*.html', '*.htm', '*.vue', '*.svelte'],
      options: {
        printWidth: 120,
        tabWidth: 2,
        useTabs: false,
        semi: false,
        singleQuote: true,
        bracketSameLine: false,
        htmlWhitespaceSensitivity: 'css'
      }
    },

    // YAML Files
    {
      files: ['*.yml', '*.yaml'],
      options: {
        printWidth: 120,
        tabWidth: 2,
        useTabs: false,
        semi: false,
        singleQuote: true,
        trailingComma: 'none',
        bracketSpacing: true
      }
    },

    // XML Files
    {
      files: ['*.xml', '*.svg'],
      options: {
        printWidth: 120,
        tabWidth: 2,
        useTabs: false,
        xmlWhitespaceSensitivity: 'ignore',
        xmlSelfClosingSpace: true
      }
    },

    // GraphQL
    {
      files: ['*.graphql', '*.gql'],
      options: {
        printWidth: 100,
        tabWidth: 2,
        useTabs: false,
        semi: false,
        singleQuote: true,
        trailingComma: 'none'
      }
    },

    // Configuration Files
    {
      files: [
        '.eslintrc.js',
        '.eslintrc.cjs',
        'prettier.config.js',
        'tailwind.config.js',
        'vite.config.js',
        'vitest.config.js',
        'jest.config.js',
        'webpack.config.js',
        'rollup.config.js',
        'next.config.js',
        'nuxt.config.js'
      ],
      options: {
        printWidth: 100,
        tabWidth: 2,
        useTabs: false,
        semi: true,
        singleQuote: true,
        trailingComma: 'es5',
        bracketSpacing: true,
        arrowParens: 'avoid'
      }
    },

    // Test Files
    {
      files: [
        '*.test.js',
        '*.test.ts',
        '*.test.jsx',
        '*.test.tsx',
        '*.spec.js',
        '*.spec.ts',
        '*.spec.jsx',
        '*.spec.tsx'
      ],
      options: {
        printWidth: 100,
        tabWidth: 2,
        useTabs: false,
        semi: true,
        singleQuote: true,
        trailingComma: 'es5',
        bracketSpacing: true,
        arrowParens: 'avoid'
      }
    },

    // Storybook Files
    {
      files: ['*.stories.js', '*.stories.ts', '*.stories.jsx', '*.stories.tsx'],
      options: {
        printWidth: 100,
        tabWidth: 2,
        useTabs: false,
        semi: true,
        singleQuote: true,
        trailingComma: 'es5',
        bracketSpacing: true,
        arrowParens: 'avoid'
      }
    },

    // Shell Scripts
    {
      files: ['*.sh', '*.bash', '*.zsh'],
      options: {
        printWidth: 120,
        tabWidth: 2,
        useTabs: false,
        keepLf: true
      }
    },

    // Docker Files
    {
      files: ['Dockerfile*', '*.dockerfile'],
      options: {
        printWidth: 120,
        tabWidth: 2,
        useTabs: false
      }
    }
  ],

  // Plugin Configuration
  plugins: [
    // Add plugins as needed for specific file types
    // Examples:
    // '@prettier/plugin-xml',
    // 'prettier-plugin-tailwindcss',
    // 'prettier-plugin-organize-imports'
  ],

  // Additional Options for Specific Use Cases
  // These can be uncommented and configured as needed

  // Import Sorting (requires prettier-plugin-organize-imports)
  // organizeImportsSkipDestructiveCodeActions: true,

  // Tailwind CSS Class Sorting (requires prettier-plugin-tailwindcss)
  // tailwindConfig: './tailwind.config.js',
  // tailwindFunctions: ['clsx', 'cn', 'cva'],

  // PHP (requires @prettier/plugin-php)
  // phpVersion: '8.1',

  // Java (requires prettier-plugin-java)
  // tabWidth: 4, // Override for Java files

  // SQL (requires prettier-plugin-sql)
  // language: 'postgresql',
  // keywordCase: 'upper',

  // Performance and Debugging
  requirePragma: false,
  insertPragma: false,
  rangeStart: 0,
  rangeEnd: Infinity
};