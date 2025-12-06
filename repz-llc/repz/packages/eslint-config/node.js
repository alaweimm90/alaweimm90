/**
 * REPZ Shared ESLint Configuration - Node.js
 * Enterprise-grade Node.js and server-side standards
 * Extends base config with Node.js-specific rules
 */

module.exports = {
  extends: [
    './index.js'
  ],
  env: {
    node: true,
    es2022: true
  },
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module'
  },
  rules: {
    // Node.js Core Rules
    'callback-return': ['error', ['callback', 'cb', 'next', 'done']],
    'global-require': 'error',
    'handle-callback-err': ['error', '^(err|error)$'],
    'no-buffer-constructor': 'error',
    'no-mixed-requires': ['error', {
      grouping: true,
      allowCall: false
    }],
    'no-new-require': 'error',
    'no-path-concat': 'error',
    'no-process-env': 'warn',
    'no-process-exit': 'error',
    'no-restricted-modules': 'off', // Project-specific
    'no-sync': ['error', {
      allowAtRootLevel: true
    }],

    // Node.js Best Practices
    'prefer-promise-reject-errors': 'error',
    'no-return-await': 'error',
    'require-atomic-updates': 'error',

    // Security Rules
    'no-eval': 'error',
    'no-implied-eval': 'error',
    'no-new-func': 'error',
    'no-script-url': 'error',

    // Performance Rules
    'no-loop-func': 'error',
    'no-extend-native': 'error',
    'no-proto': 'error',

    // Error Handling
    'no-throw-literal': 'error',
    'prefer-promise-reject-errors': 'error',

    // Import/Export Rules for Node.js
    'no-duplicate-imports': 'error',
    'sort-imports': ['error', {
      ignoreCase: true,
      ignoreDeclarationSort: true,
      ignoreMemberSort: false,
      memberSyntaxSortOrder: ['none', 'all', 'multiple', 'single'],
      allowSeparatedGroups: true
    }],

    // Console and Debugging
    'no-console': ['warn', {
      allow: ['warn', 'error', 'info']
    }],
    'no-debugger': 'error',
    'no-alert': 'off', // Not applicable in Node.js

    // Variable Declarations
    'no-undef-init': 'error',
    'no-undefined': 'error',
    'no-use-before-define': ['error', {
      functions: false,
      classes: true,
      variables: true
    }],

    // Function Rules
    'func-names': ['error', 'as-needed'],
    'func-style': ['error', 'declaration', {
      allowArrowFunctions: true
    }],
    'max-nested-callbacks': ['error', 4],
    'no-empty-function': ['error', {
      allow: ['constructors']
    }],

    // Object and Array Rules
    'object-shorthand': ['error', 'always'],
    'prefer-destructuring': ['error', {
      array: false,
      object: true
    }, {
      enforceForRenamedProperties: false
    }],

    // String Rules
    'prefer-template': 'error',
    'template-curly-spacing': ['error', 'never'],

    // Arrow Function Rules
    'arrow-body-style': ['error', 'as-needed'],
    'arrow-parens': ['error', 'as-needed'],
    'arrow-spacing': 'error',
    'prefer-arrow-callback': ['error', {
      allowNamedFunctions: false,
      allowUnboundThis: true
    }],

    // Class Rules
    'class-methods-use-this': ['error', {
      exceptMethods: []
    }],
    'no-useless-constructor': 'error',

    // Regex Rules
    'no-invalid-regexp': 'error',
    'no-regex-spaces': 'error',
    'prefer-regex-literals': 'error',

    // Strict Mode
    'strict': ['error', 'global'],

    // Whitespace and Formatting (Prettier handles most of this)
    'eol-last': ['error', 'always'],
    'no-multiple-empty-lines': ['error', {
      max: 2,
      maxEOF: 1,
      maxBOF: 0
    }],
    'no-trailing-spaces': 'error'
  },

  // Override for specific Node.js file patterns
  overrides: [
    {
      // Express.js and server files
      files: [
        '**/server.{js,ts}',
        '**/app.{js,ts}',
        '**/routes/**/*.{js,ts}',
        '**/middleware/**/*.{js,ts}',
        '**/controllers/**/*.{js,ts}'
      ],
      rules: {
        'no-console': 'off', // Allow console in server files
        'callback-return': 'off' // Express middleware patterns
      }
    },
    {
      // Configuration files
      files: [
        '**/*.config.{js,ts}',
        '**/config/**/*.{js,ts}',
        '**/webpack.*.js',
        '**/rollup.*.js',
        '**/vite.*.js'
      ],
      rules: {
        'no-console': 'off',
        'global-require': 'off',
        'no-process-env': 'off'
      }
    },
    {
      // Build and deployment scripts
      files: [
        '**/scripts/**/*.{js,ts}',
        '**/tools/**/*.{js,ts}',
        '**/build/**/*.{js,ts}'
      ],
      rules: {
        'no-console': 'off',
        'no-process-exit': 'off',
        'no-process-env': 'off'
      }
    },
    {
      // Test files
      files: [
        '**/*.test.{js,ts}',
        '**/*.spec.{js,ts}',
        '**/test/**/*.{js,ts}',
        '**/__tests__/**/*.{js,ts}'
      ],
      env: {
        jest: true,
        mocha: true,
        'vitest-globals/env': true
      },
      rules: {
        'no-console': 'off',
        'max-nested-callbacks': 'off',
        'no-magic-numbers': 'off',
        'no-unused-expressions': 'off'
      }
    },
    {
      // Database and migration files
      files: [
        '**/migrations/**/*.{js,ts}',
        '**/seeds/**/*.{js,ts}',
        '**/models/**/*.{js,ts}',
        '**/database/**/*.{js,ts}'
      ],
      rules: {
        'no-console': 'off',
        'callback-return': 'off'
      }
    },
    {
      // CLI and command-line tools
      files: [
        '**/bin/**/*.{js,ts}',
        '**/cli/**/*.{js,ts}',
        '**/*.cli.{js,ts}'
      ],
      rules: {
        'no-console': 'off',
        'no-process-exit': 'off',
        'no-process-env': 'off'
      }
    },
    {
      // Serverless functions (Supabase Edge Functions, Vercel, etc.)
      files: [
        '**/functions/**/*.{js,ts}',
        '**/edge-functions/**/*.{js,ts}',
        '**/api/**/*.{js,ts}'
      ],
      env: {
        'shared-node-browser': true
      },
      rules: {
        'no-console': ['warn', {
          allow: ['warn', 'error', 'info', 'log']
        }],
        'no-process-env': 'off'
      }
    }
  ]
};