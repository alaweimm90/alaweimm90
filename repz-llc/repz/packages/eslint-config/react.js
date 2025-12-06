/**
 * REPZ Shared ESLint Configuration - React
 * Enterprise-grade React and JSX standards
 * Extends base config with React-specific rules
 */

module.exports = {
  extends: [
    './index.js',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
    'plugin:jsx-a11y/recommended'
  ],
  plugins: [
    'react',
    'react-hooks',
    'jsx-a11y'
  ],
  settings: {
    react: {
      version: 'detect'
    }
  },
  parserOptions: {
    ecmaFeatures: {
      jsx: true
    }
  },
  rules: {
    // React Core Rules
    'react/react-in-jsx-scope': 'off', // Not needed with React 17+
    'react/prop-types': 'error',
    'react/display-name': 'error',
    'react/no-children-prop': 'error',
    'react/no-danger-with-children': 'error',
    'react/no-deprecated': 'error',
    'react/no-direct-mutation-state': 'error',
    'react/no-find-dom-node': 'error',
    'react/no-is-mounted': 'error',
    'react/no-render-return-value': 'error',
    'react/no-string-refs': 'error',
    'react/no-unescaped-entities': 'error',
    'react/no-unknown-property': 'error',
    'react/require-render-return': 'error',
    'react/self-closing-comp': 'error',
    'react/void-dom-elements-no-children': 'error',

    // React Best Practices
    'react/boolean-prop-naming': ['error', {
      rule: '^(is|has|should|can|did|will)[A-Z]([A-Za-z0-9]?)+',
      message: 'Boolean props should start with is, has, should, can, did, or will'
    }],
    'react/button-has-type': 'error',
    'react/default-props-match-prop-types': 'error',
    'react/destructuring-assignment': ['error', 'always'],
    'react/forbid-prop-types': ['error', {
      forbid: ['any', 'array', 'object'],
      checkContextTypes: true,
      checkChildContextTypes: true
    }],
    'react/function-component-definition': ['error', {
      namedComponents: 'arrow-function',
      unnamedComponents: 'arrow-function'
    }],
    'react/jsx-boolean-value': ['error', 'never'],
    'react/jsx-child-element-spacing': 'error',
    'react/jsx-closing-bracket-location': ['error', 'line-aligned'],
    'react/jsx-closing-tag-location': 'error',
    'react/jsx-curly-brace-presence': ['error', {
      props: 'never',
      children: 'never'
    }],
    'react/jsx-curly-spacing': ['error', {
      when: 'never',
      children: true
    }],
    'react/jsx-equals-spacing': ['error', 'never'],
    'react/jsx-filename-extension': ['error', {
      extensions: ['.jsx', '.tsx']
    }],
    'react/jsx-first-prop-new-line': ['error', 'multiline-multiprop'],
    'react/jsx-fragments': ['error', 'syntax'],
    'react/jsx-indent': ['error', 2],
    'react/jsx-indent-props': ['error', 2],
    'react/jsx-key': ['error', {
      checkFragmentShorthand: true,
      checkKeyMustBeforeSpread: true
    }],
    'react/jsx-max-props-per-line': ['error', {
      maximum: 1,
      when: 'multiline'
    }],
    'react/jsx-newline': ['error', {
      prevent: true
    }],
    'react/jsx-no-bind': ['error', {
      ignoreRefs: true,
      allowArrowFunctions: true,
      allowFunctions: false,
      allowBind: false
    }],
    'react/jsx-no-constructed-context-values': 'error',
    'react/jsx-no-duplicate-props': 'error',
    'react/jsx-no-script-url': 'error',
    'react/jsx-no-target-blank': ['error', {
      enforceDynamicLinks: 'always'
    }],
    'react/jsx-no-useless-fragment': 'error',
    'react/jsx-one-expression-per-line': ['error', {
      allow: 'single-child'
    }],
    'react/jsx-pascal-case': 'error',
    'react/jsx-props-no-multi-spaces': 'error',
    'react/jsx-sort-props': ['error', {
      callbacksLast: true,
      shorthandFirst: true,
      multiline: 'last',
      ignoreCase: true,
      reservedFirst: true
    }],
    'react/jsx-tag-spacing': ['error', {
      closingSlash: 'never',
      beforeSelfClosing: 'always',
      afterOpening: 'never',
      beforeClosing: 'never'
    }],
    'react/jsx-uses-react': 'error',
    'react/jsx-uses-vars': 'error',
    'react/jsx-wrap-multilines': ['error', {
      declaration: 'parens-new-line',
      assignment: 'parens-new-line',
      return: 'parens-new-line',
      arrow: 'parens-new-line',
      condition: 'parens-new-line',
      logical: 'parens-new-line',
      prop: 'parens-new-line'
    }],
    'react/no-access-state-in-setstate': 'error',
    'react/no-array-index-key': 'warn',
    'react/no-did-mount-set-state': 'error',
    'react/no-did-update-set-state': 'error',
    'react/no-multi-comp': ['error', {
      ignoreStateless: true
    }],
    'react/no-redundant-should-component-update': 'error',
    'react/no-this-in-sfc': 'error',
    'react/no-typos': 'error',
    'react/no-unsafe': 'error',
    'react/no-unused-prop-types': 'error',
    'react/no-unused-state': 'error',
    'react/no-will-update-set-state': 'error',
    'react/prefer-es6-class': ['error', 'always'],
    'react/prefer-stateless-function': ['error', {
      ignorePureComponents: true
    }],
    'react/require-default-props': 'error',
    'react/sort-comp': 'error',
    'react/sort-prop-types': 'error',
    'react/state-in-constructor': ['error', 'always'],
    'react/static-property-placement': ['error', 'property assignment'],
    'react/style-prop-object': 'error',

    // React Hooks Rules
    'react-hooks/rules-of-hooks': 'error',
    'react-hooks/exhaustive-deps': 'warn',

    // JSX Accessibility Rules
    'jsx-a11y/alt-text': 'error',
    'jsx-a11y/anchor-has-content': 'error',
    'jsx-a11y/anchor-is-valid': 'error',
    'jsx-a11y/aria-activedescendant-has-tabindex': 'error',
    'jsx-a11y/aria-props': 'error',
    'jsx-a11y/aria-proptypes': 'error',
    'jsx-a11y/aria-role': 'error',
    'jsx-a11y/aria-unsupported-elements': 'error',
    'jsx-a11y/click-events-have-key-events': 'error',
    'jsx-a11y/heading-has-content': 'error',
    'jsx-a11y/html-has-lang': 'error',
    'jsx-a11y/iframe-has-title': 'error',
    'jsx-a11y/img-redundant-alt': 'error',
    'jsx-a11y/interactive-supports-focus': 'error',
    'jsx-a11y/label-has-associated-control': 'error',
    'jsx-a11y/media-has-caption': 'error',
    'jsx-a11y/mouse-events-have-key-events': 'error',
    'jsx-a11y/no-access-key': 'error',
    'jsx-a11y/no-autofocus': 'error',
    'jsx-a11y/no-distracting-elements': 'error',
    'jsx-a11y/no-interactive-element-to-noninteractive-role': 'error',
    'jsx-a11y/no-noninteractive-element-interactions': 'error',
    'jsx-a11y/no-noninteractive-element-to-interactive-role': 'error',
    'jsx-a11y/no-noninteractive-tabindex': 'error',
    'jsx-a11y/no-redundant-roles': 'error',
    'jsx-a11y/no-static-element-interactions': 'error',
    'jsx-a11y/role-has-required-aria-props': 'error',
    'jsx-a11y/role-supports-aria-props': 'error',
    'jsx-a11y/scope': 'error',
    'jsx-a11y/tabindex-no-positive': 'error'
  },

  // Override for component files
  overrides: [
    {
      files: ['**/*.stories.{js,ts,jsx,tsx}'],
      rules: {
        'react/prop-types': 'off',
        'react/require-default-props': 'off'
      }
    },
    {
      files: ['**/*.test.{js,ts,jsx,tsx}', '**/*.spec.{js,ts,jsx,tsx}'],
      rules: {
        'react/prop-types': 'off',
        'react/require-default-props': 'off',
        'jsx-a11y/no-autofocus': 'off'
      }
    }
  ]
};