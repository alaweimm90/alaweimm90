/**
 * REPZ Shared ESLint Configuration - TypeScript
 * Enterprise-grade TypeScript standards
 * Extends base config with TypeScript-specific rules
 */

module.exports = {
  extends: [
    './index.js',
    '@typescript-eslint/recommended',
    '@typescript-eslint/recommended-requiring-type-checking'
  ],
  parser: '@typescript-eslint/parser',
  plugins: [
    '@typescript-eslint',
    'import'
  ],
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    project: ['./tsconfig.json', './tsconfig.*.json']
  },
  settings: {
    'import/resolver': {
      typescript: {
        alwaysTryTypes: true,
        project: ['./tsconfig.json', './tsconfig.*.json']
      }
    }
  },
  rules: {
    // TypeScript Core Rules
    '@typescript-eslint/adjacent-overload-signatures': 'error',
    '@typescript-eslint/array-type': ['error', {
      default: 'array-simple',
      readonly: 'array-simple'
    }],
    '@typescript-eslint/await-thenable': 'error',
    '@typescript-eslint/ban-ts-comment': ['error', {
      'ts-expect-error': 'allow-with-description',
      'ts-ignore': false,
      'ts-nocheck': false,
      'ts-check': false
    }],
    '@typescript-eslint/ban-tslint-comment': 'error',
    '@typescript-eslint/ban-types': 'error',
    '@typescript-eslint/class-literal-property-style': ['error', 'fields'],
    '@typescript-eslint/consistent-indexed-access': 'error',
    '@typescript-eslint/consistent-type-assertions': ['error', {
      assertionStyle: 'as',
      objectLiteralTypeAssertions: 'never'
    }],
    '@typescript-eslint/consistent-type-definitions': ['error', 'interface'],
    '@typescript-eslint/consistent-type-exports': ['error', {
      fixMixedExportsWithInlineTypeSpecifier: true
    }],
    '@typescript-eslint/consistent-type-imports': ['error', {
      prefer: 'type-imports',
      disallowTypeAnnotations: false,
      fixStyle: 'inline-type-imports'
    }],
    '@typescript-eslint/explicit-function-return-type': ['error', {
      allowExpressions: true,
      allowTypedFunctionExpressions: true,
      allowHigherOrderFunctions: true,
      allowDirectConstAssertionInArrowFunctions: true
    }],
    '@typescript-eslint/explicit-member-accessibility': ['error', {
      accessibility: 'explicit',
      overrides: {
        accessors: 'explicit',
        constructors: 'no-public',
        methods: 'explicit',
        properties: 'off',
        parameterProperties: 'explicit'
      }
    }],
    '@typescript-eslint/explicit-module-boundary-types': ['error', {
      allowArgumentsExplicitlyTypedAsAny: false,
      allowDirectConstAssertionInArrowFunctions: true,
      allowHigherOrderFunctions: true,
      allowTypedFunctionExpressions: true
    }],
    '@typescript-eslint/member-delimiter-style': ['error', {
      multiline: {
        delimiter: 'semi',
        requireLast: true
      },
      singleline: {
        delimiter: 'semi',
        requireLast: false
      }
    }],
    '@typescript-eslint/member-ordering': ['error', {
      default: [
        // Index signature
        'signature',
        
        // Fields
        'public-static-field',
        'protected-static-field',
        'private-static-field',
        'public-decorated-field',
        'protected-decorated-field',
        'private-decorated-field',
        'public-instance-field',
        'protected-instance-field',
        'private-instance-field',
        'public-abstract-field',
        'protected-abstract-field',
        'private-abstract-field',
        
        // Constructors
        'public-constructor',
        'protected-constructor',
        'private-constructor',
        
        // Methods
        'public-static-method',
        'protected-static-method',
        'private-static-method',
        'public-decorated-method',
        'protected-decorated-method',
        'private-decorated-method',
        'public-instance-method',
        'protected-instance-method',
        'private-instance-method',
        'public-abstract-method',
        'protected-abstract-method',
        'private-abstract-method'
      ]
    }],
    '@typescript-eslint/method-signature-style': ['error', 'property'],
    '@typescript-eslint/naming-convention': [
      'error',
      {
        selector: 'default',
        format: ['camelCase']
      },
      {
        selector: 'variable',
        format: ['camelCase', 'UPPER_CASE', 'PascalCase']
      },
      {
        selector: 'parameter',
        format: ['camelCase'],
        leadingUnderscore: 'allow'
      },
      {
        selector: 'memberLike',
        modifiers: ['private'],
        format: ['camelCase'],
        leadingUnderscore: 'require'
      },
      {
        selector: 'typeLike',
        format: ['PascalCase']
      },
      {
        selector: 'interface',
        format: ['PascalCase'],
        custom: {
          regex: '^I[A-Z]',
          match: false
        }
      },
      {
        selector: 'enumMember',
        format: ['UPPER_CASE']
      }
    ],
    '@typescript-eslint/no-base-to-string': 'error',
    '@typescript-eslint/no-confusing-non-null-assertion': 'error',
    '@typescript-eslint/no-confusing-void-expression': ['error', {
      ignoreArrowShorthand: true
    }],
    '@typescript-eslint/no-duplicate-enum-values': 'error',
    '@typescript-eslint/no-dynamic-delete': 'error',
    '@typescript-eslint/no-empty-interface': ['error', {
      allowSingleExtends: true
    }],
    '@typescript-eslint/no-explicit-any': ['error', {
      fixToUnknown: true,
      ignoreRestArgs: false
    }],
    '@typescript-eslint/no-extra-non-null-assertion': 'error',
    '@typescript-eslint/no-extraneous-class': ['error', {
      allowConstructorOnly: false,
      allowEmpty: false,
      allowStaticOnly: false,
      allowWithDecorator: true
    }],
    '@typescript-eslint/no-floating-promises': 'error',
    '@typescript-eslint/no-for-in-array': 'error',
    '@typescript-eslint/no-implicit-any-catch': ['error', {
      allowExplicitAny: false
    }],
    '@typescript-eslint/no-inferrable-types': ['error', {
      ignoreParameters: true,
      ignoreProperties: true
    }],
    '@typescript-eslint/no-invalid-void-type': 'error',
    '@typescript-eslint/no-meaningless-void-operator': 'error',
    '@typescript-eslint/no-misused-new': 'error',
    '@typescript-eslint/no-misused-promises': ['error', {
      checksConditionals: true,
      checksVoidReturn: true
    }],
    '@typescript-eslint/no-namespace': ['error', {
      allowDeclarations: false,
      allowDefinitionFiles: true
    }],
    '@typescript-eslint/no-non-null-asserted-nullish-coalescing': 'error',
    '@typescript-eslint/no-non-null-asserted-optional-chain': 'error',
    '@typescript-eslint/no-non-null-assertion': 'error',
    '@typescript-eslint/no-redundant-type-constituents': 'error',
    '@typescript-eslint/no-require-imports': 'error',
    '@typescript-eslint/no-this-alias': 'error',
    '@typescript-eslint/no-type-alias': ['error', {
      allowAliases: 'in-unions-and-intersections',
      allowCallbacks: 'always',
      allowConditionalTypes: 'always',
      allowConstructors: 'never',
      allowLiterals: 'in-unions-and-intersections',
      allowMappedTypes: 'always',
      allowTupleTypes: 'always'
    }],
    '@typescript-eslint/no-unnecessary-boolean-literal-compare': 'error',
    '@typescript-eslint/no-unnecessary-condition': 'error',
    '@typescript-eslint/no-unnecessary-qualifier': 'error',
    '@typescript-eslint/no-unnecessary-type-arguments': 'error',
    '@typescript-eslint/no-unnecessary-type-assertion': 'error',
    '@typescript-eslint/no-unnecessary-type-constraint': 'error',
    '@typescript-eslint/no-unsafe-argument': 'error',
    '@typescript-eslint/no-unsafe-assignment': 'error',
    '@typescript-eslint/no-unsafe-call': 'error',
    '@typescript-eslint/no-unsafe-member-access': 'error',
    '@typescript-eslint/no-unsafe-return': 'error',
    '@typescript-eslint/no-useless-empty-export': 'error',
    '@typescript-eslint/no-var-requires': 'error',
    '@typescript-eslint/non-nullable-type-assertion-style': 'error',
    '@typescript-eslint/prefer-as-const': 'error',
    '@typescript-eslint/prefer-enum-initializers': 'error',
    '@typescript-eslint/prefer-for-of': 'error',
    '@typescript-eslint/prefer-function-type': 'error',
    '@typescript-eslint/prefer-includes': 'error',
    '@typescript-eslint/prefer-literal-enum-member': 'error',
    '@typescript-eslint/prefer-namespace-keyword': 'error',
    '@typescript-eslint/prefer-nullish-coalescing': 'error',
    '@typescript-eslint/prefer-optional-chain': 'error',
    '@typescript-eslint/prefer-readonly': 'error',
    '@typescript-eslint/prefer-readonly-parameter-types': 'off', // Too strict for most projects
    '@typescript-eslint/prefer-reduce-type-parameter': 'error',
    '@typescript-eslint/prefer-regexp-exec': 'error',
    '@typescript-eslint/prefer-return-this-type': 'error',
    '@typescript-eslint/prefer-string-starts-ends-with': 'error',
    '@typescript-eslint/prefer-ts-expect-error': 'error',
    '@typescript-eslint/promise-function-async': 'error',
    '@typescript-eslint/require-array-sort-compare': 'error',
    '@typescript-eslint/restrict-plus-operands': 'error',
    '@typescript-eslint/restrict-template-expressions': ['error', {
      allowNumber: true,
      allowBoolean: true,
      allowAny: false,
      allowNullish: false
    }],
    '@typescript-eslint/sort-type-union-intersection-members': 'error',
    '@typescript-eslint/strict-boolean-expressions': ['error', {
      allowString: false,
      allowNumber: false,
      allowNullableObject: false,
      allowNullableBoolean: false,
      allowNullableString: false,
      allowNullableNumber: false,
      allowAny: false
    }],
    '@typescript-eslint/switch-exhaustiveness-check': 'error',
    '@typescript-eslint/triple-slash-reference': 'error',
    '@typescript-eslint/type-annotation-spacing': 'error',
    '@typescript-eslint/typedef': ['error', {
      arrayDestructuring: false,
      arrowParameter: false,
      memberVariableDeclaration: false,
      objectDestructuring: false,
      parameter: false,
      propertyDeclaration: true,
      variableDeclaration: false,
      variableDeclarationIgnoreFunction: false
    }],
    '@typescript-eslint/unbound-method': ['error', {
      ignoreStatic: true
    }],
    '@typescript-eslint/unified-signatures': 'error',

    // Extension Rules (replace base ESLint rules)
    'brace-style': 'off',
    '@typescript-eslint/brace-style': ['error', '1tbs'],
    'comma-dangle': 'off',
    '@typescript-eslint/comma-dangle': ['error', 'always-multiline'],
    'comma-spacing': 'off',
    '@typescript-eslint/comma-spacing': ['error', {
      before: false,
      after: true
    }],
    'default-param-last': 'off',
    '@typescript-eslint/default-param-last': 'error',
    'dot-notation': 'off',
    '@typescript-eslint/dot-notation': 'error',
    'func-call-spacing': 'off',
    '@typescript-eslint/func-call-spacing': ['error', 'never'],
    'indent': 'off',
    '@typescript-eslint/indent': ['error', 2],
    'init-declarations': 'off',
    '@typescript-eslint/init-declarations': 'off',
    'keyword-spacing': 'off',
    '@typescript-eslint/keyword-spacing': ['error'],
    'lines-between-class-members': 'off',
    '@typescript-eslint/lines-between-class-members': ['error', 'always'],
    'no-array-constructor': 'off',
    '@typescript-eslint/no-array-constructor': 'error',
    'no-dupe-class-members': 'off',
    '@typescript-eslint/no-dupe-class-members': 'error',
    'no-duplicate-imports': 'off',
    '@typescript-eslint/no-duplicate-imports': 'error',
    'no-empty-function': 'off',
    '@typescript-eslint/no-empty-function': 'error',
    'no-extra-parens': 'off',
    '@typescript-eslint/no-extra-parens': ['error', 'all', {
      ignoreJSX: 'all'
    }],
    'no-extra-semi': 'off',
    '@typescript-eslint/no-extra-semi': 'error',
    'no-implied-eval': 'off',
    '@typescript-eslint/no-implied-eval': 'error',
    'no-invalid-this': 'off',
    '@typescript-eslint/no-invalid-this': 'error',
    'no-loop-func': 'off',
    '@typescript-eslint/no-loop-func': 'error',
    'no-loss-of-precision': 'off',
    '@typescript-eslint/no-loss-of-precision': 'error',
    'no-magic-numbers': 'off',
    '@typescript-eslint/no-magic-numbers': ['error', {
      ignore: [-1, 0, 1, 2],
      ignoreArrayIndexes: true,
      ignoreDefaultValues: true,
      ignoreReadonlyClassProperties: true
    }],
    'no-redeclare': 'off',
    '@typescript-eslint/no-redeclare': 'error',
    'no-restricted-imports': 'off',
    '@typescript-eslint/no-restricted-imports': 'off',
    'no-shadow': 'off',
    '@typescript-eslint/no-shadow': 'error',
    'no-throw-literal': 'off',
    '@typescript-eslint/no-throw-literal': 'error',
    'no-unused-expressions': 'off',
    '@typescript-eslint/no-unused-expressions': 'error',
    'no-unused-vars': 'off',
    '@typescript-eslint/no-unused-vars': ['error', {
      argsIgnorePattern: '^_',
      varsIgnorePattern: '^_',
      ignoreRestSiblings: true
    }],
    'no-use-before-define': 'off',
    '@typescript-eslint/no-use-before-define': 'error',
    'no-useless-constructor': 'off',
    '@typescript-eslint/no-useless-constructor': 'error',
    'object-curly-spacing': 'off',
    '@typescript-eslint/object-curly-spacing': ['error', 'always'],
    'padding-line-between-statements': 'off',
    '@typescript-eslint/padding-line-between-statements': [
      'error',
      { blankLine: 'always', prev: '*', next: 'return' },
      { blankLine: 'always', prev: ['const', 'let', 'var'], next: '*' },
      { blankLine: 'any', prev: ['const', 'let', 'var'], next: ['const', 'let', 'var'] },
      { blankLine: 'always', prev: 'directive', next: '*' },
      { blankLine: 'any', prev: 'directive', next: 'directive' }
    ],
    'quotes': 'off',
    '@typescript-eslint/quotes': ['error', 'single', {
      avoidEscape: true,
      allowTemplateLiterals: true
    }],
    'require-await': 'off',
    '@typescript-eslint/require-await': 'error',
    'return-await': 'off',
    '@typescript-eslint/return-await': ['error', 'in-try-catch'],
    'semi': 'off',
    '@typescript-eslint/semi': ['error', 'always'],
    'space-before-blocks': 'off',
    '@typescript-eslint/space-before-blocks': 'error',
    'space-before-function-paren': 'off',
    '@typescript-eslint/space-before-function-paren': ['error', {
      anonymous: 'always',
      named: 'never',
      asyncArrow: 'always'
    }],
    'space-infix-ops': 'off',
    '@typescript-eslint/space-infix-ops': 'error',

    // Import Rules
    'import/order': ['error', {
      groups: [
        'builtin',
        'external',
        'internal',
        'parent',
        'sibling',
        'index',
        'object',
        'type'
      ],
      'newlines-between': 'always',
      alphabetize: {
        order: 'asc',
        caseInsensitive: true
      }
    }],
    'import/no-duplicates': 'error',
    'import/no-unresolved': 'error',
    'import/no-cycle': 'error',
    'import/no-self-import': 'error'
  },

  // Override for specific file patterns
  overrides: [
    {
      files: ['**/*.js', '**/*.jsx'],
      rules: {
        '@typescript-eslint/explicit-function-return-type': 'off',
        '@typescript-eslint/explicit-module-boundary-types': 'off',
        '@typescript-eslint/no-var-requires': 'off'
      }
    },
    {
      files: ['**/*.d.ts'],
      rules: {
        '@typescript-eslint/no-unused-vars': 'off',
        '@typescript-eslint/triple-slash-reference': 'off'
      }
    },
    {
      files: ['**/*.test.{ts,tsx}', '**/*.spec.{ts,tsx}'],
      rules: {
        '@typescript-eslint/no-magic-numbers': 'off',
        '@typescript-eslint/no-non-null-assertion': 'off',
        '@typescript-eslint/no-explicit-any': 'off'
      }
    }
  ]
};