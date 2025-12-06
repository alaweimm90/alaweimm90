/**
 * Style Dictionary Configuration
 * Generates design tokens in multiple formats for cross-platform use
 *
 * This configuration exports Live It Iconic design tokens to:
 * - CSS (CSS custom properties)
 * - JavaScript (ES6 modules)
 * - JSON (for design tools)
 * - TypeScript (with type definitions)
 *
 * Run: npx style-dictionary build
 */

module.exports = {
  source: ['src/design-tokens/**/*.ts'],

  platforms: {
    // =====================================================================
    // CSS VARIABLES - For use in stylesheets
    // =====================================================================
    css: {
      transformGroup: 'css',
      buildPath: 'dist/tokens/',
      transforms: [
        'attribute/cti',
        'name/cti/kebab',
        'time/seconds',
        'color/css',
      ],
      files: [
        {
          destination: 'variables.css',
          format: 'css/variables',
          options: {
            outputReferences: true,
          },
        },
        {
          destination: 'variables-scoped.css',
          format: 'css/variables',
          selector: '.lii-tokens',
        },
      ],
    },

    // =====================================================================
    // JAVASCRIPT/TYPESCRIPT - For React, Vue, etc.
    // =====================================================================
    js: {
      transformGroup: 'js',
      buildPath: 'dist/tokens/',
      transforms: [
        'attribute/cti',
        'name/cti/camel',
        'time/seconds',
        'color/hex',
        'size/px',
      ],
      files: [
        {
          destination: 'tokens.js',
          format: 'javascript/es6',
        },
        {
          destination: 'tokens.mjs',
          format: 'javascript/es6-commonjs',
        },
      ],
    },

    // =====================================================================
    // JSON - For design tools and configuration
    // =====================================================================
    json: {
      transformGroup: 'js',
      buildPath: 'dist/tokens/',
      files: [
        {
          destination: 'tokens.json',
          format: 'json',
        },
        {
          destination: 'tokens-flat.json',
          format: 'json/flat',
        },
        {
          destination: 'tokens-nested.json',
          format: 'json/nested',
        },
      ],
    },

    // =====================================================================
    // ANDROID - For Android development
    // =====================================================================
    android: {
      transformGroup: 'android',
      buildPath: 'dist/tokens/',
      transforms: [
        'attribute/cti',
        'name/cti/snake',
        'color/hex8android',
        'size/remToSp',
        'size/remToDp',
      ],
      files: [
        {
          destination: 'colors.xml',
          format: 'android/colors',
        },
        {
          destination: 'dimens.xml',
          format: 'android/dimens',
        },
      ],
    },

    // =====================================================================
    // iOS - For Swift development
    // =====================================================================
    ios: {
      transformGroup: 'ios-swift',
      buildPath: 'dist/tokens/ios/',
      transforms: [
        'attribute/cti',
        'name/cti/camel',
        'color/UIColor',
        'size/remToPt',
        'fontSizes/remToNs',
        'fontFamily/objC',
        'fontWeights/nsFontWeight',
      ],
      files: [
        {
          destination: 'LIITokens.swift',
          format: 'ios-swift/class.swift',
          options: {
            className: 'LIITokens',
          },
        },
      ],
    },

    // =====================================================================
    // FIGMA - For Figma plugin integration
    // =====================================================================
    figma: {
      transformGroup: 'js',
      buildPath: 'dist/tokens/',
      files: [
        {
          destination: 'tokens-figma.json',
          format: 'json',
          options: {
            fileHeader: [
              'Live It Iconic Design Tokens',
              'Generated from style-dictionary',
              'For use with Figma plugins',
            ],
          },
        },
      ],
    },

    // =====================================================================
    // STORYBOOK - For component documentation
    // =====================================================================
    storybook: {
      transformGroup: 'js',
      buildPath: 'dist/tokens/',
      transforms: [
        'attribute/cti',
        'name/cti/camel',
        'color/hex',
      ],
      files: [
        {
          destination: 'tokens-storybook.js',
          format: 'javascript/es6',
          options: {
            outputReferences: true,
          },
        },
      ],
    },

    // =====================================================================
    // WEB COMPONENTS - For custom elements
    // =====================================================================
    webComponents: {
      transformGroup: 'css',
      buildPath: 'dist/tokens/',
      files: [
        {
          destination: 'tokens-web-components.css',
          format: 'css/variables',
          selector: ':root',
          options: {
            outputReferences: true,
          },
        },
      ],
    },

    // =====================================================================
    // DOCUMENTATION - For design system documentation
    // =====================================================================
    docs: {
      transformGroup: 'js',
      buildPath: 'docs/',
      files: [
        {
          destination: 'tokens-reference.json',
          format: 'json/nested',
          options: {
            showFileHeader: true,
          },
        },
      ],
    },
  },

  // =========================================================================
  // CUSTOM TRANSFORMS
  // =========================================================================
  customTransforms: [
    {
      name: 'color/hex',
      type: 'value',
      transformer: (token) => {
        if (token.type === 'color' && token.value.startsWith('#')) {
          return token.value;
        }
        return token.value;
      },
    },
    {
      name: 'size/px',
      type: 'value',
      transformer: (token) => {
        if (token.type === 'sizing' && typeof token.value === 'number') {
          return `${token.value}px`;
        }
        return token.value;
      },
    },
  ],

  // =========================================================================
  // CUSTOM FORMATS
  // =========================================================================
  customFormats: {
    'json/flat': function (dictionary) {
      const flatTokens = {};
      const flatten = (obj, prefix = '') => {
        Object.keys(obj).forEach((key) => {
          const value = obj[key];
          const newKey = prefix ? `${prefix}.${key}` : key;

          if (value && typeof value === 'object' && value.value === undefined) {
            flatten(value, newKey);
          } else if (value && typeof value === 'object' && value.value) {
            flatTokens[newKey] = value.value;
          } else {
            flatTokens[newKey] = value;
          }
        });
      };

      flatten(dictionary.tokens);
      return JSON.stringify(flatTokens, null, 2);
    },

    'json/nested': function (dictionary) {
      return JSON.stringify(dictionary.tokens, null, 2);
    },

    'css/variables': function (dictionary, config) {
      const { selector = ':root', outputReferences = false } = config.options || {};
      let output = `${selector} {\n`;

      const tokens = dictionary.allTokens;

      tokens.forEach((token) => {
        if (token.type === 'color') {
          output += `  --lii-color-${token.path.join('-')}: ${token.value};\n`;
        } else if (token.type === 'spacing') {
          output += `  --lii-spacing-${token.path.join('-')}: ${token.value};\n`;
        } else if (token.type === 'sizing') {
          output += `  --lii-size-${token.path.join('-')}: ${token.value};\n`;
        } else {
          output += `  --lii-${token.path.join('-')}: ${token.value};\n`;
        }
      });

      output += '}\n';
      return output;
    },
  },

  // =========================================================================
  // FILTERS
  // =========================================================================
  filterStates: ['active', 'hover', 'disabled', 'dark'],

  // =========================================================================
  // RESOLVE FORMAT
  // =========================================================================
  resolveFormats: (formats) => {
    // Only include specified formats
    return formats;
  },

  // =========================================================================
  // HOOKS
  // =========================================================================
  hooks: {
    formats: {
      'before': function () {
        console.log('Starting token generation...');
      },
      'after': function () {
        console.log('Token generation complete!');
      },
    },
  },
};

/**
 * USAGE:
 *
 * 1. Install style-dictionary:
 *    npm install -g style-dictionary
 *
 * 2. Generate all formats:
 *    style-dictionary build
 *
 * 3. Generate specific platform:
 *    style-dictionary build --platform css
 *    style-dictionary build --platform js
 *    style-dictionary build --platform ios
 *
 * 4. Watch for changes:
 *    style-dictionary build --watch
 *
 * OUTPUT STRUCTURE:
 *
 * dist/tokens/
 * ├── variables.css                 # CSS custom properties
 * ├── variables-scoped.css          # Scoped CSS variables
 * ├── tokens.js                     # ES6 export
 * ├── tokens.mjs                    # CommonJS export
 * ├── tokens.json                   # Nested JSON
 * ├── tokens-flat.json              # Flat JSON
 * ├── tokens-nested.json            # Nested JSON
 * ├── tokens-figma.json             # Figma format
 * ├── tokens-web-components.css     # Web components CSS
 * ├── colors.xml                    # Android colors
 * ├── dimens.xml                    # Android dimensions
 * └── ios/
 *     └── LIITokens.swift           # Swift tokens
 */
