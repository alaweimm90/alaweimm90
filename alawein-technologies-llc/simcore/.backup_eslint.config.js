import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist", "public/**"] },
  // Baseline rules for all files (no type info required)
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ["**/*.{ts,tsx,js,jsx}"],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      // Keep Fast Refresh rule as warn
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],

      // TypeScript rule adjustments
      // Allow escape hatches but surface issues in CI and during dev
      "@typescript-eslint/no-unused-vars": [
        "warn",
        { "argsIgnorePattern": "^_", "varsIgnorePattern": "^_" }
      ],
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/no-require-imports": "off",
      "@typescript-eslint/no-empty-object-type": "warn",
      "@typescript-eslint/consistent-type-imports": [
        "warn",
        { "fixStyle": "inline-type-imports" }
      ],
      "@typescript-eslint/no-floating-promises": "off",
      "@typescript-eslint/no-unused-expressions": "warn",
      "@typescript-eslint/ban-ts-comment": "warn",
      "no-empty": "warn",

      // React Hooks rules should be enforced strictly
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",

      // General JS/TS rules - reduce noise to warnings
      "no-case-declarations": "warn",
      "prefer-const": "warn",
      "no-control-regex": "warn",
      "no-useless-escape": "warn",
      "no-implied-eval": "off",
      "@typescript-eslint/no-implied-eval": "off",
      "no-var": "warn",
    },
  },
  // Type-aware rules only for app source
  {
    files: ["src/**/*.{ts,tsx}"],
    extends: [...tseslint.configs.recommendedTypeChecked],
    languageOptions: {
      parserOptions: {
        project: ["./tsconfig.json", "./tsconfig.app.json"],
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      "@typescript-eslint/no-floating-promises": "warn",
      "@typescript-eslint/no-unused-vars": ["warn", { "argsIgnorePattern": "^_", "varsIgnorePattern": "^_" }],
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/no-unnecessary-type-assertion": "warn",
      "@typescript-eslint/restrict-template-expressions": "warn",
      "@typescript-eslint/no-redundant-type-constituents": "warn",
      "@typescript-eslint/unbound-method": "warn",
      "@typescript-eslint/no-unsafe-assignment": "off",
      "@typescript-eslint/no-unsafe-call": "off",
      "@typescript-eslint/no-unsafe-member-access": "off",
      "@typescript-eslint/no-unsafe-return": "off",
      "@typescript-eslint/no-unsafe-argument": "off",
      "@typescript-eslint/require-await": "off",
      "@typescript-eslint/ban-ts-comment": "warn",
      "@typescript-eslint/no-misused-promises": ["warn", { checksVoidReturn: false }],
      "@typescript-eslint/no-empty-object-type": "warn",
    },
  },
  // Relax rules in test files to reduce noise
  {
    files: [
      "src/**/*.test.{ts,tsx}",
      "src/**/__tests__/**/*.{ts,tsx}",
    ],
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/ban-ts-comment": "off",
      "@typescript-eslint/no-unused-vars": ["warn", { "argsIgnorePattern": "^_" }],
      "@typescript-eslint/unbound-method": "off",
      "no-empty": "off",
      "no-implied-eval": "off",
      "@typescript-eslint/no-implied-eval": "off",
      "no-var": "off",
    }
  }
);
