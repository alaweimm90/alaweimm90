import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: [
    "dist",
    "_graveyard/**",
    "packages/**",
    ".storybook/**",
    "supabase/functions/**",
    "scripts/**",
    "src/__tests__/**",
    "src/utils/audit-cli.js",
    "src/utils/enforcement-tools.ts",
    "**/*.test.ts",
    "**/*.test.tsx",
    "**/*.spec.ts",  
    "**/*.spec.tsx",
    "**/*.stories.ts",
    "**/*.stories.tsx",
    "*.config.js",
    "*.config.ts",
    "tailwind.config.ts"
  ] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ["**/*.{ts,tsx}"],
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
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],
      "@typescript-eslint/no-unused-expressions": "off",
      "@typescript-eslint/no-unused-vars": "off",
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/no-require-imports": "warn",
      "@typescript-eslint/ban-ts-comment": "warn",
      "@typescript-eslint/no-empty-object-type": "warn",
      "@typescript-eslint/no-unsafe-function-type": "warn",
      "no-case-declarations": "warn",
      "no-misleading-character-class": "warn",
      "no-constant-binary-expression": "warn",
      "no-useless-escape": "warn",
      "prefer-rest-params": "warn",
      "react-hooks/exhaustive-deps": "warn",
      "react-hooks/rules-of-hooks": "warn",
    },
  }
);
