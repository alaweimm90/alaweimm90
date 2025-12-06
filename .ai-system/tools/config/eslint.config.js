export default [
  { ignores: ['dist', 'node_modules', '.git'] },
  { files: ['**/*.{js,ts,tsx}'], rules: { 'no-console': 'warn' } },
];
