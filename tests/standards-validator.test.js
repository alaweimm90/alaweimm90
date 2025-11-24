const { toTitleCase, checkMarkdownTitle, checkYaml } = require('../scripts/standards-lib');
const fs = require('fs');
const path = require('path');

test('toTitleCase converts filename parts', () => {
  expect(toTitleCase('quick_start-guide')).toBe('Quick Start Guide');
});

test('checkMarkdownTitle passes when first line matches', () => {
  const tmp = path.join(__dirname, 'TMP_TEST.md');
  fs.writeFileSync(tmp, '# Tmp Test\nBody');
  expect(checkMarkdownTitle(tmp)).toBe(true);
  fs.unlinkSync(tmp);
});

test('checkMarkdownTitle fails when first line mismatches', () => {
  const tmp = path.join(__dirname, 'TMP_TEST.md');
  fs.writeFileSync(tmp, 'Wrong Title\nBody');
  expect(checkMarkdownTitle(tmp)).toBe(false);
  fs.unlinkSync(tmp);
});

test('checkYaml detects tabs', () => {
  const tmp = path.join(__dirname, 'tmp.yaml');
  fs.writeFileSync(tmp, 'root:\n\tchild: value');
  expect(checkYaml(tmp)).toBe(false);
  fs.unlinkSync(tmp);
});

test('checkYaml passes without tabs', () => {
  const tmp = path.join(__dirname, 'tmp.yaml');
  fs.writeFileSync(tmp, 'root:\n  child: value');
  expect(checkYaml(tmp)).toBe(true);
  fs.unlinkSync(tmp);
});
