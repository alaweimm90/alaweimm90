const fs = require('fs');
const path = require('path');

function toTitleCase(s) {
  return s
    .replace(/[-_]+/g, ' ')
    .trim()
    .split(' ')
    .filter(Boolean)
    .map((w) => w[0].toUpperCase() + w.slice(1).toLowerCase())
    .join(' ');
}

function checkMarkdownTitle(file) {
  const name = path.basename(file, path.extname(file));
  const expected = `# ${toTitleCase(name)}`;
  const text = fs.readFileSync(file, 'utf8');
  const firstLine = (text.split(/\r?\n/, 1)[0] || '').trim();
  return firstLine === expected;
}

function checkYaml(file) {
  const text = fs.readFileSync(file, 'utf8');
  const hasTabs = /\t/.test(text);
  return !hasTabs;
}

module.exports = { toTitleCase, checkMarkdownTitle, checkYaml };
