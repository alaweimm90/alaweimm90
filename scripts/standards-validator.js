#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const { toTitleCase, checkMarkdownTitle, checkYaml } = require('./standards-lib');

function main() {
  const args = process.argv.slice(2);
  const mode = args.includes('--docs') ? 'docs' : args.includes('--yaml') ? 'yaml' : null;
  if (!mode) process.exit(0);
  const files = args.filter(a => !a.startsWith('--'));
  let ok = true;
  for (const f of files) {
    if (mode === 'docs' && path.extname(f).toLowerCase() === '.md') {
      if (!checkMarkdownTitle(f)) {
        ok = false;
        console.error(`Doc title mismatch: ${f}`);
      }
    }
    if (mode === 'yaml' && ['.yml', '.yaml'].includes(path.extname(f).toLowerCase())) {
      if (!checkYaml(f)) {
        ok = false;
        console.error(`YAML contains tabs: ${f}`);
      }
    }
  }
  process.exit(ok ? 0 : 1);
}

if (require.main === module) main();
