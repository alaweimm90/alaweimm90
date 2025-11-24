#!/usr/bin/env node

const { checkMarkdownTitle, checkYaml } = require('./standards-lib');

function main() {
  const args = process.argv.slice(2);
  const mode = args.includes('--docs') ? 'docs' : args.includes('--yaml') ? 'yaml' : null;
  if (!mode) process.exit(0);
  const files = args.filter(a => !a.startsWith('--'));
  let ok = true;
  for (const f of files) {
    const dot = f.lastIndexOf('.');
    const ext = dot >= 0 ? f.slice(dot).toLowerCase() : '';
    if (mode === 'docs' && ext === '.md') {
      if (!checkMarkdownTitle(f)) {
        ok = false;
        console.error(`Doc title mismatch: ${f}`);
      }
    }
    if (mode === 'yaml' && ['.yml', '.yaml'].includes(ext)) {
      if (!checkYaml(f)) {
        ok = false;
        console.error(`YAML contains tabs: ${f}`);
      }
    }
  }
  process.exit(ok ? 0 : 1);
}

if (require.main === module) main();
