import * as fs from 'fs';
import * as path from 'path';

function ensureDir(p: string): void {
  if (!fs.existsSync(p)) fs.mkdirSync(p, { recursive: true });
}

function fixOrg(orgPath: string, subdirs: string[]): void {
  for (const s of subdirs) ensureDir(path.join(orgPath, s));
}

function main(): void {
  const root = path.join(process.cwd(), 'organizations');
  const targets = ['alawein-business'];
  for (const t of targets) {
    const orgPath = path.join(root, t);
    if (fs.existsSync(orgPath)) fixOrg(orgPath, ['website', 'portal', 'app']);
  }

  console.log('Org directories ensured for business portfolio');
}

main();
