import * as fs from 'fs';
import * as path from 'path';

interface OrgItem {
  name: string;
  path: string;
  type: 'org' | 'portfolio';
  requiredFiles: string[];
  missingFiles: string[];
  recommendedSubdirs: string[];
  missingSubdirs: string[];
}

interface OrgReport {
  timestamp: string;
  root: string;
  items: OrgItem[];
  summary: {
    total: number;
    compliant: number;
    nonCompliant: number;
  };
}

function listDirs(dir: string): string[] {
  try {
    return fs
      .readdirSync(dir, { withFileTypes: true })
      .filter((d) => d.isDirectory())
      .map((d) => path.join(dir, d.name));
  } catch {
    return [];
  }
}

function exists(p: string): boolean {
  try {
    return fs.existsSync(p);
  } catch {
    return false;
  }
}

function checkOrg(orgPath: string): OrgItem {
  const name = path.basename(orgPath);
  const required = ['README.md', 'LICENSE', 'SECURITY.md'];
  const missingFiles = required.filter((f) => !exists(path.join(orgPath, f)));
  const recommendedSubdirs = ['website', 'portal', 'app'];
  const missingSubdirs = recommendedSubdirs.filter((s) => !exists(path.join(orgPath, s)));
  const type: 'org' | 'portfolio' = name.toLowerCase().includes('business') ? 'portfolio' : 'org';
  return {
    name,
    path: orgPath,
    type,
    requiredFiles: required,
    missingFiles,
    recommendedSubdirs,
    missingSubdirs,
  };
}

function buildReport(root: string): OrgReport {
  const orgs = listDirs(root);
  const items = orgs.map(checkOrg);
  const compliant = items.filter((i) => i.missingFiles.length === 0).length;
  return {
    timestamp: new Date().toISOString(),
    root,
    items,
    summary: { total: items.length, compliant, nonCompliant: items.length - compliant },
  };
}

function main() {
  const root = path.join(process.cwd(), 'organizations');
  const report = buildReport(root);
  const outDir = path.join(process.cwd(), '.atlas', 'reports');
  fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(outDir, 'org-validation.json');
  fs.writeFileSync(outFile, JSON.stringify(report, null, 2));
  // eslint-disable-next-line no-console
  console.log(`Org validation written to ${outFile}`);
}

main();
