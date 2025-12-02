import * as fs from 'fs';
import * as path from 'path';

interface FunctionStat {
  file: string;
  name: string;
  lines: number;
}

interface Report {
  timestamp: string;
  python: {
    files: number;
    longFunctions: FunctionStat[];
  };
  typescript: {
    files: number;
    longFunctions: FunctionStat[];
  };
  duplicates: {
    functionsByName: Record<string, string[]>;
  };
}

function listFilesRecursive(dir: string, ext: string): string[] {
  const res: string[] = [];
  if (!fs.existsSync(dir)) return res;
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) res.push(...listFilesRecursive(full, ext));
    else if (e.isFile() && full.endsWith(ext)) res.push(full);
  }
  return res;
}

function scanTsFunctions(file: string): FunctionStat[] {
  const text = fs.readFileSync(file, 'utf8');
  const lines = text.split(/\r?\n/);
  const stats: FunctionStat[] = [];
  const stack: Array<{ name: string; start: number; brace: number }> = [];
  lines.forEach((line, i) => {
    const m =
      line.match(/function\s+(\w+)\s*\(/) || line.match(/(const|export\s+const)\s+(\w+)\s*=\s*\(/);
    if (m) stack.push({ name: m[1] || m[2], start: i + 1, brace: 0 });
    const opens = (line.match(/\{/g) || []).length;
    const closes = (line.match(/\}/g) || []).length;
    if (stack.length) stack[stack.length - 1].brace += opens - closes;
    while (stack.length && stack[stack.length - 1].brace <= 0) {
      const s = stack.pop()!;
      const fnLines = i + 1 - s.start;
      stats.push({ file, name: s.name, lines: fnLines });
    }
  });
  return stats;
}

function scanPyFunctions(file: string): FunctionStat[] {
  const text = fs.readFileSync(file, 'utf8');
  const lines = text.split(/\r?\n/);
  const stats: FunctionStat[] = [];
  let current: { name: string; start: number; indent: number } | null = null;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const fm = line.match(/^def\s+(\w+)\s*\(/);
    if (fm) {
      if (current) {
        stats.push({ file, name: current.name, lines: i - current.start });
      }
      const indent = (line.match(/^\s*/) || [''])[0].length;
      current = { name: fm[1], start: i + 1, indent };
      continue;
    }
    if (current) {
      const indent = (line.match(/^\s*/) || [''])[0].length;
      if (line.trim() && indent <= current.indent && !line.trim().startsWith('#')) {
        stats.push({ file, name: current.name, lines: i - current.start });
        current = null;
      }
    }
  }
  if (current) stats.push({ file, name: current.name, lines: lines.length - current.start + 1 });
  return stats;
}

function main() {
  const root = process.cwd();
  const pyFiles = listFilesRecursive(path.join(root, 'automation'), '.py');
  const tsFiles = listFilesRecursive(path.join(root, 'automation-ts', 'src'), '.ts');
  const pyStats = pyFiles.flatMap(scanPyFunctions).filter((s) => s.lines > 20);
  const tsStats = tsFiles.flatMap(scanTsFunctions).filter((s) => s.lines > 20);
  const names: Record<string, Set<string>> = {};
  for (const s of [...pyStats, ...tsStats]) {
    if (!names[s.name]) names[s.name] = new Set();
    names[s.name].add(s.file);
  }
  const duplicates: Record<string, string[]> = {};
  for (const [k, v] of Object.entries(names)) {
    if (v.size > 1) duplicates[k] = Array.from(v);
  }
  const report: Report = {
    timestamp: new Date().toISOString(),
    python: { files: pyFiles.length, longFunctions: pyStats },
    typescript: { files: tsFiles.length, longFunctions: tsStats },
    duplicates: { functionsByName: duplicates },
  };
  const outDir = path.join(root, '.atlas', 'reports');
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, 'refactor-audit.json'), JSON.stringify(report, null, 2));

  console.log(`Audit written to ${path.join(outDir, 'refactor-audit.json')}`);
}

main();
