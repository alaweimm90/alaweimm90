import { runStaticAnalysis } from '../src/lib/code/staticScan';
import { CodeFinding } from '../src/types';

describe('Code Security Analysis', () => {
  describe('Static analysis with built-in rules', () => {
    it('should detect subprocess shell injection (CWE-78)', async () => {
      const vulnerableCode = {
        path: 'vulnerable.py',
        content: `
import subprocess
def search(query):
    # Vulnerable to command injection
    result = subprocess.run("grep " + query + " /var/log/app.log", shell=True)
    return result
`
      };
      
      const result = await runStaticAnalysis([vulnerableCode]);
      
      expect(result.findings.length).toBeGreaterThan(0);
      
      const shellInjection = result.findings.find(f => f.cwe === 'CWE-78');
      expect(shellInjection).toBeDefined();
      expect(shellInjection?.severity).toBe('HIGH');
      expect(shellInjection?.rule).toContain('shell');
    });

    it('should detect path traversal (CWE-22)', async () => {
      const pathTraversalCode = {
        path: 'pathtraversal.py',
        content: `
def read_file(filename):
    # Vulnerable to path traversal
    with open('/uploads/' + filename, 'r') as f:
        return f.read()
`
      };
      
      const result = await runStaticAnalysis([pathTraversalCode]);
      
      const pathTraversal = result.findings.find(f => f.cwe === 'CWE-22');
      expect(pathTraversal).toBeDefined();
      expect(pathTraversal?.severity).toBe('HIGH');
    });

    it('should detect hardcoded credentials (CWE-798)', async () => {
      const hardcodedCredsCode = {
        path: 'config.py',
        content: `
# Configuration file
DATABASE_URL = "postgresql://admin:password123@localhost/db"
API_KEY = "sk-1234567890abcdef"
SECRET_TOKEN = "super_secret_token_123"
`
      };
      
      const result = await runStaticAnalysis([hardcodedCredsCode]);
      
      const hardcodedCreds = result.findings.find(f => 
        f.cwe === 'CWE-798' || f.cwe === 'CWE-259'
      );
      expect(hardcodedCreds).toBeDefined();
      expect(hardcodedCreds?.severity).toBe('HIGH');
    });

    it('should detect weak cryptography (CWE-327)', async () => {
      const weakCryptoCode = {
        path: 'crypto.py',
        content: `
import hashlib
def hash_password(password):
    # Using weak hash algorithm
    return hashlib.md5(password.encode()).hexdigest()
    
def another_weak_hash(data):
    return hashlib.sha1(data).hexdigest()
`
      };
      
      const result = await runStaticAnalysis([weakCryptoCode]);
      
      const weakCrypto = result.findings.find(f => f.cwe === 'CWE-327');
      expect(weakCrypto).toBeDefined();
      expect(weakCrypto?.severity).toBe('MEDIUM');
    });

    it('should detect SQL injection patterns (CWE-89)', async () => {
      const sqlInjectionCode = {
        path: 'database.py',
        content: `
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    return execute_query(query)
    
def search_products(name):
    sql = f"SELECT * FROM products WHERE name LIKE '%{name}%'"
    return db.execute(sql)
`
      };
      
      const result = await runStaticAnalysis([sqlInjectionCode]);
      
      const sqlInjection = result.findings.find(f => f.cwe === 'CWE-89');
      expect(sqlInjection).toBeDefined();
      expect(sqlInjection?.severity).toBe('HIGH');
    });

    it('should calculate findings per KLOC correctly', async () => {
      const code1 = {
        path: 'file1.py',
        content: Array(100).fill('print("line")').join('\n') // 100 lines
      };
      
      const code2 = {
        path: 'file2.py',
        content: `
import subprocess
subprocess.run("ls", shell=True)  # Finding
password = "hardcoded123"  # Finding
` + Array(50).fill('# comment').join('\n') // ~53 lines
      };
      
      const result = await runStaticAnalysis([code1, code2]);
      
      expect(result.metrics.filesScanned).toBe(2);
      expect(result.metrics.totalLOC).toBeGreaterThan(0);
      expect(result.metrics.findingsPerKloc).toBeGreaterThan(0);
      
      // Should have at least 2 findings from code2
      expect(result.findings.length).toBeGreaterThanOrEqual(2);
    });

    it('should handle multiple file types', async () => {
      const pythonCode = {
        path: 'app.py',
        content: 'password = "secret123"'
      };
      
      const jsCode = {
        path: 'app.js',
        content: 'eval(userInput); // dangerous'
      };
      
      const result = await runStaticAnalysis([pythonCode, jsCode]);
      
      expect(result.findings.length).toBeGreaterThan(0);
      expect(result.metrics.filesScanned).toBe(2);
    });

    it('should respect file size limits', async () => {
      // Create a file that exceeds size limit (>2MB by default)
      const largeFile = {
        path: 'large.py',
        content: 'x = 1\n'.repeat(1000000) // Very large file
      };
      
      const normalFile = {
        path: 'normal.py',
        content: 'password = "test"'
      };
      
      // Set a smaller limit for testing
      process.env.VITE_MAX_FILE_SIZE = '1000';
      
      const result = await runStaticAnalysis([largeFile, normalFile]);
      
      // Should only scan the normal file
      expect(result.metrics.filesScanned).toBe(1);
      
      // Reset env
      delete process.env.VITE_MAX_FILE_SIZE;
    });

    it('should deduplicate findings', async () => {
      const codeWithDuplicates = {
        path: 'duplicate.py',
        content: `
password = "secret1"
password = "secret2"
password = "secret3"
# All on different lines but same rule
`
      };
      
      const result = await runStaticAnalysis([codeWithDuplicates]);
      
      // Should deduplicate by path-line-rule combination
      const uniqueKeys = new Set(
        result.findings.map(f => `${f.path}-${f.line}-${f.rule}`)
      );
      
      expect(uniqueKeys.size).toBe(result.findings.length);
    });
  });
});