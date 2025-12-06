#!/usr/bin/env tsx

/**
 * REPZ API Schema Builder
 * Builds OpenAPI specifications and generates TypeScript types
 */

import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';
import * as yaml from 'yaml';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT_DIR = join(__dirname, '..');

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m'
};

function log(message: string, color: keyof typeof colors = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function ensureDir(dir: string) {
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
}

/**
 * Convert OpenAPI YAML to JSON
 */
function convertYamlToJson() {
  log('📄 Converting OpenAPI YAML to JSON...', 'blue');
  
  const yamlPath = join(ROOT_DIR, 'openapi.yaml');
  const jsonPath = join(ROOT_DIR, 'openapi.json');
  
  if (!existsSync(yamlPath)) {
    throw new Error('openapi.yaml not found');
  }
  
  const yamlContent = readFileSync(yamlPath, 'utf8');
  const jsonContent = yaml.parse(yamlContent);
  
  writeFileSync(jsonPath, JSON.stringify(jsonContent, null, 2));
  log('✅ OpenAPI JSON generated', 'green');
}

/**
 * Generate TypeScript types from OpenAPI spec
 */
function generateTypes() {
  log('🔧 Generating TypeScript types...', 'blue');
  
  const typesDir = join(ROOT_DIR, 'src', 'types');
  ensureDir(typesDir);
  
  try {
    // Generate types using openapi-typescript
    execSync(
      `npx openapi-typescript ${join(ROOT_DIR, 'openapi.yaml')} -o ${join(typesDir, 'generated.ts')}`,
      { cwd: ROOT_DIR, stdio: 'inherit' }
    );
    
    log('✅ TypeScript types generated', 'green');
  } catch (error) {
    log('❌ Failed to generate TypeScript types', 'red');
    throw error;
  }
}

/**
 * Generate Zod schemas from OpenAPI spec
 */
function generateZodSchemas() {
  log('🛡️ Generating Zod validation schemas...', 'blue');
  
  const schemasDir = join(ROOT_DIR, 'src', 'schemas');
  ensureDir(schemasDir);
  
  // Read OpenAPI spec
  const openApiSpec = JSON.parse(readFileSync(join(ROOT_DIR, 'openapi.json'), 'utf8'));
  
  // Generate Zod schemas for each component schema
  const zodSchemas: string[] = [
    "import { z } from 'zod';",
    "",
    "// Generated Zod schemas from OpenAPI specification",
    ""
  ];
  
  if (openApiSpec.components?.schemas) {
    Object.entries(openApiSpec.components.schemas).forEach(([name, schema]: [string, any]) => {
      const zodSchema = convertOpenApiSchemaToZod(name, schema);
      zodSchemas.push(zodSchema);
      zodSchemas.push("");
    });
  }
  
  // Add exports
  zodSchemas.push("// Export all schemas");
  if (openApiSpec.components?.schemas) {
    const exports = Object.keys(openApiSpec.components.schemas)
      .map(name => `  ${name}Schema`)
      .join(',\n');
    zodSchemas.push(`export {\n${exports}\n};`);
  }
  
  writeFileSync(join(schemasDir, 'generated.ts'), zodSchemas.join('\n'));
  log('✅ Zod schemas generated', 'green');
}

/**
 * Convert OpenAPI schema to Zod schema
 */
function convertOpenApiSchemaToZod(name: string, schema: any): string {
  const zodSchema = convertSchemaToZod(schema, 0);
  return `export const ${name}Schema = ${zodSchema};`;
}

function convertSchemaToZod(schema: any, depth: number = 0): string {
  const indent = '  '.repeat(depth);
  
  if (schema.$ref) {
    // Handle references
    const refName = schema.$ref.split('/').pop();
    return `${refName}Schema`;
  }
  
  if (schema.allOf) {
    // Handle allOf (intersection)
    const schemas = schema.allOf.map((s: any) => convertSchemaToZod(s, depth));
    return schemas.join('.and(') + ')'.repeat(schemas.length - 1);
  }
  
  if (schema.oneOf || schema.anyOf) {
    // Handle oneOf/anyOf (union)
    const schemas = (schema.oneOf || schema.anyOf).map((s: any) => convertSchemaToZod(s, depth));
    return `z.union([${schemas.join(', ')}])`;
  }
  
  if (schema.type === 'object') {
    if (schema.additionalProperties === true) {
      return 'z.record(z.any())';
    }
    
    if (schema.properties) {
      const properties = Object.entries(schema.properties)
        .map(([key, prop]: [string, any]) => {
          const propSchema = convertSchemaToZod(prop, depth + 1);
          const isRequired = schema.required?.includes(key);
          const optionalSuffix = isRequired ? '' : '.optional()';
          return `${indent}  ${key}: ${propSchema}${optionalSuffix}`;
        })
        .join(',\n');
      
      return `z.object({\n${properties}\n${indent}})`;
    }
    
    return 'z.object({})';
  }
  
  if (schema.type === 'array') {
    const itemSchema = schema.items ? convertSchemaToZod(schema.items, depth) : 'z.any()';
    let arraySchema = `z.array(${itemSchema})`;
    
    if (schema.minItems !== undefined) {
      arraySchema += `.min(${schema.minItems})`;
    }
    if (schema.maxItems !== undefined) {
      arraySchema += `.max(${schema.maxItems})`;
    }
    
    return arraySchema;
  }
  
  if (schema.type === 'string') {
    let stringSchema = 'z.string()';
    
    if (schema.format === 'email') {
      stringSchema += '.email()';
    } else if (schema.format === 'uri' || schema.format === 'url') {
      stringSchema += '.url()';
    } else if (schema.format === 'uuid') {
      stringSchema += '.uuid()';
    } else if (schema.format === 'date') {
      stringSchema += '.date()';
    } else if (schema.format === 'date-time') {
      stringSchema += '.datetime()';
    }
    
    if (schema.minLength !== undefined) {
      stringSchema += `.min(${schema.minLength})`;
    }
    if (schema.maxLength !== undefined) {
      stringSchema += `.max(${schema.maxLength})`;
    }
    if (schema.pattern) {
      stringSchema += `.regex(/${schema.pattern}/)`;
    }
    if (schema.enum) {
      const enumValues = schema.enum.map((v: string) => `"${v}"`).join(', ');
      stringSchema = `z.enum([${enumValues}])`;
    }
    
    return stringSchema;
  }
  
  if (schema.type === 'number' || schema.type === 'integer') {
    let numberSchema = schema.type === 'integer' ? 'z.number().int()' : 'z.number()';
    
    if (schema.minimum !== undefined) {
      numberSchema += `.min(${schema.minimum})`;
    }
    if (schema.maximum !== undefined) {
      numberSchema += `.max(${schema.maximum})`;
    }
    
    return numberSchema;
  }
  
  if (schema.type === 'boolean') {
    return 'z.boolean()';
  }
  
  // Fallback
  return 'z.any()';
}

/**
 * Generate API documentation
 */
function generateDocs() {
  log('📚 Generating API documentation...', 'blue');
  
  const docsDir = join(ROOT_DIR, 'docs');
  ensureDir(docsDir);
  
  try {
    // Generate HTML documentation using redoc-cli
    execSync(
      `npx redoc-cli build ${join(ROOT_DIR, 'openapi.yaml')} --output ${join(docsDir, 'api-reference.html')}`,
      { cwd: ROOT_DIR, stdio: 'inherit' }
    );
    
    log('✅ API documentation generated', 'green');
  } catch (error) {
    log('❌ Failed to generate API documentation', 'red');
    throw error;
  }
}

/**
 * Validate OpenAPI specification
 */
function validateSpec() {
  log('🔍 Validating OpenAPI specification...', 'blue');
  
  try {
    execSync(
      `npx swagger-codegen validate -i ${join(ROOT_DIR, 'openapi.yaml')}`,
      { cwd: ROOT_DIR, stdio: 'inherit' }
    );
    
    log('✅ OpenAPI specification is valid', 'green');
  } catch (error) {
    log('❌ OpenAPI specification validation failed', 'red');
    throw error;
  }
}

/**
 * Generate client SDK (optional)
 */
function generateClientSdk() {
  log('🔨 Generating client SDK...', 'blue');
  
  const generatedDir = join(ROOT_DIR, 'generated');
  ensureDir(generatedDir);
  
  try {
    // Generate TypeScript fetch client
    execSync(
      `npx openapi-generator-cli generate -i ${join(ROOT_DIR, 'openapi.yaml')} -g typescript-fetch -o ${join(generatedDir, 'client')}`,
      { cwd: ROOT_DIR, stdio: 'inherit' }
    );
    
    log('✅ Client SDK generated', 'green');
  } catch (error) {
    log('⚠️ Failed to generate client SDK (optional)', 'yellow');
    // Don't throw - this is optional
  }
}

/**
 * Main build process
 */
async function main() {
  const args = process.argv.slice(2);
  const watch = args.includes('--watch');
  
  log('🚀 Building REPZ API Schema...', 'bright');
  
  try {
    // Core build steps
    convertYamlToJson();
    generateTypes();
    generateZodSchemas();
    validateSpec();
    generateDocs();
    
    // Optional steps
    if (!watch) {
      generateClientSdk();
    }
    
    log('✨ API Schema build completed successfully!', 'green');
    
    if (watch) {
      log('👀 Watching for changes...', 'yellow');
      
      // Simple file watcher (in production, use chokidar or similar)
      const { watch: fsWatch } = await import('fs');
      
      fsWatch(join(ROOT_DIR, 'openapi.yaml'), { persistent: true }, (eventType) => {
        if (eventType === 'change') {
          log('🔄 OpenAPI spec changed, rebuilding...', 'cyan');
          try {
            convertYamlToJson();
            generateTypes();
            generateZodSchemas();
            validateSpec();
            generateDocs();
            log('✅ Rebuild completed', 'green');
          } catch (error) {
            log('❌ Rebuild failed', 'red');
            console.error(error);
          }
        }
      });
      
      // Keep process alive
      process.stdin.resume();
    }
    
  } catch (error) {
    log('❌ Build failed', 'red');
    console.error(error);
    process.exit(1);
  }
}

// Run the build
main().catch(console.error);