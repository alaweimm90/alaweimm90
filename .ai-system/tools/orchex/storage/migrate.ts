/**
 * ORCHEX Storage Migration Utility
 * Migrate data between storage backends
 */

import { JsonStorageBackend } from './json-backend.js';
import { SqliteStorageBackend } from './sqlite-backend.js';
import { COLLECTIONS, CollectionName } from './types.js';

// ============================================================================
// Migration Options
// ============================================================================

export interface MigrationOptions {
  source: 'json' | 'sqlite';
  target: 'json' | 'sqlite';
  sourcePath?: string;
  targetPath?: string;
  collections?: CollectionName[];
  dryRun?: boolean;
  verbose?: boolean;
}

export interface MigrationResult {
  success: boolean;
  migratedCollections: string[];
  totalRecords: number;
  errors: string[];
  duration: number;
}

// ============================================================================
// Migration Implementation
// ============================================================================

/**
 * Migrate data between storage backends
 */
export async function migrateStorage(options: MigrationOptions): Promise<MigrationResult> {
  const startTime = Date.now();
  const result: MigrationResult = {
    success: false,
    migratedCollections: [],
    totalRecords: 0,
    errors: [],
    duration: 0,
  };

  const log = options.verbose ? console.log : (): void => {};

  try {
    // Create source backend
    log(`Creating source backend: ${options.source}`);
    const source =
      options.source === 'json'
        ? new JsonStorageBackend(options.sourcePath || '.ORCHEX/data')
        : new SqliteStorageBackend(options.sourcePath || '.ORCHEX/ORCHEX.db');

    // Create target backend
    log(`Creating target backend: ${options.target}`);
    const target =
      options.target === 'json'
        ? new JsonStorageBackend(options.targetPath || '.ORCHEX/data-migrated')
        : new SqliteStorageBackend(options.targetPath || '.ORCHEX/ORCHEX-migrated.db');

    // Initialize both backends
    log('Initializing backends...');
    await source.initialize();
    await target.initialize();

    // Determine which collections to migrate
    const collectionsToMigrate = options.collections || Object.values(COLLECTIONS);

    // Migrate each collection
    for (const collection of collectionsToMigrate) {
      log(`\nMigrating collection: ${collection}`);

      try {
        // Get all records from source
        const records = await source.getAll<Record<string, unknown>>(collection);
        log(`  Found ${records.total} records`);

        if (records.total === 0) {
          log(`  Skipping empty collection`);
          continue;
        }

        if (options.dryRun) {
          log(`  [DRY RUN] Would migrate ${records.total} records`);
          result.totalRecords += records.total;
          result.migratedCollections.push(collection);
          continue;
        }

        // Migrate records to target
        let migratedCount = 0;
        for (const record of records.items) {
          // Extract key from record (assuming 'id' field exists)
          const key = (record as { id?: string }).id || `record-${migratedCount}`;
          await target.set(collection, key, record);
          migratedCount++;
        }

        log(`  Migrated ${migratedCount} records`);
        result.totalRecords += migratedCount;
        result.migratedCollections.push(collection);
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown error';
        result.errors.push(`Failed to migrate ${collection}: ${message}`);
        log(`  ERROR: ${message}`);
      }
    }

    // Close backends
    log('\nClosing backends...');
    await source.close();
    await target.close();

    result.success = result.errors.length === 0;
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    result.errors.push(`Migration failed: ${message}`);
  }

  result.duration = Date.now() - startTime;

  return result;
}

// ============================================================================
// CLI Interface
// ============================================================================

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    console.log(`
ORCHEX Storage Migration Utility

Usage:
  npx tsx tools/ORCHEX/storage/migrate.ts <source> <target> [options]

Arguments:
  source    Source backend: json or sqlite
  target    Target backend: json or sqlite

Options:
  --source-path <path>   Path for source backend
  --target-path <path>   Path for target backend
  --collection <name>    Migrate specific collection (can be repeated)
  --dry-run              Show what would be migrated without making changes
  --verbose              Show detailed progress
  --help, -h             Show this help message

Examples:
  # Migrate from JSON to SQLite
  npx tsx tools/ORCHEX/storage/migrate.ts json sqlite --verbose

  # Migrate specific collection
  npx tsx tools/ORCHEX/storage/migrate.ts json sqlite --collection agents --verbose

  # Dry run to see what would be migrated
  npx tsx tools/ORCHEX/storage/migrate.ts json sqlite --dry-run --verbose
`);
    process.exit(0);
  }

  // Parse arguments
  const source = args[0] as 'json' | 'sqlite';
  const target = args[1] as 'json' | 'sqlite';

  if (!['json', 'sqlite'].includes(source) || !['json', 'sqlite'].includes(target)) {
    console.error('Error: source and target must be "json" or "sqlite"');
    process.exit(1);
  }

  const options: MigrationOptions = {
    source,
    target,
    dryRun: args.includes('--dry-run'),
    verbose: args.includes('--verbose'),
  };

  // Parse path options
  const sourcePathIdx = args.indexOf('--source-path');
  if (sourcePathIdx !== -1 && args[sourcePathIdx + 1]) {
    options.sourcePath = args[sourcePathIdx + 1];
  }

  const targetPathIdx = args.indexOf('--target-path');
  if (targetPathIdx !== -1 && args[targetPathIdx + 1]) {
    options.targetPath = args[targetPathIdx + 1];
  }

  // Parse collection options
  const collections: CollectionName[] = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--collection' && args[i + 1]) {
      collections.push(args[i + 1] as CollectionName);
    }
  }
  if (collections.length > 0) {
    options.collections = collections;
  }

  // Run migration
  console.log(`\nOrchex Storage Migration`);
  console.log(`========================`);
  console.log(`Source: ${source}${options.sourcePath ? ` (${options.sourcePath})` : ''}`);
  console.log(`Target: ${target}${options.targetPath ? ` (${options.targetPath})` : ''}`);
  if (options.dryRun) {
    console.log(`Mode: DRY RUN (no changes will be made)`);
  }
  console.log('');

  const result = await migrateStorage(options);

  // Print results
  console.log(`\nMigration ${result.success ? 'COMPLETED' : 'FAILED'}`);
  console.log(`Duration: ${result.duration}ms`);
  console.log(`Collections migrated: ${result.migratedCollections.length}`);
  console.log(`Total records: ${result.totalRecords}`);

  if (result.errors.length > 0) {
    console.log(`\nErrors:`);
    for (const error of result.errors) {
      console.log(`  - ${error}`);
    }
  }

  process.exit(result.success ? 0 : 1);
}

// Run CLI if executed directly
const isMainModule = process.argv[1]?.includes('migrate');
if (isMainModule) {
  main().catch(console.error);
}
