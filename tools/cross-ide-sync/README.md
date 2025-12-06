# Cross-IDE Sync

Automatically sync prompts from GitHub to all AI IDEs.

## Supported IDEs

- Amazon Q (`~/.aws/amazonq/prompts`)
- Claude Desktop (`~/AppData/Roaming/Claude/prompts`)
- Windsurf (`~/.windsurf/prompts`)
- Cline (`~/.cline/prompts`)
- Cursor (`~/.cursor/prompts`)

## Usage

### Sync Once
```bash
python cli.py sync
```

### Sync to Specific IDE
```bash
python cli.py sync amazonq
python cli.py sync amazonq,claude-desktop
```

### Watch Mode (Auto-sync on changes)
```bash
python cli.py watch
```

Requires: `pip install watchdog`

## How It Works

1. Source: `docs/ai-knowledge/prompts/` (GitHub repo)
2. Syncer copies all `.md` files to each IDE's prompt directory
3. Preserves folder structure
4. Watch mode detects changes and auto-syncs

## Integration

Add to startup script for automatic sync:
```bash
cd tools/cross-ide-sync && python cli.py watch &
```
