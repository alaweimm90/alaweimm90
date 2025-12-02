# Orchestrator

> Task management and context injection for AI-assisted development

**Category:** core

## Commands

### `npm run ai:start`

Start tracking a new task

**Arguments:**

| Name        | Type   | Required | Description                                               | Default |
| ----------- | ------ | -------- | --------------------------------------------------------- | ------- |
| type        | string | Yes      | Task type (feature, bugfix, refactor, docs, test, devops) | -       |
| scope       | string | No       | Comma-separated scope areas                               | -       |
| description | string | Yes      | Task description                                          | -       |

### `npm run ai:complete`

Mark current task as complete

**Arguments:**

| Name         | Type    | Required | Description                           | Default |
| ------------ | ------- | -------- | ------------------------------------- | ------- |
| success      | boolean | Yes      | Whether task was successful           | -       |
| filesChanged | string  | No       | Comma-separated list of changed files | -       |
| linesAdded   | number  | No       | Lines of code added                   | 0       |
| linesRemoved | number  | No       | Lines of code removed                 | 0       |
| testsAdded   | number  | No       | Number of tests added                 | 0       |
| notes        | string  | No       | Completion notes                      | -       |

### `npm run ai:context`

Get AI context for a task type

### `npm run ai:metrics`

View or update AI effectiveness metrics

### `npm run ai:history`

View task history

## Examples

### Start a feature task

```bash
npm run ai:start feature auth,api "Add OAuth authentication"
```

### Complete a task

```bash
npm run ai:complete true "src/auth.ts,src/api.ts" 150 20 5 "Added OAuth flow"
```
