# Configuration Directory

This directory contains all centralized configuration files for the repository, organized by concern for better maintainability and clarity.

## Directory Structure

### `linters/`
Linting configuration files for code quality tools:
- `.yamllint.yaml` - YAML file linting rules
- `.pre-commit-config.yaml` - Pre-commit hooks configuration

### `formatters/`
Code formatting configuration files:
- `.prettierrc` - Prettier formatting rules (if present)

### `ci-cd/`
Continuous Integration and Continuous Deployment configurations:
- `codecov.yml` - Code coverage reporting configuration

### `docker/`
Containerization configurations:
- `docker-compose.yml` - Docker Compose services configuration (if present)

### `monitoring/`
Monitoring and observability configurations:
- `prometheus.yml` - Prometheus monitoring configuration

### `docs/`
Documentation tool configurations:
- `mkdocs.yaml` - MkDocs documentation site configuration (if present)

### `nodejs/`
Node.js and TypeScript configurations:
- `package.json` - Node.js project configuration (if present)
- `tsconfig.json` - TypeScript compiler configuration (if present)

## Migration Notes

This configuration structure was implemented as part of repository organization improvements to:

1. **Reduce Root Clutter**: Previously scattered configuration files in the root directory
2. **Improve Discoverability**: Logical grouping by concern makes configuration files easier to find
3. **Enable Team Standards**: Consistent structure across all projects

## Tool Compatibility

All tools that previously used root-level configuration files should continue to work by:

1. **Pre-commit hooks**: Automatically detect `.pre-commit-config.yaml` in config directory
2. **Docker Compose**: Use `-f config/docker/docker-compose.yml` if moved
3. **Node.js tools**: May need path updates in scripts or package.json references

## Adding New Configurations

When adding new configuration files:

1. Identify the appropriate subdirectory based on tool purpose
2. Follow existing naming conventions
3. Update this README if adding new categories
4. Consider creating a symlink if the tool requires root-level presence

## Legacy Support

Some tools may require configuration files to be in the root directory. In such cases:

1. Keep the primary configuration in the appropriate config subdirectory
2. Create a symbolic link (on Unix systems) or copy the file to root if necessary
3. Document the tool's requirements and workaround approach

---

**Note**: This directory was created as part of the Repository Organization Analysis initiative. See `docs/REPOSITORY_ORGANIZATION_ANALYSIS.md` for the complete reorganization plan.
