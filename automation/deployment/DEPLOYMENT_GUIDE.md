# Deployment Guide

Comprehensive guide for deploying portfolios, knowledge bases, and web interfaces.

## Quick Start

### Deploy a Portfolio

```bash
# Using atlas CLI
atlas deploy portfolio --source ./my-portfolio --platform netlify

# Using Python
python -m automation.deployment.portfolio ./my-portfolio
```

### Deploy a Knowledge Base

```bash
# Organize and deploy Downloads folder
atlas deploy knowledge-base --source ~/Downloads

# Using Python
python -m automation.deployment.knowledge_base ~/Downloads
```

### Generate Web Interface

```python
from automation.deployment import WebInterfaceGenerator, WebConfig

config = WebConfig(
    title="My Site",
    description="A beautiful web interface",
    primary_color="#6366f1"
)

generator = WebInterfaceGenerator("./output", config)
generator.generate_dashboard(sections=[...], stats={...})
```

## Portfolio Deployment

### Supported Platforms

| Platform | Command | Requirements |
|----------|---------|--------------|
| **Netlify** | `--platform netlify` | `netlify-cli` installed |
| **Vercel** | `--platform vercel` | `vercel` CLI installed |
| **GitHub Pages** | `--platform github-pages` | Git repository |
| **Local** | `--platform local` | None |

### Configuration

Create `portfolio.config.json`:

```json
{
  "name": "my-portfolio",
  "platform": "netlify",
  "build_command": "npm run build",
  "publish_dir": "dist",
  "environment": "production",
  "accessibility_level": "AA",
  "domain": "my-portfolio.netlify.app"
}
```

### Accessibility Enhancements

The deployer automatically applies these accessibility improvements:

1. **Language attribute** - Adds `lang="en"` to HTML
2. **Viewport meta** - Ensures responsive design
3. **Skip links** - Adds "Skip to main content" link
4. **Focus indicators** - Ensures visible focus states
5. **ARIA labels** - Adds missing accessibility labels

### Deployment Pipeline

```
Source → Validate → Build → Accessibility → Manifest → Deploy
```

1. **Validate**: Check source files exist
2. **Build**: Run build command (npm, etc.)
3. **Accessibility**: Apply WCAG enhancements
4. **Manifest**: Generate deployment manifest
5. **Deploy**: Push to platform

## Knowledge Base Deployment

### File Categories

Files are automatically organized into:

| Category | Extensions |
|----------|------------|
| **Documents** | .md, .txt, .pdf, .doc, .docx |
| **Code** | .py, .js, .ts, .java, .cpp, .go |
| **Data** | .json, .yaml, .csv, .sql |
| **Images** | .png, .jpg, .gif, .svg, .webp |
| **Archives** | .zip, .tar, .gz, .rar |
| **Config** | .env, .ini, .toml |
| **Web** | .html, .css, .scss |
| **Notebooks** | .ipynb |
| **Media** | .mp3, .mp4, .wav |

### Web Interface Features

- **Search**: Real-time file search
- **Categories**: Visual category grouping
- **Dark mode**: Automatic theme switching
- **Accessibility**: WCAG AA compliant
- **Responsive**: Works on all devices

### Deployment Options

```bash
# Dry run (preview changes)
python -m automation.deployment.knowledge_base ~/Downloads --dry-run

# Custom output directory
python -m automation.deployment.knowledge_base ~/Downloads --output ./organized

# Generate web interface only
python -m automation.deployment.knowledge_base ~/Downloads --web-only
```

## Web Interface Generator

### Available Templates

1. **Dashboard** - Stats and sections layout
2. **File Browser** - Searchable file listing
3. **Documentation** - Sidebar navigation docs
4. **Portfolio** - Project showcase

### Configuration Options

```python
WebConfig(
    title="Site Title",
    description="Site description",
    theme="auto",           # light, dark, auto
    primary_color="#3498db",
    logo_url="/logo.png",
    footer_text="© 2024",
    enable_search=True,
    enable_dark_mode=True,
    accessibility_level="AA"  # A, AA, AAA
)
```

### Accessibility Features

All generated interfaces include:

- **Skip links** - Keyboard navigation
- **ARIA labels** - Screen reader support
- **Focus indicators** - Visible focus states
- **Color contrast** - WCAG compliant colors
- **Reduced motion** - Respects user preferences
- **Semantic HTML** - Proper heading hierarchy

## CLI Reference

### Portfolio Commands

```bash
# Deploy portfolio
atlas deploy portfolio --source <path> [options]

Options:
  --platform <name>     Deployment platform (netlify, vercel, github-pages)
  --build <command>     Build command to run
  --output <path>       Output directory
  --domain <url>        Custom domain
  --dry-run             Preview without deploying
```

### Knowledge Base Commands

```bash
# Deploy knowledge base
atlas deploy knowledge-base --source <path> [options]

Options:
  --output <path>       Output directory for organized files
  --web-only            Only generate web interface
  --dry-run             Preview organization without moving files
  --no-web              Skip web interface generation
```

### Web Generator Commands

```bash
# Generate web interface
atlas generate web --type <template> --output <path> [options]

Templates:
  dashboard             Stats and sections
  file-browser          Searchable files
  documentation         Docs with sidebar
  portfolio             Project showcase

Options:
  --title <text>        Page title
  --theme <name>        Color theme
  --config <file>       Configuration file
```

## Troubleshooting

### Common Issues

**Build fails with npm error**
```bash
# Ensure Node.js is installed
node --version

# Clear npm cache
npm cache clean --force
```

**Netlify CLI not found**
```bash
# Install Netlify CLI
npm install -g netlify-cli

# Login to Netlify
netlify login
```

**Permission denied on output directory**
```bash
# Check permissions
ls -la ./output

# Fix permissions
chmod 755 ./output
```

### Debug Mode

```bash
# Enable verbose logging
atlas deploy portfolio --source ./site --verbose

# Check deployment logs
cat ./deploy.log
```

## Best Practices

### Portfolio Deployment

1. **Test locally first** - Run build and preview
2. **Check accessibility** - Use Lighthouse audit
3. **Optimize images** - Compress before deploy
4. **Use environment variables** - Don't hardcode secrets
5. **Set up CI/CD** - Automate deployments

### Knowledge Base Organization

1. **Regular cleanup** - Run weekly
2. **Backup first** - Keep original files safe
3. **Review categories** - Customize for your needs
4. **Update index** - Regenerate after changes
5. **Share web interface** - Host for team access

### Web Interface Design

1. **Mobile first** - Design for small screens
2. **Test accessibility** - Use screen reader
3. **Optimize performance** - Minimize assets
4. **Support dark mode** - Respect user preference
5. **Add analytics** - Track usage (optional)
