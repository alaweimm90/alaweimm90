#!/usr/bin/env python3
"""
mh - MetaHub Unified CLI

Single entry point for all portfolio tools:
  mh govern    - Governance and compliance
  mh validate  - Schema and structure validation
  mh security  - Security scanning
  mh config    - Configuration management
  mh ci        - CI/CD workflow tools
  mh log       - Structured logging utilities

Usage:
    mh --help
    mh govern enforce ./path
    mh validate schema config.yaml
    mh security scan-all ./
"""

import sys
from pathlib import Path

import click

# Add parent paths for imports
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / ".metaHub" / "tools"))


@click.group()
@click.version_option(version="1.0.0", prog_name="mh")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool):
    """MetaHub CLI - Unified portfolio tools"""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["root"] = ROOT


# ============================================================================
# GOVERN - Import from existing governance.py
# ============================================================================
@cli.group("govern")
@click.pass_context
def govern(ctx: click.Context):
    """Governance and compliance commands"""
    pass


@govern.command("enforce")
@click.argument("path", type=click.Path(exists=True))
@click.option("--strict", is_flag=True, help="Treat warnings as violations")
@click.option(
    "--report", "report_fmt", type=click.Choice(["text", "json"]), default="text"
)
@click.pass_context
def govern_enforce(ctx: click.Context, path: str, strict: bool, report_fmt: str):
    """Enforce governance policies on PATH"""
    from cli.governance import cmd_enforce

    ctx.invoke(
        cmd_enforce,
        path=path,
        strict=strict,
        report_fmt=report_fmt,
        output=None,
        fail_on_warnings=False,
        schema=None,
    )


@govern.command("catalog")
@click.option(
    "--format", "fmt", type=click.Choice(["text", "json", "yaml"]), default="text"
)
@click.pass_context
def govern_catalog(ctx: click.Context, fmt: str):
    """Generate service catalog"""
    from cli.governance import cmd_catalog

    ctx.invoke(
        cmd_catalog,
        path=str(ctx.obj["root"] / "organizations"),
        output_fmt=fmt,
        output=None,
        include_deps=True,
        include_metrics=True,
    )


@govern.command("audit")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.pass_context
def govern_audit(ctx: click.Context, output: str):
    """Run AI-powered governance audit"""
    from cli.governance import cmd_audit

    ctx.invoke(
        cmd_audit,
        path=str(ctx.obj["root"]),
        output=output,
        include_recommendations=True,
        severity_threshold="low",
    )


# ============================================================================
# VALIDATE - Schema and structure validation
# ============================================================================
@cli.group("validate")
@click.pass_context
def validate(ctx: click.Context):
    """Validation commands"""
    pass


@validate.command("schema")
@click.argument("file", type=click.Path(exists=True))
@click.option("--schema", "-s", type=click.Path(exists=True), help="JSON Schema file")
@click.pass_context
def validate_schema(ctx: click.Context, file: str, schema: str):
    """Validate FILE against JSON schema"""
    from lib.validation import Validator
    import yaml
    import json

    validator = Validator()
    file_path = Path(file)

    # Load data
    with open(file_path) as f:
        if file_path.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    # Load schema
    if schema:
        with open(schema) as f:
            schema_data = json.load(f)
    else:
        click.echo("No schema provided, skipping schema validation", err=True)
        return

    is_valid, errors = validator.validate_schema(data, schema_data)

    if is_valid:
        click.secho(f"✓ {file} is valid", fg="green")
    else:
        click.secho(f"✗ {file} validation failed:", fg="red")
        for err in errors:
            click.echo(f"  - {err}")
        sys.exit(1)


@validate.command("structure")
@click.argument("path", type=click.Path(exists=True))
@click.option("--tier", type=int, default=2, help="Tier level (1-4)")
@click.pass_context
def validate_structure(ctx: click.Context, path: str, tier: int):
    """Validate repository structure at PATH"""
    from lib.validation import Validator

    validator = Validator()
    requirements = validator.TIER_REQUIREMENTS.get(tier, validator.TIER_REQUIREMENTS[2])

    is_valid, errors = validator.validate_structure(Path(path), requirements)

    if is_valid:
        click.secho(f"✓ Structure valid for tier {tier}", fg="green")
    else:
        click.secho(f"✗ Structure validation failed:", fg="red")
        for err in errors:
            click.echo(f"  - {err}")
        sys.exit(1)


@validate.command("docker")
@click.argument("dockerfile", type=click.Path(exists=True))
@click.pass_context
def validate_docker(ctx: click.Context, dockerfile: str):
    """Validate Dockerfile security"""
    from lib.validation import Validator

    validator = Validator()

    with open(dockerfile) as f:
        content = f.read()

    is_valid, errors = validator.validate_dockerfile(content)

    if is_valid:
        click.secho(f"✓ {dockerfile} passes security checks", fg="green")
    else:
        click.secho(f"✗ {dockerfile} has security issues:", fg="red")
        for err in errors:
            click.echo(f"  - {err}")
        sys.exit(1)


# ============================================================================
# SECURITY - Security scanning
# ============================================================================
@cli.group("security")
@click.pass_context
def security(ctx: click.Context):
    """Security scanning commands"""
    pass


@security.command("scan-all")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.pass_context
def security_scan_all(ctx: click.Context, path: str, output: str):
    """Run all security scans on PATH"""
    import subprocess

    scripts_dir = ctx.obj["root"] / "tools" / "security"
    script = scripts_dir / "security-scan-all.sh"

    if not script.exists():
        click.secho(f"Security script not found: {script}", fg="red")
        sys.exit(1)

    click.echo("Running comprehensive security scan...")
    result = subprocess.run(["bash", str(script)], cwd=path, capture_output=False)
    sys.exit(result.returncode)


@security.command("scan-secrets")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.pass_context
def security_scan_secrets(ctx: click.Context, path: str):
    """Scan for exposed secrets"""
    import subprocess

    scripts_dir = ctx.obj["root"] / "tools" / "security"
    script = scripts_dir / "secret-scan.sh"

    if not script.exists():
        click.secho(f"Script not found: {script}", fg="red")
        sys.exit(1)

    result = subprocess.run(["bash", str(script)], cwd=path, capture_output=False)
    sys.exit(result.returncode)


@security.command("scan-deps")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.pass_context
def security_scan_deps(ctx: click.Context, path: str):
    """Scan dependencies for vulnerabilities"""
    import subprocess

    scripts_dir = ctx.obj["root"] / "tools" / "security"
    script = scripts_dir / "dependency-scan.sh"

    if not script.exists():
        click.secho(f"Script not found: {script}", fg="red")
        sys.exit(1)

    result = subprocess.run(["bash", str(script)], cwd=path, capture_output=False)
    sys.exit(result.returncode)


# ============================================================================
# CONFIG - Configuration management
# ============================================================================
@cli.group("config")
@click.pass_context
def config(ctx: click.Context):
    """Configuration management commands"""
    pass


@config.command("load")
@click.argument("file", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["json", "yaml"]), default=None)
@click.pass_context
def config_load(ctx: click.Context, file: str, fmt: str):
    """Load and validate configuration file"""
    import yaml
    import json

    file_path = Path(file)

    # Auto-detect format
    if fmt is None:
        if file_path.suffix in [".yaml", ".yml"]:
            fmt = "yaml"
        else:
            fmt = "json"

    try:
        with open(file_path) as f:
            if fmt == "yaml":
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        click.secho(f"✓ Loaded {file}", fg="green")
        click.echo(json.dumps(data, indent=2))
    except Exception as e:
        click.secho(f"✗ Failed to load {file}: {e}", fg="red")
        sys.exit(1)


@config.command("merge")
@click.argument("base", type=click.Path(exists=True))
@click.argument("overlay", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file")
@click.pass_context
def config_merge(ctx: click.Context, base: str, overlay: str, output: str):
    """Merge two configuration files (overlay on base)"""
    import yaml
    import json

    def deep_merge(base: dict, overlay: dict) -> dict:
        result = base.copy()
        for key, value in overlay.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def load_file(path: Path) -> dict:
        with open(path) as f:
            if path.suffix in [".yaml", ".yml"]:
                return yaml.safe_load(f)
            return json.load(f)

    base_data = load_file(Path(base))
    overlay_data = load_file(Path(overlay))
    merged = deep_merge(base_data, overlay_data)

    result = yaml.dump(merged, default_flow_style=False)

    if output:
        with open(output, "w") as f:
            f.write(result)
        click.secho(f"✓ Merged to {output}", fg="green")
    else:
        click.echo(result)


# ============================================================================
# CI - CI/CD workflow tools
# ============================================================================
@cli.group("ci")
@click.pass_context
def ci(ctx: click.Context):
    """CI/CD workflow commands"""
    pass


@ci.command("validate")
@click.argument("workflow", type=click.Path(exists=True))
@click.pass_context
def ci_validate(ctx: click.Context, workflow: str):
    """Validate GitHub Actions workflow file"""
    import yaml

    try:
        with open(workflow) as f:
            data = yaml.safe_load(f)

        errors = []

        # Basic validation
        if "on" not in data:
            errors.append("Missing 'on' trigger definition")
        if "jobs" not in data:
            errors.append("Missing 'jobs' section")

        if errors:
            click.secho(f"✗ {workflow} has issues:", fg="red")
            for err in errors:
                click.echo(f"  - {err}")
            sys.exit(1)
        else:
            click.secho(f"✓ {workflow} is valid", fg="green")
    except yaml.YAMLError as e:
        click.secho(f"✗ Invalid YAML: {e}", fg="red")
        sys.exit(1)


@ci.command("list-workflows")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.pass_context
def ci_list_workflows(ctx: click.Context, path: str):
    """List all GitHub Actions workflows"""
    workflows_dir = Path(path) / ".github" / "workflows"

    if not workflows_dir.exists():
        click.echo("No .github/workflows directory found")
        return

    for wf in sorted(workflows_dir.glob("*.yml")) + sorted(
        workflows_dir.glob("*.yaml")
    ):
        click.echo(f"  {wf.name}")


# ============================================================================
# LOG - Structured logging utilities
# ============================================================================
@cli.group("log")
@click.pass_context
def log(ctx: click.Context):
    """Logging utilities"""
    pass


@log.command("setup")
@click.option(
    "--level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), default="INFO"
)
@click.option("--json", "json_fmt", is_flag=True, help="Use JSON format")
@click.pass_context
def log_setup(ctx: click.Context, level: str, json_fmt: bool):
    """Show logging setup snippet"""
    if json_fmt:
        snippet = f"""import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({{
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }})

logging.basicConfig(level=logging.{level})
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.getLogger().handlers = [handler]
"""
    else:
        snippet = f"""import logging

logging.basicConfig(
    level=logging.{level},
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
"""
    click.echo(snippet)


# ============================================================================
# SUPABASE - Database and auth scaffolding
# ============================================================================
@cli.group("supabase")
@click.pass_context
def supabase(ctx: click.Context):
    """Supabase scaffolding commands"""
    pass


@supabase.command("init")
@click.argument("output_dir", type=click.Path(), default="./src/lib")
@click.pass_context
def supabase_init(ctx: click.Context, output_dir: str):
    """Generate Supabase client configuration"""
    templates_dir = ctx.obj["root"] / ".metaHub" / "tools" / "templates" / "supabase"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy client template
    template_file = templates_dir / "client.ts.template"
    if template_file.exists():
        output_file = output_path / "supabase.ts"
        output_file.write_text(template_file.read_text())
        click.secho(f"✓ Created {output_file}", fg="green")
    else:
        click.secho(f"✗ Template not found: {template_file}", fg="red")
        sys.exit(1)

    # Create .env.example
    env_example = output_path.parent / ".env.example"
    if not env_example.exists():
        env_content = """# Supabase Configuration
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
"""
        env_example.write_text(env_content)
        click.secho(f"✓ Created {env_example}", fg="green")

    click.echo("\nNext steps:")
    click.echo("  1. Update .env with your Supabase credentials")
    click.echo("  2. Run: mh supabase gen-types")
    click.echo("  3. Import: import { supabase } from '@/lib/supabase'")


@supabase.command("gen-auth")
@click.argument("output_dir", type=click.Path(), default="./src/services")
@click.pass_context
def supabase_gen_auth(ctx: click.Context, output_dir: str):
    """Generate authentication service"""
    templates_dir = ctx.obj["root"] / ".metaHub" / "tools" / "templates" / "supabase"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    template_file = templates_dir / "authService.ts.template"
    if template_file.exists():
        output_file = output_path / "authService.ts"
        output_file.write_text(template_file.read_text())
        click.secho(f"✓ Created {output_file}", fg="green")
    else:
        click.secho(f"✗ Template not found: {template_file}", fg="red")
        sys.exit(1)

    click.echo("\nUsage:")
    click.echo("  import { authService } from '@/services/authService';")
    click.echo("  await authService.signIn({ email, password });")


@supabase.command("gen-types")
@click.option("--project-id", "-p", help="Supabase project ID")
@click.argument("output", type=click.Path(), default="./src/lib/types.ts")
@click.pass_context
def supabase_gen_types(ctx: click.Context, project_id: str, output: str):
    """Generate TypeScript types from database schema"""
    import subprocess

    if not project_id:
        click.echo("Fetching types requires project ID or supabase CLI login...")

    cmd = ["npx", "supabase", "gen", "types", "typescript", "--local"]
    if project_id:
        cmd = [
            "npx",
            "supabase",
            "gen",
            "types",
            "typescript",
            "--project-id",
            project_id,
        ]

    click.echo(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            Path(output).write_text(result.stdout)
            click.secho(f"✓ Types written to {output}", fg="green")
        else:
            click.secho(f"✗ Error: {result.stderr}", fg="red")
    except FileNotFoundError:
        click.secho(
            "✗ npx not found. Install Node.js and run: npm i -g supabase", fg="red"
        )


# ============================================================================
# ANALYTICS - Analytics scaffolding
# ============================================================================
@cli.group("analytics")
@click.pass_context
def analytics_cmd(ctx: click.Context):
    """Analytics scaffolding commands"""
    pass


@analytics_cmd.command("init")
@click.argument("output_dir", type=click.Path(), default="./src/lib")
@click.option(
    "--provider", type=click.Choice(["ga4", "gtm", "plausible"]), default="ga4"
)
@click.pass_context
def analytics_init(ctx: click.Context, output_dir: str, provider: str):
    """Generate analytics configuration"""
    templates_dir = ctx.obj["root"] / ".metaHub" / "tools" / "templates" / "analytics"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    template_file = templates_dir / "analytics.ts.template"
    if template_file.exists():
        output_file = output_path / "analytics.ts"
        content = template_file.read_text()
        # Customize based on provider
        if provider != "ga4":
            content = content.replace(
                "providers: ['ga4']", f"providers: ['{provider}']"
            )
        output_file.write_text(content)
        click.secho(f"✓ Created {output_file}", fg="green")
    else:
        click.secho(f"✗ Template not found: {template_file}", fg="red")
        sys.exit(1)

    # Create .env.example entries
    env_vars = {
        "ga4": "VITE_GA_MEASUREMENT_ID=G-XXXXXXXXXX",
        "gtm": "VITE_GTM_ID=GTM-XXXXXXX",
        "plausible": "VITE_PLAUSIBLE_DOMAIN=yourdomain.com",
    }
    click.echo(f"\nAdd to .env:\n  {env_vars[provider]}")
    click.echo("\nUsage:")
    click.echo("  import { analytics } from '@/lib/analytics';")
    click.echo("  analytics.init();")
    click.echo("  analytics.track('button_click', { button_id: 'cta' });")


# ============================================================================
# STRIPE - Payment scaffolding
# ============================================================================
@cli.group("stripe")
@click.pass_context
def stripe_cmd(ctx: click.Context):
    """Stripe payment scaffolding commands"""
    pass


@stripe_cmd.command("gen-service")
@click.argument("output_dir", type=click.Path(), default="./src/services")
@click.pass_context
def stripe_gen_service(ctx: click.Context, output_dir: str):
    """Generate payment service"""
    templates_dir = ctx.obj["root"] / ".metaHub" / "tools" / "templates" / "stripe"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    template_file = templates_dir / "paymentService.ts.template"
    if template_file.exists():
        output_file = output_path / "paymentService.ts"
        output_file.write_text(template_file.read_text())
        click.secho(f"✓ Created {output_file}", fg="green")
    else:
        click.secho(f"✗ Template not found: {template_file}", fg="red")
        sys.exit(1)

    click.echo("\nNext steps:")
    click.echo("  1. Install: npm install @stripe/stripe-js")
    click.echo("  2. Add to .env: VITE_STRIPE_PUBLISHABLE_KEY=pk_...")
    click.echo("  3. Create API routes for /api/payments/*")


# ============================================================================
# TEMPLATE - Generic template scaffolding
# ============================================================================
@cli.group("template")
@click.pass_context
def template_cmd(ctx: click.Context):
    """Template scaffolding commands"""
    pass


@template_cmd.command("list")
@click.pass_context
def template_list(ctx: click.Context):
    """List available templates"""
    templates_dir = ctx.obj["root"] / ".metaHub" / "tools" / "templates"

    if not templates_dir.exists():
        click.echo("No templates directory found")
        return

    click.echo("Available templates:\n")
    for category in sorted(templates_dir.iterdir()):
        if category.is_dir():
            click.secho(f"  {category.name}/", fg="cyan")
            for tmpl in sorted(category.glob("*.template")):
                name = tmpl.stem  # Remove .template extension
                click.echo(f"    - {name}")


@template_cmd.command("apply")
@click.argument("template_path")
@click.argument("output", type=click.Path())
@click.option("--var", "-v", multiple=True, help="Variable substitution (key=value)")
@click.pass_context
def template_apply(ctx: click.Context, template_path: str, output: str, var: tuple):
    """Apply a template to OUTPUT"""
    templates_dir = ctx.obj["root"] / ".metaHub" / "tools" / "templates"

    # Find template
    template_file = templates_dir / f"{template_path}.template"
    if not template_file.exists():
        # Try with category prefix
        for cat in templates_dir.iterdir():
            if cat.is_dir():
                possible = cat / f"{template_path}.template"
                if possible.exists():
                    template_file = possible
                    break

    if not template_file.exists():
        click.secho(f"✗ Template not found: {template_path}", fg="red")
        click.echo("Run 'mh template list' to see available templates")
        sys.exit(1)

    # Read and substitute variables
    content = template_file.read_text()
    for v in var:
        if "=" in v:
            key, value = v.split("=", 1)
            content = content.replace(f"${{{key}}}", value)
            content = content.replace(f"{{${key}}}", value)

    # Write output
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    click.secho(f"✓ Created {output}", fg="green")


# ============================================================================
# ECOMMERCE - E-commerce scaffolding
# ============================================================================
@cli.group("ecommerce")
@click.pass_context
def ecommerce_cmd(ctx: click.Context):
    """E-commerce scaffolding commands"""
    pass


@ecommerce_cmd.command("gen-cart")
@click.argument("output_dir", type=click.Path(), default="./src/services")
@click.pass_context
def ecommerce_gen_cart(ctx: click.Context, output_dir: str):
    """Generate shopping cart service"""
    templates_dir = ctx.obj["root"] / ".metaHub" / "tools" / "templates" / "ecommerce"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    template_file = templates_dir / "cartService.ts.template"
    if template_file.exists():
        output_file = output_path / "cartService.ts"
        output_file.write_text(template_file.read_text())
        click.secho(f"✓ Created {output_file}", fg="green")
    else:
        click.secho(f"✗ Template not found", fg="red")
        sys.exit(1)

    click.echo("\nUsage:")
    click.echo("  import { cartService } from '@/services/cartService';")
    click.echo("  cartService.addItem({ id: '123', name: 'Product', price: 29.99 });")


@ecommerce_cmd.command("gen-order")
@click.argument("output_dir", type=click.Path(), default="./src/services")
@click.pass_context
def ecommerce_gen_order(ctx: click.Context, output_dir: str):
    """Generate order service"""
    templates_dir = ctx.obj["root"] / ".metaHub" / "tools" / "templates" / "ecommerce"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    template_file = templates_dir / "orderService.ts.template"
    if template_file.exists():
        output_file = output_path / "orderService.ts"
        output_file.write_text(template_file.read_text())
        click.secho(f"✓ Created {output_file}", fg="green")
    else:
        click.secho(f"✗ Template not found", fg="red")
        sys.exit(1)


@ecommerce_cmd.command("gen-checkout")
@click.argument("output_dir", type=click.Path(), default="./src/hooks")
@click.pass_context
def ecommerce_gen_checkout(ctx: click.Context, output_dir: str):
    """Generate checkout hook"""
    templates_dir = ctx.obj["root"] / ".metaHub" / "tools" / "templates" / "ecommerce"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    template_file = templates_dir / "useCheckout.ts.template"
    if template_file.exists():
        output_file = output_path / "useCheckout.ts"
        output_file.write_text(template_file.read_text())
        click.secho(f"✓ Created {output_file}", fg="green")
    else:
        click.secho(f"✗ Template not found", fg="red")
        sys.exit(1)

    click.echo("\nUsage:")
    click.echo("  import { useCheckout } from '@/hooks/useCheckout';")
    click.echo("  const { cart, goToStep, placeOrder } = useCheckout();")


@ecommerce_cmd.command("init")
@click.argument("output_dir", type=click.Path(), default="./src")
@click.pass_context
def ecommerce_init(ctx: click.Context, output_dir: str):
    """Generate all e-commerce services"""
    ctx.invoke(ecommerce_gen_cart, output_dir=f"{output_dir}/services")
    ctx.invoke(ecommerce_gen_order, output_dir=f"{output_dir}/services")
    ctx.invoke(ecommerce_gen_checkout, output_dir=f"{output_dir}/hooks")

    click.echo("\n✓ E-commerce scaffolding complete!")
    click.echo("  Generated: cartService, orderService, useCheckout")


# ============================================================================
# PREDICT - ML prediction pipelines
# ============================================================================
@cli.group("predict")
@click.pass_context
def predict_cmd(ctx: click.Context):
    """ML prediction pipeline commands"""
    pass


@predict_cmd.command("init")
@click.argument("output_dir", type=click.Path(), default="./ml")
@click.pass_context
def predict_init(ctx: click.Context, output_dir: str):
    """Generate ML prediction pipeline"""
    templates_dir = ctx.obj["root"] / ".metaHub" / "tools" / "templates" / "predict"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    template_file = templates_dir / "pipeline.py.template"
    if template_file.exists():
        output_file = output_path / "pipeline.py"
        output_file.write_text(template_file.read_text())
        click.secho(f"✓ Created {output_file}", fg="green")
    else:
        click.secho(f"✗ Template not found", fg="red")
        sys.exit(1)

    # Create requirements
    reqs = output_path / "requirements.txt"
    reqs.write_text(
        """# ML Pipeline dependencies
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
"""
    )
    click.secho(f"✓ Created {reqs}", fg="green")

    click.echo("\nNext steps:")
    click.echo("  1. pip install -r ml/requirements.txt")
    click.echo("  2. Edit ml/pipeline.py with your model")
    click.echo("  3. Run: python ml/pipeline.py")


@predict_cmd.command("train")
@click.argument("data_file", type=click.Path(exists=True))
@click.option("--target", "-t", required=True, help="Target column name")
@click.option("--model", "-m", default="random_forest", help="Model type")
@click.option("--output", "-o", default="./models", help="Output directory")
@click.pass_context
def predict_train(
    ctx: click.Context, data_file: str, target: str, model: str, output: str
):
    """Train a model on DATA_FILE"""
    click.echo(f"Training {model} model on {data_file}...")
    click.echo(f"Target column: {target}")
    click.echo(f"Output: {output}")

    # This is a placeholder - actual training would import the pipeline
    click.secho(
        "\n⚠ Run the generated pipeline.py directly for full training:", fg="yellow"
    )
    click.echo(f"  python {output}/pipeline.py --data {data_file} --target {target}")


# ============================================================================
# ORCHESTRATE - Workflow orchestration (from legacy scripts)
# ============================================================================
@cli.group("orchestrate")
@click.pass_context
def orchestrate_cmd(ctx: click.Context):
    """Workflow orchestration commands (checkpoints, telemetry, recovery)"""
    pass


@orchestrate_cmd.command("checkpoint")
@click.argument("action", type=click.Choice(["create", "restore", "list", "validate"]))
@click.option("--workflow", "-w", help="Workflow name")
@click.option("--id", "checkpoint_id", help="Checkpoint ID (for restore/validate)")
@click.pass_context
def orchestrate_checkpoint(
    ctx: click.Context, action: str, workflow: str, checkpoint_id: str
):
    """Manage workflow checkpoints"""
    legacy_dir = ctx.obj["root"] / "tools" / "legacy" / "orchestration"
    script = legacy_dir / "orchestration_checkpoint.py"

    if not script.exists():
        click.secho(f"✗ Legacy script not found: {script}", fg="red")
        sys.exit(1)

    import subprocess

    cmd = ["python", str(script), action]
    if workflow:
        cmd.extend(["--workflow", workflow])
    if checkpoint_id:
        cmd.extend(["--id", checkpoint_id])

    result = subprocess.run(cmd, capture_output=False)
    sys.exit(result.returncode)


@orchestrate_cmd.command("telemetry")
@click.argument("action", type=click.Choice(["record", "report", "dashboard"]))
@click.option("--event", help="Event type (for record)")
@click.option("--tool", help="Tool name")
@click.option("--status", help="Event status")
@click.option("--period", default="24h", help="Report period (for report)")
@click.pass_context
def orchestrate_telemetry(
    ctx: click.Context, action: str, event: str, tool: str, status: str, period: str
):
    """Collect and view orchestration metrics"""
    legacy_dir = ctx.obj["root"] / "tools" / "legacy" / "orchestration"
    script = legacy_dir / "orchestration_telemetry.py"

    if not script.exists():
        click.secho(f"✗ Legacy script not found: {script}", fg="red")
        sys.exit(1)

    import subprocess

    cmd = ["python", str(script), action]
    if event:
        cmd.extend(["--event", event])
    if tool:
        cmd.extend(["--tool", tool])
    if status:
        cmd.extend(["--status", status])
    if action == "report":
        cmd.extend(["--period", period])

    result = subprocess.run(cmd, capture_output=False)
    sys.exit(result.returncode)


@orchestrate_cmd.command("recover")
@click.option("--workflow", "-w", required=True, help="Workflow name")
@click.option("--from-checkpoint", "checkpoint_id", help="Checkpoint to restore from")
@click.pass_context
def orchestrate_recover(ctx: click.Context, workflow: str, checkpoint_id: str):
    """Recover a failed workflow"""
    legacy_dir = ctx.obj["root"] / "tools" / "legacy" / "orchestration"
    script = legacy_dir / "self_healing_workflow.py"

    if not script.exists():
        click.secho(f"✗ Legacy script not found: {script}", fg="red")
        sys.exit(1)

    import subprocess

    cmd = ["python", str(script), "recover", "--workflow", workflow]
    if checkpoint_id:
        cmd.extend(["--from-checkpoint", checkpoint_id])

    result = subprocess.run(cmd, capture_output=False)
    sys.exit(result.returncode)


@orchestrate_cmd.command("status")
@click.option("--workflow", "-w", help="Workflow name (or all)")
@click.pass_context
def orchestrate_status(ctx: click.Context, workflow: str):
    """Show workflow status"""
    legacy_dir = ctx.obj["root"] / "tools" / "legacy" / "orchestration"
    script = legacy_dir / "self_healing_workflow.py"

    if not script.exists():
        click.secho(f"✗ Legacy script not found: {script}", fg="red")
        sys.exit(1)

    import subprocess

    cmd = ["python", str(script), "status"]
    if workflow:
        cmd.extend(["--workflow", workflow])

    result = subprocess.run(cmd, capture_output=False)
    sys.exit(result.returncode)


@orchestrate_cmd.command("degrade")
@click.option("--level", "-l", type=int, default=2, help="Degradation level (1-4)")
@click.pass_context
def orchestrate_degrade(ctx: click.Context, level: int):
    """Set graceful degradation level"""
    legacy_dir = ctx.obj["root"] / "tools" / "legacy" / "orchestration"
    script = legacy_dir / "self_healing_workflow.py"

    if not script.exists():
        click.secho(f"✗ Legacy script not found: {script}", fg="red")
        sys.exit(1)

    import subprocess

    cmd = ["python", str(script), "degrade", "--level", str(level)]

    result = subprocess.run(cmd, capture_output=False)
    sys.exit(result.returncode)


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
