"""
Database migration management with Alembic integration and version control.
"""

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import MetaData, Table, create_engine, text

logger = logging.getLogger(__name__)


class MigrationManager:
    """
    Database migration manager with version control,
    rollback support, and automated migration generation.
    """

    def __init__(
        self,
        database_url: str,
        migrations_dir: str = "migrations"
    ):
        """Initialize migration manager."""
        self.database_url = database_url
        self.migrations_dir = Path(migrations_dir)
        self.migrations_dir.mkdir(exist_ok=True)

        self.engine = create_engine(database_url)
        self.metadata = MetaData()

        # Initialize Alembic
        self._init_alembic()

    def _init_alembic(self) -> None:
        """Initialize Alembic configuration."""
        self.alembic_cfg = Config()
        self.alembic_cfg.set_main_option("script_location", str(self.migrations_dir))
        self.alembic_cfg.set_main_option("sqlalchemy.url", self.database_url)

        # Create alembic.ini if not exists
        ini_path = self.migrations_dir / "alembic.ini"
        if not ini_path.exists():
            self._create_alembic_ini()

    def _create_alembic_ini(self) -> None:
        """Create Alembic configuration file."""
        ini_content = f"""
[alembic]
script_location = {self.migrations_dir}
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = {self.database_url}

[post_write_hooks]
hooks = black
black.type = console_scripts
black.entrypoint = black
black.options = -l 100

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""
        ini_path = self.migrations_dir / "alembic.ini"
        ini_path.write_text(ini_content)

    async def create_migration(
        self,
        message: str,
        auto_generate: bool = True
    ) -> str:
        """
        Create a new migration.

        Args:
            message: Migration description
            auto_generate: Auto-generate migration from model changes

        Returns:
            Migration ID
        """
        try:
            if auto_generate:
                command.revision(
                    self.alembic_cfg,
                    message=message,
                    autogenerate=True
                )
            else:
                command.revision(
                    self.alembic_cfg,
                    message=message
                )

            migration_id = self._get_latest_migration_id()
            logger.info(f"Created migration {migration_id}: {message}")
            return migration_id

        except Exception as e:
            logger.error(f"Failed to create migration: {e}")
            raise

    async def apply_migrations(
        self,
        target: Optional[str] = None
    ) -> List[str]:
        """
        Apply pending migrations.

        Args:
            target: Target migration version (None for latest)

        Returns:
            List of applied migration IDs
        """
        try:
            current = self._get_current_version()
            command.upgrade(self.alembic_cfg, target or "head")
            new_version = self._get_current_version()

            logger.info(f"Applied migrations from {current} to {new_version}")
            return [new_version] if new_version != current else []

        except Exception as e:
            logger.error(f"Failed to apply migrations: {e}")
            raise

    async def rollback_migration(
        self,
        steps: int = 1
    ) -> str:
        """
        Rollback migrations.

        Args:
            steps: Number of migrations to rollback

        Returns:
            New current version
        """
        try:
            command.downgrade(self.alembic_cfg, f"-{steps}")
            current = self._get_current_version()

            logger.info(f"Rolled back {steps} migration(s). Current: {current}")
            return current

        except Exception as e:
            logger.error(f"Failed to rollback migration: {e}")
            raise

    async def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get migration history."""
        history = []

        with self.engine.connect() as conn:
            context = MigrationContext.configure(conn)
            current = context.get_current_revision()

            # Get all migrations
            migrations_path = self.migrations_dir / "versions"
            if migrations_path.exists():
                for migration_file in migrations_path.glob("*.py"):
                    # Parse migration file for metadata
                    content = migration_file.read_text()

                    # Extract revision ID
                    import re
                    revision_match = re.search(r"revision = '([^']+)'", content)
                    down_revision_match = re.search(r"down_revision = '([^']+)'", content)

                    if revision_match:
                        revision = revision_match.group(1)
                        down_revision = down_revision_match.group(1) if down_revision_match else None

                        history.append({
                            "id": revision,
                            "parent": down_revision,
                            "applied": revision == current or revision < current if current else False,
                            "file": migration_file.name,
                            "created_at": datetime.fromtimestamp(migration_file.stat().st_ctime)
                        })

        return sorted(history, key=lambda x: x["created_at"], reverse=True)

    async def validate_migrations(self) -> Dict[str, Any]:
        """Validate migration consistency."""
        issues = []

        try:
            # Check if database is up to date
            with self.engine.connect() as conn:
                context = MigrationContext.configure(conn)
                current = context.get_current_revision()

            # Get latest migration
            latest = self._get_latest_migration_id()

            if current != latest:
                issues.append({
                    "type": "outdated",
                    "message": f"Database at {current}, latest is {latest}"
                })

            # Check for migration conflicts
            # (simplified - real implementation would check dependency graph)

            return {
                "valid": len(issues) == 0,
                "current_version": current,
                "latest_version": latest,
                "issues": issues
            }

        except Exception as e:
            logger.error(f"Migration validation failed: {e}")
            return {
                "valid": False,
                "issues": [{"type": "error", "message": str(e)}]
            }

    def _get_current_version(self) -> Optional[str]:
        """Get current migration version."""
        with self.engine.connect() as conn:
            context = MigrationContext.configure(conn)
            return context.get_current_revision()

    def _get_latest_migration_id(self) -> Optional[str]:
        """Get latest migration ID."""
        migrations_path = self.migrations_dir / "versions"
        if not migrations_path.exists():
            return None

        latest_file = max(
            migrations_path.glob("*.py"),
            key=lambda p: p.stat().st_ctime,
            default=None
        )

        if latest_file:
            content = latest_file.read_text()
            import re
            match = re.search(r"revision = '([^']+)'", content)
            return match.group(1) if match else None

        return None

    async def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create database backup before migration."""
        backup_name = backup_name or f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # This is database-specific
        # PostgreSQL example:
        if "postgresql" in self.database_url:
            import subprocess
            backup_file = f"{backup_name}.sql"

            subprocess.run([
                "pg_dump",
                self.database_url,
                "-f", backup_file
            ])

            logger.info(f"Created backup: {backup_file}")
            return backup_file

        return ""

    async def get_stats(self) -> Dict[str, Any]:
        """Get migration statistics."""
        history = await self.get_migration_history()
        validation = await self.validate_migrations()

        return {
            "total_migrations": len(history),
            "applied_migrations": sum(1 for m in history if m["applied"]),
            "pending_migrations": sum(1 for m in history if not m["applied"]),
            "current_version": validation["current_version"],
            "latest_version": validation["latest_version"],
            "is_up_to_date": validation["valid"]
        }