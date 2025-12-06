"""
Database storage for failures using SQLite
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from hall_of_failures.models import Failure, FailureType, Severity

Base = declarative_base()


class FailureRecord(Base):
    """SQLAlchemy model for failures table"""
    __tablename__ = "failures"

    id = Column(String, primary_key=True)
    hypothesis = Column(Text, nullable=False)
    failure_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    context = Column(Text, nullable=False)  # JSON string
    lessons_learned = Column(Text, nullable=False)  # JSON array
    prevention_strategies = Column(Text, nullable=False)  # JSON array
    root_causes = Column(Text, nullable=False)  # JSON array
    similarity_hash = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

    def to_failure(self) -> Failure:
        """Convert to Failure model"""
        return Failure(
            id=self.id,
            hypothesis=self.hypothesis,
            failure_type=FailureType(self.failure_type),
            severity=Severity(self.severity),
            description=self.description,
            context=json.loads(self.context),
            lessons_learned=json.loads(self.lessons_learned),
            prevention_strategies=json.loads(self.prevention_strategies),
            root_causes=json.loads(self.root_causes),
            similarity_hash=self.similarity_hash,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_failure(cls, failure: Failure) -> "FailureRecord":
        """Create from Failure model"""
        return cls(
            id=failure.id,
            hypothesis=failure.hypothesis,
            failure_type=failure.failure_type.value,
            severity=failure.severity.value,
            description=failure.description,
            context=json.dumps(failure.context),
            lessons_learned=json.dumps(failure.lessons_learned),
            prevention_strategies=json.dumps(failure.prevention_strategies),
            root_causes=json.dumps(failure.root_causes),
            similarity_hash=failure.similarity_hash or failure.compute_similarity_hash(),
            created_at=failure.created_at,
            updated_at=failure.updated_at,
        )


class FailureDatabase:
    """
    SQLite database for storing failures

    Features:
    - CRUD operations
    - Querying by failure type, severity, date
    - Full-text search
    - Similarity hash lookup
    """

    def __init__(self, db_path: str = "failures.db"):
        """
        Initialize database

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables
        Base.metadata.create_all(self.engine)

        # Create indices
        self._create_indices()

    def _create_indices(self):
        """Create database indices for performance"""
        with self.engine.connect() as conn:
            # Check if indices exist before creating
            try:
                conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_failure_type ON failures(failure_type)"))
                conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_similarity_hash ON failures(similarity_hash)"))
                conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_created_at ON failures(created_at)"))
                conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_severity ON failures(severity)"))
                conn.commit()
            except Exception:
                pass  # Indices may already exist

    def add_failure(self, failure: Failure) -> bool:
        """
        Add failure to database

        Args:
            failure: Failure to add

        Returns:
            True if successful
        """
        with self.SessionLocal() as session:
            try:
                # Ensure hash is computed
                if not failure.similarity_hash:
                    failure.similarity_hash = failure.compute_similarity_hash()

                record = FailureRecord.from_failure(failure)
                session.add(record)
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                print(f"Error adding failure: {e}")
                return False

    def get_failure(self, failure_id: str) -> Optional[Failure]:
        """Get failure by ID"""
        with self.SessionLocal() as session:
            record = session.query(FailureRecord).filter_by(id=failure_id).first()
            return record.to_failure() if record else None

    def get_all_failures(self, limit: Optional[int] = None) -> List[Failure]:
        """Get all failures"""
        with self.SessionLocal() as session:
            query = session.query(FailureRecord).order_by(FailureRecord.created_at.desc())

            if limit:
                query = query.limit(limit)

            return [record.to_failure() for record in query.all()]

    def query_failures(
        self,
        failure_type: Optional[FailureType] = None,
        severity: Optional[Severity] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Failure]:
        """
        Query failures with filters

        Args:
            failure_type: Filter by failure type
            severity: Filter by severity
            since: Filter by date (created after)
            limit: Limit results

        Returns:
            List of matching failures
        """
        with self.SessionLocal() as session:
            query = session.query(FailureRecord)

            if failure_type:
                query = query.filter(FailureRecord.failure_type == failure_type.value)

            if severity:
                query = query.filter(FailureRecord.severity == severity.value)

            if since:
                query = query.filter(FailureRecord.created_at >= since)

            query = query.order_by(FailureRecord.created_at.desc())

            if limit:
                query = query.limit(limit)

            return [record.to_failure() for record in query.all()]

    def search_failures(self, search_text: str, limit: int = 20) -> List[Failure]:
        """
        Full-text search across hypothesis and description

        Args:
            search_text: Text to search for
            limit: Maximum results

        Returns:
            List of matching failures
        """
        with self.SessionLocal() as session:
            # Simple text search (for full-text search, use FTS5 in SQLite)
            search_pattern = f"%{search_text}%"

            query = session.query(FailureRecord).filter(
                sa.or_(
                    FailureRecord.hypothesis.like(search_pattern),
                    FailureRecord.description.like(search_pattern)
                )
            ).order_by(FailureRecord.created_at.desc()).limit(limit)

            return [record.to_failure() for record in query.all()]

    def find_by_similarity_hash(self, similarity_hash: str, limit: int = 10) -> List[Failure]:
        """Find failures with matching similarity hash"""
        with self.SessionLocal() as session:
            query = session.query(FailureRecord).filter(
                FailureRecord.similarity_hash == similarity_hash
            ).order_by(FailureRecord.created_at.desc()).limit(limit)

            return [record.to_failure() for record in query.all()]

    def update_failure(self, failure: Failure) -> bool:
        """Update existing failure"""
        with self.SessionLocal() as session:
            try:
                record = session.query(FailureRecord).filter_by(id=failure.id).first()

                if not record:
                    return False

                # Update fields
                record.hypothesis = failure.hypothesis
                record.failure_type = failure.failure_type.value
                record.severity = failure.severity.value
                record.description = failure.description
                record.context = json.dumps(failure.context)
                record.lessons_learned = json.dumps(failure.lessons_learned)
                record.prevention_strategies = json.dumps(failure.prevention_strategies)
                record.root_causes = json.dumps(failure.root_causes)
                record.updated_at = datetime.now()

                session.commit()
                return True
            except Exception as e:
                session.rollback()
                print(f"Error updating failure: {e}")
                return False

    def delete_failure(self, failure_id: str) -> bool:
        """Delete failure by ID"""
        with self.SessionLocal() as session:
            try:
                record = session.query(FailureRecord).filter_by(id=failure_id).first()

                if not record:
                    return False

                session.delete(record)
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                print(f"Error deleting failure: {e}")
                return False

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.SessionLocal() as session:
            total = session.query(FailureRecord).count()

            # Count by type
            by_type = {}
            for ft in FailureType:
                count = session.query(FailureRecord).filter_by(failure_type=ft.value).count()
                by_type[ft.value] = count

            # Count by severity
            by_severity = {}
            for sev in Severity:
                count = session.query(FailureRecord).filter_by(severity=sev.value).count()
                by_severity[sev.value] = count

            # Recent failures
            recent_count = session.query(FailureRecord).filter(
                FailureRecord.created_at >= datetime.now()
            ).count()

            return {
                "total_failures": total,
                "by_type": by_type,
                "by_severity": by_severity,
                "database_path": str(self.db_path),
            }

    def clear_all(self) -> bool:
        """Clear all failures (use with caution!)"""
        with self.SessionLocal() as session:
            try:
                session.query(FailureRecord).delete()
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                print(f"Error clearing database: {e}")
                return False
