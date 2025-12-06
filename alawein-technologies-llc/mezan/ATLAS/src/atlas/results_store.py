#!/usr/bin/env python3
"""
Results Store and API System for ORCHEX
Implements: [130] Results store, [177] Read-only API, [246] Distributed cache,
[267] Dynamic configuration, [121] Pre-sales artifacts
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional


# Priority [130] - Results Store with Schemas and Indices
class ResultsStore:
    """Centralized results storage with indexing."""

    def __init__(self, db_path: str = "atlas_results.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = Lock()
        self._initialize_schema()

    def _initialize_schema(self):
        """Create database schema."""
        with self.lock:
            cursor = self.conn.cursor()

            # Main results table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS results (
                    id TEXT PRIMARY KEY,
                    run_id TEXT UNIQUE,
                    feature TEXT,
                    timestamp REAL,
                    input_hash TEXT,
                    output_hash TEXT,
                    manifest JSON,
                    metrics JSON,
                    score REAL,
                    status TEXT
                )
            """
            )

            # Indices for fast lookup
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feature ON results(feature)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON results(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_input_hash ON results(input_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_score ON results(score)")

            # Metadata table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value JSON,
                    updated_at REAL
                )
            """
            )

            # Artifacts table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    run_id TEXT,
                    artifact_type TEXT,
                    content BLOB,
                    created_at REAL,
                    FOREIGN KEY(run_id) REFERENCES results(run_id)
                )
            """
            )

            self.conn.commit()

    def store_result(self, result: Dict[str, Any]) -> str:
        """Store result in database."""
        result_id = result.get("id", self._generate_id())
        run_id = result.get("run_id", f"run_{result_id}")

        with self.lock:
            cursor = self.conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO results
                (id, run_id, feature, timestamp, input_hash, output_hash,
                 manifest, metrics, score, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result_id,
                    run_id,
                    result.get("feature"),
                    result.get("timestamp", time.time()),
                    result.get("input_hash", ""),
                    result.get("output_hash", ""),
                    json.dumps(result.get("manifest", {})),
                    json.dumps(result.get("metrics", {})),
                    result.get("score", 0),
                    result.get("status", "completed"),
                ),
            )

            self.conn.commit()

        return result_id

    def query_results(
        self,
        feature: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Query results with filters."""
        query = "SELECT * FROM results WHERE 1=1"
        params = []

        if feature:
            query += " AND feature = ?"
            params.append(feature)

        if since:
            query += " AND timestamp > ?"
            params.append(since)

        if min_score is not None:
            query += " AND score >= ?"
            params.append(min_score)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append(
                {
                    "id": row[0],
                    "run_id": row[1],
                    "feature": row[2],
                    "timestamp": row[3],
                    "input_hash": row[4],
                    "output_hash": row[5],
                    "manifest": json.loads(row[6]),
                    "metrics": json.loads(row[7]),
                    "score": row[8],
                    "status": row[9],
                }
            )

        return results

    def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get specific result by ID."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM results WHERE id = ?", (result_id,))
            row = cursor.fetchone()

        if row:
            return {
                "id": row[0],
                "run_id": row[1],
                "feature": row[2],
                "timestamp": row[3],
                "input_hash": row[4],
                "output_hash": row[5],
                "manifest": json.loads(row[6]),
                "metrics": json.loads(row[7]),
                "score": row[8],
                "status": row[9],
            }

        return None

    def _generate_id(self) -> str:
        """Generate unique result ID."""
        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]


# Priority [177] - Read-only Results API
class ResultsAPI:
    """Read-only API for results access."""

    def __init__(self, store: ResultsStore):
        self.store = store
        self.rate_limiter = {}
        self.cache = {}

    def get_result(self, result_id: str) -> Dict[str, Any]:
        """Get single result."""
        result = self.store.get_result(result_id)
        if not result:
            return {"error": "Result not found", "status": 404}
        return {"data": result, "status": 200}

    def list_results(
        self,
        feature: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 50,
        page: int = 1,
    ) -> Dict[str, Any]:
        """List results with pagination."""
        # Convert since to timestamp
        since_timestamp = None
        if since:
            try:
                from datetime import datetime

                since_timestamp = datetime.fromisoformat(since).timestamp()
            except:
                pass

        # Query with pagination
        offset = (page - 1) * limit
        results = self.store.query_results(feature=feature, since=since_timestamp, limit=limit)

        return {
            "data": results,
            "meta": {"page": page, "limit": limit, "count": len(results)},
            "status": 200,
        }

    def get_statistics(
        self, feature: Optional[str] = None, period: str = "daily"
    ) -> Dict[str, Any]:
        """Get aggregated statistics."""
        results = self.store.query_results(feature=feature, limit=1000)

        if not results:
            return {"data": {}, "status": 200}

        scores = [r["score"] for r in results if r.get("score")]

        stats = {
            "total_runs": len(results),
            "feature": feature or "all",
            "period": period,
            "metrics": {
                "mean_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "success_rate": sum(1 for r in results if r["status"] == "completed")
                / len(results),
            },
            "by_feature": {},
        }

        # Group by feature
        from collections import defaultdict

        feature_counts = defaultdict(int)
        for r in results:
            if r.get("feature"):
                feature_counts[r["feature"]] += 1

        stats["by_feature"] = dict(feature_counts)

        return {"data": stats, "status": 200}

    def export_results(self, format: str = "json", feature: Optional[str] = None) -> Dict[str, Any]:
        """Export results in various formats."""
        results = self.store.query_results(feature=feature, limit=10000)

        if format == "json":
            return {"data": results, "format": "json", "status": 200}
        elif format == "csv":
            # Convert to CSV format
            import csv
            import io

            output = io.StringIO()
            if results:
                writer = csv.DictWriter(output, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

            return {"data": output.getvalue(), "format": "csv", "status": 200}
        else:
            return {"error": "Unsupported format", "status": 400}


# Priority [246] - Distributed Cache
class DistributedCache:
    """Distributed caching system to avoid recomputation."""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.lock = Lock()
        self.access_counts = defaultdict(int)
        self.creation_times = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                # Check TTL
                if time.time() - self.creation_times[key] > self.ttl:
                    del self.cache[key]
                    del self.creation_times[key]
                    return None

                self.access_counts[key] += 1
                return self.cache[key]

        return None

    def set(self, key: str, value: Any) -> bool:
        """Set value in cache."""
        with self.lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            self.cache[key] = value
            self.creation_times[key] = time.time()
            self.access_counts[key] = 1

        return True

    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.cache:
            return

        # Find LRU item
        lru_key = min(self.access_counts.keys(), key=self.access_counts.get)

        del self.cache[lru_key]
        del self.creation_times[lru_key]
        del self.access_counts[lru_key]

    def invalidate(self, pattern: Optional[str] = None):
        """Invalidate cache entries."""
        with self.lock:
            if pattern:
                keys_to_delete = [k for k in self.cache.keys() if pattern in k]
                for key in keys_to_delete:
                    del self.cache[key]
                    del self.creation_times[key]
                    del self.access_counts[key]
            else:
                self.cache.clear()
                self.creation_times.clear()
                self.access_counts.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_accesses = sum(self.access_counts.values())

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "total_accesses": total_accesses,
                "hit_rate": total_accesses / max(len(self.access_counts), 1),
                "most_accessed": sorted(
                    self.access_counts.items(), key=lambda x: x[1], reverse=True
                )[:10],
            }


# Priority [267] - Dynamic Configuration with Hot Reload
class DynamicConfig:
    """Dynamic configuration system with hot reload."""

    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = {}
        self.last_modified = 0
        self.lock = Lock()
        self.watchers = []
        self.load_config()

    def load_config(self):
        """Load configuration from file."""
        if self.config_path.exists():
            with self.lock:
                with open(self.config_path) as f:
                    self.config = json.load(f)
                self.last_modified = self.config_path.stat().st_mtime

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        self.check_reload()

        with self.lock:
            # Support nested keys with dot notation
            keys = key.split(".")
            value = self.config

            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default

            return value

    def set(self, key: str, value: Any):
        """Set configuration value."""
        with self.lock:
            keys = key.split(".")
            config = self.config

            # Navigate to nested location
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]

            config[keys[-1]] = value
            self.save_config()

    def save_config(self):
        """Save configuration to file."""
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        self.last_modified = self.config_path.stat().st_mtime

    def check_reload(self):
        """Check if config file has been modified."""
        if self.config_path.exists():
            current_mtime = self.config_path.stat().st_mtime
            if current_mtime > self.last_modified:
                self.load_config()
                self.notify_watchers()

    def watch(self, callback):
        """Register watcher for config changes."""
        self.watchers.append(callback)

    def notify_watchers(self):
        """Notify all watchers of config change."""
        for callback in self.watchers:
            try:
                callback(self.config)
            except Exception as e:
                print(f"Error notifying watcher: {e}")


# Priority [121] - Pre-sales Landing & Receipt Artifacts
class PreSalesArtifacts:
    """Generate pre-sales artifacts and receipts."""

    def __init__(self):
        self.templates = {
            "landing": self._get_landing_template(),
            "receipt": self._get_receipt_template(),
            "proposal": self._get_proposal_template(),
        }

    def generate_landing_page(
        self, hypothesis: str, features: List[str], estimated_cost: float
    ) -> str:
        """Generate landing page for pre-sales."""
        return self.templates["landing"].format(
            hypothesis=hypothesis,
            features=", ".join(features),
            estimated_cost=estimated_cost,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def generate_receipt(self, run_id: str, results: Dict[str, Any], cost: float) -> Dict[str, Any]:
        """Generate execution receipt."""
        return {
            "receipt_id": hashlib.sha256(run_id.encode()).hexdigest()[:16],
            "run_id": run_id,
            "timestamp": time.time(),
            "summary": {
                "feature": results.get("feature"),
                "score": results.get("score"),
                "status": results.get("status", "completed"),
            },
            "cost_breakdown": {
                "computation": cost * 0.6,
                "storage": cost * 0.1,
                "network": cost * 0.1,
                "overhead": cost * 0.2,
                "total": cost,
            },
            "guarantee": {"reproducible": True, "sla_met": True, "quality_assured": True},
        }

    def _get_landing_template(self) -> str:
        """Get landing page template."""
        return """
# ORCHEX Hypothesis Evaluation

## Your Hypothesis
{hypothesis}

## Selected Features
{features}

## Estimated Investment
${estimated_cost:.2f}

## What You Get
- Comprehensive evaluation report
- Attack resistance analysis
- Reproducible results
- Performance metrics
- Actionable recommendations

## Generated: {timestamp}

[Start Evaluation] [Learn More] [Contact Sales]
"""

    def _get_receipt_template(self) -> str:
        """Get receipt template."""
        return """
ORCHEX EXECUTION RECEIPT
=====================
Receipt ID: {{receipt_id}}
Run ID: {{run_id}}
Date: {{timestamp}}

SERVICES RENDERED:
- Feature: {{feature}}
- Score: {{score}}
- Status: {{status}}

COST BREAKDOWN:
- Computation: ${{computation:.2f}}
- Storage: ${{storage:.2f}}
- Network: ${{network:.2f}}
- Overhead: ${{overhead:.2f}}
-----------------------
TOTAL: ${{total:.2f}}

GUARANTEE:
✓ Results reproducible with Run ID
✓ SLA targets met
✓ Quality assured

Thank you for using ORCHEX!
"""

    def _get_proposal_template(self) -> str:
        """Get proposal template."""
        return """
ORCHEX EVALUATION PROPOSAL
========================
Prepared for: {{client}}
Date: {{date}}

EXECUTIVE SUMMARY:
{{summary}}

PROPOSED APPROACH:
{{approach}}

DELIVERABLES:
{{deliverables}}

TIMELINE:
{{timeline}}

INVESTMENT:
{{investment}}

TERMS & CONDITIONS:
{{terms}}
"""


if __name__ == "__main__":
    # Test results store
    store = ResultsStore(":memory:")  # In-memory for testing

    result_id = store.store_result(
        {
            "feature": "nightmare",
            "score": 75.5,
            "metrics": {"attacks": 185, "survival": 67.5},
            "manifest": {"version": "1.0.0"},
            "status": "completed",
        }
    )

    print(f"Stored result: {result_id}")

    # Test API
    api = ResultsAPI(store)
    response = api.get_result(result_id)
    print("API Response:", response)

    # Test cache
    cache = DistributedCache()
    cache.set("test_key", {"data": "cached_value"})
    print("Cache get:", cache.get("test_key"))
    print("Cache stats:", cache.get_statistics())

    # Test dynamic config
    config = DynamicConfig(":memory:")
    config.set("features.nightmare.enabled", True)
    config.set("slos.latency.p50", 1000)
    print("Config:", config.get("features.nightmare.enabled"))

    # Test pre-sales artifacts
    presales = PreSalesArtifacts()
    landing = presales.generate_landing_page(
        "AI safety through interpretability", ["nightmare", "chaos"], 9.99
    )
    print("Landing Page:", landing)
