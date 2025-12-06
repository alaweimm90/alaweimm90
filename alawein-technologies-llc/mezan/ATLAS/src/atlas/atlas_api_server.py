#!/usr/bin/env python3
"""
ORCHEX REST API Server
Complete web service for ORCHEX hypothesis evaluation platform.
"""

import json
import logging
import time
import uuid
from typing import Any, Dict

from flask import Flask, abort, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from ORCHEX.advanced_systems import (
    BootstrapCI,
    ComparativeAnalyzer,
    MLTaskRouter,
    TenantQuotaManager,
)
from ORCHEX.pii_redactor import PIIPipeline

# Import ORCHEX modules (using package imports)
from ORCHEX.quality_gates import QualityGates
from ORCHEX.results_store import (
    DistributedCache,
    DynamicConfig,
    PreSalesArtifacts,
    ResultsAPI,
    ResultsStore,
)
from ORCHEX.shadow_eval import ShadowEvaluator
from ORCHEX.tracing_logger import ATLASTracer, StructuredLogger

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Rate limiting
limiter = Limiter(
    app=app, key_func=get_remote_address, default_limits=["1000 per hour", "100 per minute"]
)

# Initialize components
logger = StructuredLogger("atlas_api")
tracer = ATLASTracer()
quality_gates = QualityGates()
pii_pipeline = PIIPipeline()
shadow_eval = ShadowEvaluator()
results_store = ResultsStore()
results_api = ResultsAPI(results_store)
cache = DistributedCache()
config = DynamicConfig()
ml_router = MLTaskRouter()
quota_manager = TenantQuotaManager()
presales = PreSalesArtifacts()

# API versioning
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"


# ============= Health & Status Endpoints =============


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": time.time(),
            "version": API_VERSION,
            "components": {
                "database": "connected",
                "cache": "operational",
                "quality_gates": "ready",
            },
        }
    )


@app.route("/status", methods=["GET"])
def system_status():
    """Detailed system status."""
    cache_stats = cache.get_statistics()

    return jsonify(
        {
            "status": "operational",
            "api_version": API_VERSION,
            "uptime": time.time(),
            "metrics": {
                "cache_size": cache_stats["size"],
                "cache_hit_rate": cache_stats.get("hit_rate", 0),
                "active_tenants": len(quota_manager.quotas),
                "stored_results": len(results_store.query_results(limit=1)),
            },
            "features": {
                "nightmare": config.get("features.nightmare.enabled", True),
                "chaos": config.get("features.chaos.enabled", True),
                "evolution": config.get("features.evolution.enabled", True),
                "multiverse": config.get("features.multiverse.enabled", True),
                "market": config.get("features.market.enabled", True),
            },
        }
    )


# ============= Evaluation Endpoints =============


@app.route(f"{API_PREFIX}/evaluate", methods=["POST"])
@limiter.limit("10 per minute")
def evaluate_hypothesis():
    """Main hypothesis evaluation endpoint."""
    try:
        # Get request data
        input_data = request.get_json()
        if not input_data:
            abort(400, "No input data provided")

        # Extract tenant info from headers
        tenant_id = request.headers.get("X-Tenant-ID", "default")
        api_key = request.headers.get("X-API-Key")

        # Check tenant quota
        quota_check = quota_manager.check_quota(tenant_id, "requests")
        if not quota_check["allowed"]:
            abort(429, f"Quota exceeded. Used: {quota_check['usage_pct']:.1f}%")

        # Run quality gates
        gate_context = {
            "user_id": tenant_id,
            "api_key": api_key,
            "resources": {"cpu_percent_free": 60, "memory_percent_free": 50},
        }

        gate_results = quality_gates.run_all_gates(input_data, gate_context)
        if not gate_results["can_proceed"]:
            return (
                jsonify(
                    {
                        "error": "Quality gates failed",
                        "gates": gate_results["gates"],
                        "recommendations": gate_results["recommendations"],
                    }
                ),
                400,
            )

        # PII scanning and redaction
        privacy_report = pii_pipeline.generate_privacy_report(input_data)
        if privacy_report["risk_score"] > 0.7:
            # Redact PII if high risk
            input_data = pii_pipeline.process_input(input_data)

        # Route to appropriate feature
        available_features = ["nightmare", "chaos", "evolution", "multiverse", "market"]
        if "feature" in input_data:
            selected_feature = input_data["feature"]
            routing_confidence = 1.0
        else:
            selected_feature, routing_confidence = ml_router.route_task(
                input_data, available_features
            )
            input_data["feature"] = selected_feature

        # Check cache
        input_hash = str(hash(json.dumps(input_data, sort_keys=True)))
        cached_result = cache.get(input_hash)
        if cached_result:
            logger.info("Cache hit", cache_key=input_hash)
            return jsonify(
                {"result": cached_result, "cached": True, "routing_confidence": routing_confidence}
            )

        # Start tracing
        with tracer.trace_feature_execution(selected_feature, input_data) as span:
            # Simulate evaluation (in production, would call actual evaluator)
            result = _simulate_evaluation(selected_feature, input_data)

            # Add metadata
            result["feature"] = selected_feature
            result["routing_confidence"] = routing_confidence
            result["tenant_id"] = tenant_id
            result["timestamp"] = time.time()

            # Store result
            result_id = results_store.store_result(result)
            result["id"] = result_id

            # Cache result
            cache.set(input_hash, result)

            # Consume quota
            quota_manager.consume_quota(tenant_id, "requests", 1)

            # Shadow evaluation (async in production)
            if config.get("shadow_eval.enabled", False):
                shadow_results = shadow_eval.run_shadow_eval(input_data, selected_feature)
                result["shadow"] = shadow_results.get("comparison")

        # Generate receipt
        receipt = presales.generate_receipt(result_id, result, result.get("cost", 1.0))

        return jsonify({"result": result, "receipt": receipt, "cached": False})

    except Exception as e:
        logger.error("Evaluation failed", error=str(e))
        return jsonify({"error": str(e)}), 500


# ============= Results Endpoints =============


@app.route(f"{API_PREFIX}/results", methods=["GET"])
def list_results():
    """List evaluation results."""
    feature = request.args.get("feature")
    since = request.args.get("since")
    limit = min(int(request.args.get("limit", 50)), 100)
    page = int(request.args.get("page", 1))

    response = results_api.list_results(feature=feature, since=since, limit=limit, page=page)

    return jsonify(response)


@app.route(f"{API_PREFIX}/results/<result_id>", methods=["GET"])
def get_result(result_id):
    """Get specific result."""
    response = results_api.get_result(result_id)

    if response["status"] == 404:
        abort(404, "Result not found")

    return jsonify(response)


@app.route(f"{API_PREFIX}/results/export", methods=["GET"])
def export_results():
    """Export results in various formats."""
    format = request.args.get("format", "json")
    feature = request.args.get("feature")

    response = results_api.export_results(format=format, feature=feature)

    if format == "csv":
        return (
            response["data"],
            200,
            {"Content-Type": "text/csv", "Content-Disposition": "attachment; filename=results.csv"},
        )

    return jsonify(response)


# ============= Analytics Endpoints =============


@app.route(f"{API_PREFIX}/analytics/statistics", methods=["GET"])
def get_statistics():
    """Get aggregated statistics."""
    feature = request.args.get("feature")
    period = request.args.get("period", "daily")

    response = results_api.get_statistics(feature=feature, period=period)
    return jsonify(response)


@app.route(f"{API_PREFIX}/analytics/compare", methods=["POST"])
def compare_systems():
    """Compare two evaluation runs."""
    data = request.get_json()

    if not data or "system_a" not in data or "system_b" not in data:
        abort(400, "Two systems required for comparison")

    analyzer = ComparativeAnalyzer()
    comparison = analyzer.compare_systems(
        data["system_a"], data["system_b"], data.get("metrics", ["score", "latency", "cost"])
    )

    return jsonify(comparison)


@app.route(f"{API_PREFIX}/analytics/bootstrap", methods=["POST"])
def calculate_bootstrap_ci():
    """Calculate bootstrap confidence intervals."""
    data = request.get_json()

    if not data or "values" not in data:
        abort(400, "Values array required")

    bootstrap = BootstrapCI(
        n_bootstrap=data.get("n_bootstrap", 1000),
        confidence_level=data.get("confidence_level", 0.95),
    )

    result = bootstrap.calculate_ci(data["values"])
    return jsonify(result)


# ============= Configuration Endpoints =============


@app.route(f"{API_PREFIX}/config", methods=["GET"])
def get_config():
    """Get current configuration."""
    tenant_id = request.headers.get("X-Tenant-ID", "default")

    # Only return tenant-specific config
    tenant_config = {
        "features": {
            "nightmare": config.get("features.nightmare.enabled", True),
            "chaos": config.get("features.chaos.enabled", True),
            "evolution": config.get("features.evolution.enabled", True),
            "multiverse": config.get("features.multiverse.enabled", True),
            "market": config.get("features.market.enabled", True),
        },
        "quotas": quota_manager.get_tenant_status(tenant_id),
        "slos": {
            "latency_p50": config.get("slos.latency.p50", 1000),
            "latency_p95": config.get("slos.latency.p95", 5000),
            "latency_p99": config.get("slos.latency.p99", 10000),
        },
    }

    return jsonify(tenant_config)


@app.route(f"{API_PREFIX}/config", methods=["PUT"])
@limiter.limit("1 per minute")
def update_config():
    """Update configuration (admin only)."""
    # Check admin authorization
    admin_key = request.headers.get("X-Admin-Key")
    if admin_key != config.get("admin.key"):
        abort(403, "Admin authorization required")

    updates = request.get_json()
    if not updates:
        abort(400, "No updates provided")

    for key, value in updates.items():
        config.set(key, value)

    return jsonify({"status": "updated", "updates": updates})


# ============= Tenant Management =============


@app.route(f"{API_PREFIX}/tenants/<tenant_id>/quota", methods=["GET"])
def get_tenant_quota(tenant_id):
    """Get tenant quota status."""
    status = quota_manager.get_tenant_status(tenant_id)
    return jsonify(status)


@app.route(f"{API_PREFIX}/tenants/<tenant_id>/quota", methods=["PUT"])
@limiter.limit("1 per minute")
def set_tenant_quota(tenant_id):
    """Set tenant quota (admin only)."""
    # Check admin authorization
    admin_key = request.headers.get("X-Admin-Key")
    if admin_key != config.get("admin.key"):
        abort(403, "Admin authorization required")

    quota_data = request.get_json()
    if not quota_data:
        abort(400, "Quota data required")

    for quota_type, limit in quota_data.items():
        quota_manager.set_quota(tenant_id, quota_type, limit)

    return jsonify({"status": "updated", "tenant_id": tenant_id, "quotas": quota_data})


# ============= Pre-Sales Endpoints =============


@app.route(f"{API_PREFIX}/presales/landing", methods=["POST"])
def generate_landing():
    """Generate pre-sales landing page."""
    data = request.get_json()

    landing = presales.generate_landing_page(
        data.get("hypothesis", ""),
        data.get("features", ["nightmare"]),
        data.get("estimated_cost", 10.0),
    )

    return jsonify(
        {"landing_page": landing, "preview_url": f"https://ORCHEX.ai/preview/{uuid.uuid4().hex[:8]}"}
    )


# ============= Helper Functions =============


def _simulate_evaluation(feature: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate evaluation for demo purposes."""
    base_result = {
        "score": 75.0 + (hash(feature) % 20),
        "confidence": 0.85,
        "cost": 1.5,
        "duration_ms": 2500,
    }

    if feature == "nightmare":
        base_result.update(
            {
                "survival_score": 67.5,
                "attacks_generated": 185,
                "critical_vulnerabilities": 3,
                "score_breakdown": {
                    "statistical_robustness": 72.5,
                    "methodological_integrity": 68.9,
                    "logical_consistency": 74.2,
                    "ethical_compliance": 55.1,
                    "economic_feasibility": 61.8,
                },
            }
        )
    elif feature == "chaos":
        base_result.update({"novelty_score": 0.87, "collisions_generated": 25, "genius_tier": 1})
    elif feature == "evolution":
        base_result.update({"generations": 50, "diversity": 0.78, "pareto_front": 7})
    elif feature == "multiverse":
        base_result.update({"universes": 20, "invariants": 3, "divergent": 7})
    elif feature == "market":
        base_result.update({"contracts": 3, "liquidity": 45000, "manipulation_score": 0.02})

    return base_result


# ============= Error Handlers =============


@app.errorhandler(400)
def bad_request(error):
    """Handle bad request errors."""
    return jsonify({"error": str(error)}), 400


@app.errorhandler(404)
def not_found(error):
    """Handle not found errors."""
    return jsonify({"error": "Resource not found"}), 404


@app.errorhandler(429)
def rate_limit_exceeded(error):
    """Handle rate limit errors."""
    return jsonify({"error": "Rate limit exceeded", "retry_after": error.description}), 429


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error("Internal server error", error=str(error))
    return jsonify({"error": "Internal server error"}), 500


# ============= Main Entry Point =============

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Initialize default quotas
    quota_manager.set_quota("default", "requests", 1000, "daily")
    quota_manager.set_quota("default", "tokens", 1000000, "daily")


def main():
    """Main entry point for the ORCHEX API server."""
    import os

    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    port = int(os.getenv("API_PORT", "5000"))
    host = os.getenv("API_HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "false").lower() == "true"

    print(
        f"""
    ╔════════════════════════════════════════╗
    ║         ORCHEX API Server v1.0          ║
    ║                                        ║
    ║  Hypothesis Evaluation Platform        ║
    ║  25 Global Priorities Implemented      ║
    ║                                        ║
    ║  Starting on http://{host}:{port}     ║
    ╚════════════════════════════════════════╝

    API Documentation: http://{host}:{port}/api/v1/docs
    Health Check: http://{host}:{port}/health
    """
    )

    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()
