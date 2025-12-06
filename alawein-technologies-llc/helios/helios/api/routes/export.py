"""
Export API Routes - Data export functionality

Provides endpoints for:
- Exporting algorithms as CSV/JSON
- Exporting leaderboard data
- Generating comprehensive reports
"""

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import io
import csv
import json
from datetime import datetime

router = APIRouter()

# ============================================================================
# DATA MODELS
# ============================================================================

class ExportRequest(BaseModel):
    """Export request model."""
    format: str = "csv"  # csv, json, txt
    include_fields: Optional[list] = None


class ExportResponse(BaseModel):
    """Export response metadata."""
    format: str
    records: int
    exported_at: datetime
    filename: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/export/algorithms")
async def export_algorithms(
    format: str = Query("csv", regex="^(csv|json|txt)$"),
    domain: Optional[str] = None,
) -> StreamingResponse:
    """
    Export algorithms to various formats.

    Args:
        format: Export format (csv, json, txt)
        domain: Optional domain filter

    Returns:
        File stream with exported data
    """
    # Sample algorithm data
    algorithms = [
        {"id": "q1", "name": "Grover's Algorithm", "domain": "quantum", "quality": 85, "speedup": 2.0},
        {"id": "q2", "name": "VQE Hybrid", "domain": "quantum", "quality": 88, "speedup": 2.3},
        {"id": "ml1", "name": "Transformer Arch", "domain": "ml", "quality": 91, "speedup": 2.5},
        {"id": "o3", "name": "Hybrid Metaheuristic", "domain": "optimization", "quality": 88, "speedup": 2.4},
    ]

    # Filter by domain if specified
    if domain:
        algorithms = [a for a in algorithms if a["domain"].lower() == domain.lower()]

    if format == "csv":
        output = io.StringIO()
        if algorithms:
            writer = csv.DictWriter(output, fieldnames=algorithms[0].keys())
            writer.writeheader()
            writer.writerows(algorithms)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=algorithms.csv"}
        )

    elif format == "json":
        output = json.dumps(algorithms, indent=2)
        return StreamingResponse(
            iter([output]),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=algorithms.json"}
        )

    elif format == "txt":
        lines = ["# HELIOS Algorithms Export\n"]
        for algo in algorithms:
            lines.append(f"ID: {algo['id']}, Name: {algo['name']}, Domain: {algo['domain']}")
            lines.append(f"  Quality: {algo['quality']}, Speedup: {algo['speedup']}\n")

        return StreamingResponse(
            iter(lines),
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=algorithms.txt"}
        )

    raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")


@router.post("/export/leaderboard")
async def export_leaderboard(
    format: str = Query("csv", regex="^(csv|json|txt)$"),
    domain: Optional[str] = None,
) -> StreamingResponse:
    """
    Export leaderboard data.

    Args:
        format: Export format (csv, json, txt)
        domain: Optional domain filter

    Returns:
        File stream with leaderboard data
    """
    # Sample leaderboard data
    leaderboard = [
        {"rank": 1, "name": "Transformer Arch", "domain": "ml", "quality": 91, "speedup": 2.5, "tests": 28},
        {"rank": 2, "name": "Hybrid Metaheuristic", "domain": "optimization", "quality": 88, "speedup": 2.4, "tests": 25},
        {"rank": 3, "name": "VQE Hybrid", "domain": "quantum", "quality": 88, "speedup": 2.3, "tests": 15},
    ]

    # Filter by domain if specified
    if domain:
        leaderboard = [e for e in leaderboard if e["domain"].lower() == domain.lower()]

    if format == "csv":
        output = io.StringIO()
        if leaderboard:
            writer = csv.DictWriter(output, fieldnames=leaderboard[0].keys())
            writer.writeheader()
            writer.writerows(leaderboard)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=leaderboard.csv"}
        )

    elif format == "json":
        output = json.dumps(leaderboard, indent=2)
        return StreamingResponse(
            iter([output]),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=leaderboard.json"}
        )

    elif format == "txt":
        lines = ["# HELIOS Leaderboard Export\n\n"]
        for entry in leaderboard:
            lines.append(f"#{entry['rank']} - {entry['name']} ({entry['domain']})")
            lines.append(f"  Quality: {entry['quality']}, Speedup: {entry['speedup']}x, Tests: {entry['tests']}\n")

        return StreamingResponse(
            iter(lines),
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=leaderboard.txt"}
        )

    raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")


@router.post("/export/report")
async def export_report(
    format: str = Query("json", regex="^(json|html)$"),
    domain: Optional[str] = None,
) -> StreamingResponse:
    """
    Generate and export comprehensive report.

    Args:
        format: Export format (json, html)
        domain: Optional domain filter

    Returns:
        Report file stream
    """
    report = {
        "title": "HELIOS System Report",
        "generated_at": datetime.utcnow().isoformat(),
        "domain": domain or "All Domains",
        "summary": {
            "total_algorithms": 64,
            "total_domains": 7,
            "average_quality": 0.84,
            "total_tests_run": 285,
        },
        "top_algorithms": [
            {"name": "Transformer Arch", "quality": 0.91, "speedup": 2.5},
            {"name": "Hybrid Metaheuristic", "quality": 0.88, "speedup": 2.4},
            {"name": "VQE Hybrid", "quality": 0.88, "speedup": 2.3},
        ],
        "domain_statistics": {
            "quantum": {"algorithms": 3, "avg_quality": 0.85},
            "materials": {"algorithms": 3, "avg_quality": 0.84},
            "optimization": {"algorithms": 4, "avg_quality": 0.81},
            "ml": {"algorithms": 3, "avg_quality": 0.90},
            "nas": {"algorithms": 2, "avg_quality": 0.86},
            "synthesis": {"algorithms": 2, "avg_quality": 0.84},
            "graph": {"algorithms": 2, "avg_quality": 0.84},
        }
    }

    if format == "json":
        output = json.dumps(report, indent=2)
        return StreamingResponse(
            iter([output]),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=report.json"}
        )

    elif format == "html":
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>HELIOS Report</title>
    <style>
        body {{ font-family: Arial; margin: 40px; background: #f5f5f5; }}
        .container {{ background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #2563eb; }}
        .stat {{ display: inline-block; margin: 10px 20px 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #2563eb; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{report['title']}</h1>
        <p>Generated: {report['generated_at']}</p>

        <h2>Summary</h2>
        <div class="stat"><strong>Algorithms:</strong> {report['summary']['total_algorithms']}</div>
        <div class="stat"><strong>Domains:</strong> {report['summary']['total_domains']}</div>
        <div class="stat"><strong>Avg Quality:</strong> {report['summary']['average_quality']:.2f}</div>

        <h2>Top Algorithms</h2>
        <table>
            <tr><th>Name</th><th>Quality</th><th>Speedup</th></tr>
"""
        for algo in report['top_algorithms']:
            html += f"<tr><td>{algo['name']}</td><td>{algo['quality']:.2f}</td><td>{algo['speedup']:.1f}x</td></tr>"

        html += """
        </table>
    </div>
</body>
</html>
"""
        return StreamingResponse(
            iter([html]),
            media_type="text/html",
            headers={"Content-Disposition": "attachment; filename=report.html"}
        )

    raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")


@router.get("/export/status")
async def export_status():
    """
    Get export service status.

    Returns:
        Status information
    """
    return {
        "status": "operational",
        "supported_formats": ["csv", "json", "txt", "html"],
        "max_records": 100000,
        "compression": "gzip",
        "timestamp": datetime.utcnow().isoformat(),
    }
