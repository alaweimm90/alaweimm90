#!/usr/bin/env python3
"""
REPZ Autonomous Improvement Workflow Runner

Main execution script for the 500-step autonomous workflow system.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.settings import get_settings
from src.workflow.engine import WorkflowEngine
from src.monitoring.metrics import MetricsCollector
from src.ai.decision_engine import AIDecisionEngine
from src.database.connection import init_database, close_database
from src.reporting.pdf_generator import PDFReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('repz_workflow.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """Main workflow execution function"""
    logger.info("🚀 Starting REPZ Autonomous Improvement Workflow")

    settings = get_settings()

    try:
        # Initialize database
        logger.info("📊 Initializing database...")
        await init_database(settings.database_url)

        # Initialize components
        logger.info("🤖 Initializing AI and monitoring systems...")
        metrics_collector = MetricsCollector()
        ai_engine = AIDecisionEngine()
        workflow_engine = WorkflowEngine(
            ai_engine=ai_engine,
            metrics_collector=metrics_collector
        )

        # Start metrics collection
        await metrics_collector.start_collection()

        # Execute workflow
        logger.info("⚡ Starting 500-step workflow execution...")

        # Wait for completion (in real implementation, this would be event-driven)
        while True:
            status = await workflow_engine.get_status()
            progress = status.get('progress_percentage', 0)
            current_step = status.get('current_step', 0)
            logger.info(f"Progress: {progress:.1f}% (Step {current_step}/500)")
            if status.get('status') == 'completed':
                break
            elif status.get('status') == 'error':
                logger.error("Workflow execution failed")
                break

            await asyncio.sleep(10)  # Check every 10 seconds

        # Generate final reports
        logger.info("📄 Generating final reports...")

        # Get final data
        final_workflow_status = await workflow_engine.get_status()
        final_metrics = await metrics_collector.get_all_metrics()
        final_ai_insights = await ai_engine.get_current_insights()

        # Generate PDF report
        pdf_generator = PDFReportGenerator()
        pdf_path = await pdf_generator.generate_comprehensive_report(
            workflow_data=final_workflow_status,
            metrics_data=final_metrics,
            ai_insights=final_ai_insights,
            output_path="outputs/repz_workflow_report.pdf"
        )

        logger.info(f"✅ PDF report generated: {pdf_path}")

        # Generate additional outputs (Excel, Colab, etc.)
        await generate_additional_outputs(final_workflow_status, final_metrics, final_ai_insights)

        logger.info("🎉 REPZ Autonomous Improvement Workflow completed successfully!")

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise
    finally:
        # Cleanup
        await close_database()
        logger.info("🧹 Cleanup completed")


async def generate_additional_outputs(workflow_data, metrics_data, ai_insights):
    """Generate additional output formats"""
    logger.info("📊 Generating additional outputs...")

    # Create outputs directory
    Path("outputs").mkdir(exist_ok=True)

    # Generate Excel spreadsheet
    await generate_excel_report(workflow_data, metrics_data, "outputs/repz_metrics.xlsx")

    # Generate Colab notebook
    await generate_colab_notebook(workflow_data, metrics_data, ai_insights, "outputs/repz_analysis.ipynb")

    # Generate intake forms
    await generate_intake_forms("outputs/stakeholder_intake_forms/")

    logger.info("📊 Additional outputs generated")


async def generate_excel_report(workflow_data, metrics_data, output_path):
    """Generate Excel spreadsheet with analysis data"""
    try:
        import pandas as pd

        # Create workbook with multiple sheets
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

            # Workflow summary
            workflow_df = pd.DataFrame([workflow_data])
            workflow_df.to_excel(writer, sheet_name='Workflow_Summary', index=False)

            # System metrics
            system_metrics = metrics_data.get('system', {})
            if system_metrics:
                system_df = pd.DataFrame([system_metrics])
                system_df.to_excel(writer, sheet_name='System_Metrics', index=False)

            # Step metrics
            step_metrics = metrics_data.get('steps', {})
            if step_metrics:
                steps_df = pd.DataFrame.from_dict(step_metrics, orient='index')
                steps_df.to_excel(writer, sheet_name='Step_Metrics', index=True)

        logger.info(f"Excel report generated: {output_path}")

    except Exception as e:
        logger.error(f"Excel report generation failed: {e}")


async def generate_colab_notebook(workflow_data, metrics_data, ai_insights, output_path):
    """Generate Google Colab notebook for interactive analysis"""
    try:
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# REPZ Autonomous Improvement Workflow Analysis\n",
                        "\n",
                        "This notebook provides interactive analysis of the REPZ workflow execution results."
                    ]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "import pandas as pd\n",
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sns\n",
                        "import json\n",
                        "\n",
                        "# Load workflow data\n",
                        "workflow_data = json.loads('''{}''')\n".format(json.dumps(workflow_data)),
                        "print('Workflow completed with', workflow_data.get('progress_percentage', 0), '% progress')"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }

        import json
        with open(output_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)

        logger.info(f"Colab notebook generated: {output_path}")

    except Exception as e:
        logger.error(f"Colab notebook generation failed: {e}")


async def generate_intake_forms(output_dir):
    """Generate stakeholder intake forms"""
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generate different types of intake forms
        forms = {
            "coach_intake.json": {
                "form_type": "coach_onboarding",
                "fields": [
                    {"name": "company_name", "type": "text", "required": True},
                    {"name": "team_size", "type": "number", "required": True},
                    {"name": "current_tools", "type": "textarea", "required": False},
                    {"name": "goals", "type": "textarea", "required": True}
                ]
            },
            "athlete_intake.json": {
                "form_type": "athlete_onboarding",
                "fields": [
                    {"name": "sport", "type": "select", "options": ["powerlifting", "crossfit", "running", "other"], "required": True},
                    {"name": "experience_level", "type": "select", "options": ["beginner", "intermediate", "advanced"], "required": True},
                    {"name": "goals", "type": "textarea", "required": True},
                    {"name": "current_training", "type": "textarea", "required": False}
                ]
            },
            "technical_intake.json": {
                "form_type": "technical_integration",
                "fields": [
                    {"name": "integration_type", "type": "select", "options": ["api", "webhook", "sdk", "custom"], "required": True},
                    {"name": "current_systems", "type": "textarea", "required": True},
                    {"name": "data_requirements", "type": "textarea", "required": True},
                    {"name": "security_requirements", "type": "textarea", "required": False}
                ]
            }
        }

        import json
        for filename, form_data in forms.items():
            with open(Path(output_dir) / filename, 'w') as f:
                json.dump(form_data, f, indent=2)

        logger.info(f"Intake forms generated in: {output_dir}")

    except Exception as e:
        logger.error(f"Intake forms generation failed: {e}")


if __name__ == "__main__":
    # Run the workflow
    asyncio.run(main())