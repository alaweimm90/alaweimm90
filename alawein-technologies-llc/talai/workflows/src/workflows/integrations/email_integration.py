"""
Email Integration - Email notification system

Supports sending workflow notifications and results via email.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import logging
import json


logger = logging.getLogger(__name__)


@dataclass
class EmailConfig:
    """Email integration configuration"""
    smtp_host: str
    smtp_port: int
    username: str
    password: str
    from_address: str
    default_recipients: List[str]
    use_tls: bool = True
    use_ssl: bool = False


class EmailIntegration:
    """Email notification integration"""

    def __init__(self, config: Optional[EmailConfig] = None):
        self.config = config
        self.smtp_client = None

    async def initialize(self):
        """Initialize email client"""
        if self.config:
            self.smtp_client = aiosmtplib.SMTP(
                hostname=self.config.smtp_host,
                port=self.config.smtp_port,
                use_tls=self.config.use_tls
            )

    async def cleanup(self):
        """Cleanup resources"""
        if self.smtp_client:
            self.smtp_client.close()
            self.smtp_client = None

    async def configure(self, config: EmailConfig):
        """Update configuration"""
        self.config = config
        await self.cleanup()
        await self.initialize()

    async def test_connection(self) -> Dict[str, Any]:
        """Test email connection"""
        try:
            await self.smtp_client.connect()
            await self.smtp_client.login(self.config.username, self.config.password)
            await self.smtp_client.quit()
            return {"success": True, "server": self.config.smtp_host}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def send_email(self, subject: str, body: str,
                        recipients: Optional[List[str]] = None,
                        attachments: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Send email notification"""
        recipients = recipients or self.config.default_recipients

        # Create message
        msg = MIMEMultipart()
        msg['From'] = self.config.from_address
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = subject

        # Add body
        msg.attach(MIMEText(body, 'html' if '<html>' in body else 'plain'))

        # Add attachments
        if attachments:
            for attachment in attachments:
                self._add_attachment(msg, attachment)

        try:
            # Send email
            await self.smtp_client.connect()
            await self.smtp_client.login(self.config.username, self.config.password)
            await self.smtp_client.send_message(msg)
            await self.smtp_client.quit()

            logger.info(f"Sent email to {recipients}: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def _add_attachment(self, msg: MIMEMultipart, attachment: Dict[str, Any]):
        """Add attachment to email"""
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment['content'])
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f"attachment; filename= {attachment['filename']}"
        )
        msg.attach(part)

    async def send_workflow_report(self, workflow_id: str,
                                  results: Dict[str, Any]) -> bool:
        """Send workflow execution report"""
        subject = f"Workflow Report: {workflow_id}"

        # Create HTML report
        body = self._create_html_report(workflow_id, results)

        # Add JSON attachment
        attachments = [{
            "filename": f"workflow_{workflow_id}_results.json",
            "content": json.dumps(results, indent=2).encode()
        }]

        return await self.send_email(subject, body, attachments=attachments)

    def _create_html_report(self, workflow_id: str,
                           results: Dict[str, Any]) -> str:
        """Create HTML report for workflow"""
        status_color = "green" if results.get("status") == "completed" else "red"

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                h1 {{ color: #333; }}
                .status {{ color: {status_color}; font-weight: bold; }}
                .metrics {{ background-color: #f0f0f0; padding: 10px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <h1>Workflow Execution Report</h1>
            <p><strong>Workflow ID:</strong> {workflow_id}</p>
            <p><strong>Status:</strong> <span class="status">{results.get('status', 'Unknown')}</span></p>
            <p><strong>Execution Time:</strong> {results.get('duration_seconds', 0):.2f} seconds</p>

            <h2>Node Results</h2>
            <table>
                <tr>
                    <th>Node</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Error</th>
                </tr>
        """

        for node_id, node_result in results.get('node_results', {}).items():
            status = node_result.get('status', 'unknown')
            duration = node_result.get('duration', 'N/A')
            error = node_result.get('error', '-')

            html += f"""
                <tr>
                    <td>{node_id}</td>
                    <td>{status}</td>
                    <td>{duration}</td>
                    <td>{error}</td>
                </tr>
            """

        html += """
            </table>

            <h2>Outputs</h2>
            <div class="metrics">
                <pre>{}</pre>
            </div>

            <p><em>Generated at {}</em></p>
        </body>
        </html>
        """.format(
            json.dumps(results.get('outputs', {}), indent=2),
            datetime.now().isoformat()
        )

        return html

    async def handle_event(self, event_type: str, data: Dict[str, Any]):
        """Handle email events"""
        # Only send emails for important events
        important_events = {
            "workflow_failed": "Workflow Failed",
            "workflow_completed": "Workflow Completed",
            "critical_error": "Critical Error"
        }

        if event_type in important_events:
            subject = f"[TalAI] {important_events[event_type]}"
            body = self._format_event_body(event_type, data)
            await self.send_email(subject, body)

    def _format_event_body(self, event_type: str,
                          data: Dict[str, Any]) -> str:
        """Format event data for email body"""
        body = f"Event: {event_type}\n\n"
        body += "Details:\n"
        for key, value in data.items():
            body += f"  {key}: {value}\n"
        body += f"\nTimestamp: {datetime.now().isoformat()}"
        return body

    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            "configured": self.config is not None,
            "smtp_host": self.config.smtp_host if self.config else None,
            "from_address": self.config.from_address if self.config else None
        }