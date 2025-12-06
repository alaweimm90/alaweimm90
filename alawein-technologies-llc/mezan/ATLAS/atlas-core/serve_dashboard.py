#!/usr/bin/env python3
"""
Serve the MEZAN metrics dashboard locally
"""

import os
import json
import time
import random
from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread

class DashboardHandler(SimpleHTTPRequestHandler):
    """Custom handler for serving the dashboard and API endpoints"""

    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            # Serve the dashboard
            self.path = '/dashboard.html'
        elif self.path == '/api/metrics':
            # Serve real-time metrics
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            metrics = {
                'activeJobs': random.randint(20, 70),
                'throughput': round(random.uniform(0.5, 2.5), 1),
                'successRate': round(random.uniform(95, 100), 1),
                'avgResponse': random.randint(100, 200),
                'cpu': random.randint(50, 90),
                'memory': random.randint(40, 70),
                'queue': random.randint(0, 50),
                'timestamp': int(time.time())
            }

            self.wfile.write(json.dumps(metrics).encode())
            return

        elif self.path == '/api/jobs':
            # Serve job data
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            jobs = []
            for i in range(8):
                jobs.append({
                    'id': f'JOB-{random.randint(1000, 9999)}',
                    'type': random.choice(['Optimization', 'Analysis', 'Synthesis', 'Evaluation']),
                    'status': random.choice(['Running', 'Queued', 'Processing', 'Completed']),
                    'progress': random.randint(0, 100),
                    'duration': f'{random.randint(1, 300)}s',
                    'agent': random.choice(['ORCHEX', 'Libria', 'Synthesis', 'Hypothesis'])
                })

            self.wfile.write(json.dumps(jobs).encode())
            return

        # Default file serving
        super().do_GET()

def start_server(port=8080):
    """Start the dashboard server"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = HTTPServer(('localhost', port), DashboardHandler)
    print(f"""
╔══════════════════════════════════════════════════╗
║     MEZAN Metrics Dashboard Server               ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  Dashboard:  http://localhost:{port}/            ║
║  API Metrics: http://localhost:{port}/api/metrics ║
║  API Jobs:    http://localhost:{port}/api/jobs    ║
║                                                  ║
║  Press Ctrl+C to stop the server                ║
╚══════════════════════════════════════════════════╝
    """)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n✓ Server stopped")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Serve MEZAN Metrics Dashboard')
    parser.add_argument('-p', '--port', type=int, default=8080,
                       help='Port to serve on (default: 8080)')

    args = parser.parse_args()
    start_server(args.port)