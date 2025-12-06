#!/usr/bin/env python3
"""
Report Generation for Librex Benchmarks

Generates HTML dashboards, Markdown reports, and CSV exports from benchmark results.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd


class HTMLReportGenerator:
    """Generate interactive HTML dashboard for benchmark results"""

    def generate(self, results: Dict, output_path: Path):
        """Generate HTML report with charts and visualizations

        Args:
            results: Benchmark results dictionary
            output_path: Path to save HTML file
        """
        suite_name = results.get('suite_name', 'Unknown')
        suite_config = results.get('suite_config', {})
        metadata = results.get('metadata', {})

        # Prepare data for charts
        methods_data = self._prepare_methods_comparison(results)
        problems_data = self._prepare_problems_analysis(results)
        trends_data = self._prepare_trends_data(results)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Librex Benchmark Report - {suite_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}

        h1 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}

        .metadata {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}

        .metadata-item {{
            background: #f7f7f7;
            padding: 10px 15px;
            border-radius: 5px;
        }}

        .metadata-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}

        .metadata-value {{
            font-weight: 600;
            color: #333;
        }}

        .section {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}

        h2 {{
            color: #444;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}

        .chart-container {{
            margin: 20px 0;
            min-height: 400px;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}

        .summary-card h3 {{
            font-size: 2em;
            margin-bottom: 10px;
        }}

        .summary-card p {{
            opacity: 0.9;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}

        td {{
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
        }}

        tr:hover {{
            background: #f5f5f5;
        }}

        .best-value {{
            background: #4caf50;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: 600;
        }}

        .regression-alert {{
            background: #ff9800;
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}

        .regression-critical {{
            background: #f44336;
        }}

        footer {{
            text-align: center;
            color: white;
            padding: 20px;
            margin-top: 50px;
        }}

        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }}

        .tab {{
            padding: 10px 20px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 16px;
            color: #666;
            transition: all 0.3s;
        }}

        .tab.active {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏆 Librex Benchmark Report</h1>
            <p>Suite: <strong>{suite_name}</strong> - {suite_config.get('description', '')}</p>

            <div class="metadata">
                <div class="metadata-item">
                    <div class="metadata-label">Timestamp</div>
                    <div class="metadata-value">{results.get('start_time', 'N/A')}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Duration</div>
                    <div class="metadata-value">{results.get('total_duration', 0):.1f}s</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Git Commit</div>
                    <div class="metadata-value">{metadata.get('git_commit', 'N/A')}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Total Runs</div>
                    <div class="metadata-value">{sum(r.get('runs', 0) for r in results.get('results', []))}</div>
                </div>
            </div>
        </header>

        <section class="section">
            <h2>📊 Performance Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>{len(set(r['method'] for r in results.get('results', [])))}</h3>
                    <p>Methods Tested</p>
                </div>
                <div class="summary-card">
                    <h3>{len(set(self._get_problem_key(r['problem']) for r in results.get('results', [])))}</h3>
                    <p>Problem Instances</p>
                </div>
                <div class="summary-card">
                    <h3>{len([r for r in results.get('results', []) if r.get('status') != 'failed'])}⁄{len(results.get('results', []))}</h3>
                    <p>Successful Runs</p>
                </div>
                <div class="summary-card">
                    <h3>{np.mean([r['statistics']['mean_runtime'] for r in results.get('results', []) if 'statistics' in r]):.2f}s</h3>
                    <p>Avg Runtime</p>
                </div>
            </div>
        </section>

        <section class="section">
            <h2>🎯 Method Comparison</h2>

            <div class="tabs">
                <button class="tab active" onclick="showTab('methods-chart')">Chart View</button>
                <button class="tab" onclick="showTab('methods-table')">Table View</button>
            </div>

            <div id="methods-chart" class="tab-content active">
                <div id="methodsComparisonChart" class="chart-container"></div>
            </div>

            <div id="methods-table" class="tab-content">
                {self._generate_methods_table(results)}
            </div>
        </section>

        <section class="section">
            <h2>📈 Problem Scalability</h2>
            <div id="scalabilityChart" class="chart-container"></div>
        </section>

        <section class="section">
            <h2>🎬 Performance by Problem Instance</h2>
            <div id="problemsChart" class="chart-container"></div>
        </section>

        <section class="section">
            <h2>💾 Memory Usage Analysis</h2>
            <div id="memoryChart" class="chart-container"></div>
        </section>

        {self._generate_regressions_section(results)}

        <section class="section">
            <h2>📋 Detailed Results</h2>
            {self._generate_detailed_table(results)}
        </section>

        <footer>
            <p>Generated by Librex Benchmark System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>

    <script>
        // Method Comparison Chart
        {self._generate_methods_chart_js(methods_data)}

        // Scalability Chart
        {self._generate_scalability_chart_js(results)}

        // Problems Performance Chart
        {self._generate_problems_chart_js(problems_data)}

        // Memory Usage Chart
        {self._generate_memory_chart_js(results)}

        // Tab switching
        function showTab(tabId) {{
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.classList.remove('active');
            }});
            document.getElementById(tabId).classList.add('active');

            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>"""

        # Save HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)

    def _get_problem_key(self, problem_info: Dict) -> str:
        """Generate unique key for a problem"""
        if 'instance' in problem_info:
            return f"{problem_info['type']}_{problem_info['instance']}"
        return f"{problem_info['type']}_size{problem_info.get('size', 0)}"

    def _prepare_methods_comparison(self, results: Dict) -> Dict:
        """Prepare data for methods comparison chart"""
        methods_performance = {}

        for result in results.get('results', []):
            if result.get('status') == 'failed':
                continue

            method = result['method']
            if method not in methods_performance:
                methods_performance[method] = {
                    'objectives': [],
                    'runtimes': [],
                    'memory': []
                }

            stats = result['statistics']
            methods_performance[method]['objectives'].append(stats['best_objective'])
            methods_performance[method]['runtimes'].append(stats['mean_runtime'])
            methods_performance[method]['memory'].append(stats.get('mean_memory', 0))

        return methods_performance

    def _prepare_problems_analysis(self, results: Dict) -> Dict:
        """Prepare data for problems analysis"""
        problems_data = {}

        for result in results.get('results', []):
            if result.get('status') == 'failed':
                continue

            problem_key = self._get_problem_key(result['problem'])
            if problem_key not in problems_data:
                problems_data[problem_key] = {}

            method = result['method']
            problems_data[problem_key][method] = result['statistics']['best_objective']

        return problems_data

    def _prepare_trends_data(self, results: Dict) -> List:
        """Prepare data for trends visualization"""
        # This would integrate with benchmark_history.py for historical data
        return []

    def _generate_methods_chart_js(self, methods_data: Dict) -> str:
        """Generate JavaScript for methods comparison chart"""
        methods = list(methods_data.keys())
        avg_objectives = [np.mean(methods_data[m]['objectives']) for m in methods]
        avg_runtimes = [np.mean(methods_data[m]['runtimes']) for m in methods]

        return f"""
        var methodsData = [
            {{
                x: {methods},
                y: {avg_objectives},
                name: 'Objective Value',
                type: 'bar',
                marker: {{ color: '#667eea' }}
            }}
        ];

        var methodsLayout = {{
            title: 'Average Performance by Method',
            xaxis: {{ title: 'Method' }},
            yaxis: {{ title: 'Objective Value' }},
            showlegend: true
        }};

        Plotly.newPlot('methodsComparisonChart', methodsData, methodsLayout);
        """

    def _generate_scalability_chart_js(self, results: Dict) -> str:
        """Generate JavaScript for scalability analysis chart"""
        # Group by problem size
        size_data = {}
        for result in results.get('results', []):
            if result.get('status') == 'failed':
                continue

            size = result['problem'].get('size', 0)
            method = result['method']

            if size not in size_data:
                size_data[size] = {}
            if method not in size_data[size]:
                size_data[size][method] = []

            size_data[size][method].append(result['statistics']['mean_runtime'])

        # Create traces for each method
        traces = []
        methods = set()
        for size in size_data:
            for method in size_data[size]:
                methods.add(method)

        for method in methods:
            sizes = []
            runtimes = []
            for size in sorted(size_data.keys()):
                if method in size_data[size]:
                    sizes.append(size)
                    runtimes.append(np.mean(size_data[size][method]))

            if sizes:
                traces.append({
                    'x': sizes,
                    'y': runtimes,
                    'name': method,
                    'type': 'scatter',
                    'mode': 'lines+markers'
                })

        return f"""
        var scalabilityData = {json.dumps(traces)};

        var scalabilityLayout = {{
            title: 'Runtime Scalability Analysis',
            xaxis: {{ title: 'Problem Size' }},
            yaxis: {{ title: 'Runtime (seconds)', type: 'log' }},
            showlegend: true
        }};

        Plotly.newPlot('scalabilityChart', scalabilityData, scalabilityLayout);
        """

    def _generate_problems_chart_js(self, problems_data: Dict) -> str:
        """Generate JavaScript for problems performance chart"""
        if not problems_data:
            return "// No problem data available"

        # Create heatmap data
        problems = list(problems_data.keys())
        methods = list(set(method for p in problems_data.values() for method in p.keys()))

        z_data = []
        for problem in problems:
            row = []
            for method in methods:
                value = problems_data[problem].get(method, None)
                row.append(value if value is not None else float('nan'))
            z_data.append(row)

        return f"""
        var problemsData = [{{
            z: {json.dumps(z_data)},
            x: {json.dumps(methods)},
            y: {json.dumps(problems)},
            type: 'heatmap',
            colorscale: 'Viridis'
        }}];

        var problemsLayout = {{
            title: 'Performance Heatmap by Problem and Method',
            xaxis: {{ title: 'Method' }},
            yaxis: {{ title: 'Problem Instance' }}
        }};

        Plotly.newPlot('problemsChart', problemsData, problemsLayout);
        """

    def _generate_memory_chart_js(self, results: Dict) -> str:
        """Generate JavaScript for memory usage chart"""
        memory_data = {}

        for result in results.get('results', []):
            if result.get('status') == 'failed':
                continue

            method = result['method']
            memory = result['statistics'].get('mean_memory', 0)

            if method not in memory_data:
                memory_data[method] = []
            memory_data[method].append(memory)

        methods = list(memory_data.keys())
        memory_values = [memory_data[m] for m in methods]

        return f"""
        var memoryData = [{{
            y: {json.dumps(methods)},
            x: {json.dumps(memory_values)},
            type: 'box',
            orientation: 'h',
            marker: {{ color: '#764ba2' }}
        }}];

        var memoryLayout = {{
            title: 'Memory Usage Distribution by Method',
            xaxis: {{ title: 'Memory (MB)' }},
            yaxis: {{ title: 'Method' }}
        }};

        Plotly.newPlot('memoryChart', memoryData, memoryLayout);
        """

    def _generate_methods_table(self, results: Dict) -> str:
        """Generate HTML table for method comparison"""
        # Aggregate statistics by method
        method_stats = {}

        for result in results.get('results', []):
            if result.get('status') == 'failed':
                continue

            method = result['method']
            if method not in method_stats:
                method_stats[method] = {
                    'objectives': [],
                    'runtimes': [],
                    'memory': [],
                    'success_count': 0,
                    'total_count': 0
                }

            stats = result['statistics']
            method_stats[method]['objectives'].append(stats['best_objective'])
            method_stats[method]['runtimes'].append(stats['mean_runtime'])
            method_stats[method]['memory'].append(stats.get('mean_memory', 0))
            method_stats[method]['success_count'] += 1
            method_stats[method]['total_count'] += 1

        # Find best values for highlighting
        best_objective = min(np.mean(ms['objectives']) for ms in method_stats.values())
        best_runtime = min(np.mean(ms['runtimes']) for ms in method_stats.values())

        html = """
        <table>
            <thead>
                <tr>
                    <th>Method</th>
                    <th>Mean Objective</th>
                    <th>Std Dev</th>
                    <th>Best Found</th>
                    <th>Mean Runtime (s)</th>
                    <th>Memory (MB)</th>
                    <th>Success Rate</th>
                </tr>
            </thead>
            <tbody>
        """

        for method, stats in sorted(method_stats.items(), key=lambda x: np.mean(x[1]['objectives'])):
            mean_obj = np.mean(stats['objectives'])
            std_obj = np.std(stats['objectives'])
            best_obj = np.min(stats['objectives'])
            mean_runtime = np.mean(stats['runtimes'])
            mean_memory = np.mean(stats['memory'])
            success_rate = stats['success_count'] / stats['total_count'] * 100

            # Highlight best values
            obj_class = 'class="best-value"' if abs(mean_obj - best_objective) < 0.001 else ''
            runtime_class = 'class="best-value"' if abs(mean_runtime - best_runtime) < 0.001 else ''

            html += f"""
                <tr>
                    <td><strong>{method}</strong></td>
                    <td {obj_class}>{mean_obj:.2f}</td>
                    <td>{std_obj:.2f}</td>
                    <td>{best_obj:.2f}</td>
                    <td {runtime_class}>{mean_runtime:.2f}</td>
                    <td>{mean_memory:.1f}</td>
                    <td>{success_rate:.0f}%</td>
                </tr>
            """

        html += """
            </tbody>
        </table>
        """

        return html

    def _generate_detailed_table(self, results: Dict) -> str:
        """Generate detailed results table"""
        html = """
        <table>
            <thead>
                <tr>
                    <th>Problem</th>
                    <th>Method</th>
                    <th>Config</th>
                    <th>Best Obj</th>
                    <th>Mean Obj</th>
                    <th>Runtime (s)</th>
                    <th>Memory (MB)</th>
                    <th>Gap (%)</th>
                </tr>
            </thead>
            <tbody>
        """

        for result in results.get('results', []):
            if result.get('status') == 'failed':
                continue

            problem_key = self._get_problem_key(result['problem'])
            stats = result['statistics']

            config_str = json.dumps(result.get('config', {}), indent=0).replace('\n', ' ')
            if len(config_str) > 50:
                config_str = config_str[:50] + '...'

            gap = stats.get('optimality_gap', 'N/A')
            gap_str = f"{gap:.2f}%" if isinstance(gap, (int, float)) else gap

            html += f"""
                <tr>
                    <td>{problem_key}</td>
                    <td>{result['method']}</td>
                    <td style="font-size: 0.9em; color: #666;">{config_str}</td>
                    <td>{stats['best_objective']:.2f}</td>
                    <td>{stats['mean_objective']:.2f}</td>
                    <td>{stats['mean_runtime']:.2f}</td>
                    <td>{stats.get('mean_memory', 0):.1f}</td>
                    <td>{gap_str}</td>
                </tr>
            """

        html += """
            </tbody>
        </table>
        """

        return html

    def _generate_regressions_section(self, results: Dict) -> str:
        """Generate section for regression alerts if any"""
        # This would integrate with benchmark_history.py
        # For now, return empty section
        return ""


class MarkdownReportGenerator:
    """Generate Markdown reports for GitHub"""

    def generate(self, results: Dict, output_path: Path):
        """Generate Markdown report

        Args:
            results: Benchmark results dictionary
            output_path: Path to save Markdown file
        """
        suite_name = results.get('suite_name', 'Unknown')
        suite_config = results.get('suite_config', {})
        metadata = results.get('metadata', {})

        md_content = f"""# Librex Benchmark Report

## Suite: {suite_name}

**Description:** {suite_config.get('description', 'N/A')}
**Timestamp:** {results.get('start_time', 'N/A')}
**Duration:** {results.get('total_duration', 0):.1f} seconds
**Git Commit:** `{metadata.get('git_commit', 'N/A')}`

## Summary Statistics

| Metric | Value |
|--------|-------|
| Methods Tested | {len(set(r['method'] for r in results.get('results', [])))} |
| Problem Instances | {len(set(self._get_problem_key(r['problem']) for r in results.get('results', [])))} |
| Total Runs | {sum(r.get('runs', 0) for r in results.get('results', []))} |
| Success Rate | {len([r for r in results.get('results', []) if r.get('status') != 'failed'])}/{len(results.get('results', []))} |

## Method Performance Comparison

"""

        # Add method comparison table
        md_content += self._generate_method_comparison_table(results)

        # Add problem-specific results
        md_content += "\n## Results by Problem Instance\n\n"
        md_content += self._generate_problem_results(results)

        # Add recommendations
        md_content += "\n## Recommendations\n\n"
        md_content += self._generate_recommendations(results)

        # Save Markdown file
        with open(output_path, 'w') as f:
            f.write(md_content)

    def _get_problem_key(self, problem_info: Dict) -> str:
        """Generate unique key for a problem"""
        if 'instance' in problem_info:
            return f"{problem_info['type']}_{problem_info['instance']}"
        return f"{problem_info['type']}_size{problem_info.get('size', 0)}"

    def _generate_method_comparison_table(self, results: Dict) -> str:
        """Generate Markdown table comparing methods"""
        # Aggregate statistics by method
        method_stats = {}

        for result in results.get('results', []):
            if result.get('status') == 'failed':
                continue

            method = result['method']
            if method not in method_stats:
                method_stats[method] = {
                    'objectives': [],
                    'runtimes': [],
                    'gaps': []
                }

            stats = result['statistics']
            method_stats[method]['objectives'].append(stats['best_objective'])
            method_stats[method]['runtimes'].append(stats['mean_runtime'])
            if 'optimality_gap' in stats:
                method_stats[method]['gaps'].append(stats['optimality_gap'])

        # Sort by mean objective
        sorted_methods = sorted(method_stats.items(), key=lambda x: np.mean(x[1]['objectives']))

        table = "| Method | Mean Objective | Std Dev | Best Found | Avg Runtime (s) | Avg Gap (%) |\n"
        table += "|--------|---------------|---------|------------|-----------------|-------------|\n"

        for method, stats in sorted_methods:
            mean_obj = np.mean(stats['objectives'])
            std_obj = np.std(stats['objectives'])
            best_obj = np.min(stats['objectives'])
            mean_runtime = np.mean(stats['runtimes'])
            mean_gap = np.mean(stats['gaps']) if stats['gaps'] else 'N/A'

            gap_str = f"{mean_gap:.2f}" if isinstance(mean_gap, (int, float)) else mean_gap

            table += f"| **{method}** | {mean_obj:.2f} | {std_obj:.2f} | {best_obj:.2f} | {mean_runtime:.2f} | {gap_str} |\n"

        return table

    def _generate_problem_results(self, results: Dict) -> str:
        """Generate results grouped by problem"""
        problem_results = {}

        for result in results.get('results', []):
            if result.get('status') == 'failed':
                continue

            problem_key = self._get_problem_key(result['problem'])
            if problem_key not in problem_results:
                problem_results[problem_key] = []

            problem_results[problem_key].append({
                'method': result['method'],
                'objective': result['statistics']['best_objective'],
                'runtime': result['statistics']['mean_runtime']
            })

        content = ""
        for problem_key in sorted(problem_results.keys()):
            content += f"### {problem_key}\n\n"

            # Sort by objective
            sorted_results = sorted(problem_results[problem_key], key=lambda x: x['objective'])

            content += "| Rank | Method | Objective | Runtime (s) |\n"
            content += "|------|--------|-----------|-------------|\n"

            for rank, res in enumerate(sorted_results, 1):
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
                content += f"| {rank} {medal} | {res['method']} | {res['objective']:.2f} | {res['runtime']:.2f} |\n"

            content += "\n"

        return content

    def _generate_recommendations(self, results: Dict) -> str:
        """Generate recommendations based on results"""
        # Analyze results to generate recommendations
        method_performance = {}

        for result in results.get('results', []):
            if result.get('status') == 'failed':
                continue

            method = result['method']
            size = result['problem'].get('size', 0)

            if method not in method_performance:
                method_performance[method] = {'small': [], 'medium': [], 'large': []}

            category = 'small' if size <= 20 else 'medium' if size <= 50 else 'large'
            method_performance[method][category].append(result['statistics']['best_objective'])

        recommendations = []

        # Find best methods for each size category
        for category in ['small', 'medium', 'large']:
            best_methods = []
            for method, perf in method_performance.items():
                if perf[category]:
                    avg_obj = np.mean(perf[category])
                    best_methods.append((method, avg_obj))

            if best_methods:
                best_methods.sort(key=lambda x: x[1])
                top_3 = best_methods[:3]
                size_name = category.capitalize()
                recommendations.append(
                    f"- **{size_name} problems:** Best performers are " +
                    ", ".join([f"{m[0]} ({m[1]:.2f})" for m in top_3])
                )

        return "\n".join(recommendations) if recommendations else "- Insufficient data for recommendations\n"


class CSVReportGenerator:
    """Generate CSV exports for data analysis"""

    def generate(self, results: Dict, output_path: Path):
        """Generate CSV export

        Args:
            results: Benchmark results dictionary
            output_path: Path to save CSV file
        """
        rows = []

        for result in results.get('results', []):
            if result.get('status') == 'failed':
                continue

            problem = result['problem']
            stats = result['statistics']

            row = {
                'suite_name': results.get('suite_name'),
                'timestamp': results.get('start_time'),
                'git_commit': results.get('metadata', {}).get('git_commit'),
                'problem_type': problem.get('type'),
                'problem_size': problem.get('size'),
                'problem_instance': problem.get('instance', ''),
                'method': result['method'],
                'config': json.dumps(result.get('config', {})),
                'runs': result.get('runs'),
                'mean_objective': stats.get('mean_objective'),
                'std_objective': stats.get('std_objective'),
                'best_objective': stats.get('best_objective'),
                'worst_objective': stats.get('worst_objective'),
                'mean_runtime': stats.get('mean_runtime'),
                'std_runtime': stats.get('std_runtime'),
                'mean_memory': stats.get('mean_memory'),
                'optimality_gap': stats.get('optimality_gap'),
                'success_rate': stats.get('success_rate'),
                'convergence_speed': result.get('convergence', {}).get('mean_iterations_to_90')
            }
            rows.append(row)

        # Create DataFrame and save to CSV
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)


def main():
    """Test report generation"""
    # This would typically be called by the benchmark runner
    # Example usage:
    # with open('benchmark_results/standard_latest.json', 'r') as f:
    #     results = json.load(f)
    #
    # html_gen = HTMLReportGenerator()
    # html_gen.generate(results, Path('test_report.html'))
    #
    # md_gen = MarkdownReportGenerator()
    # md_gen.generate(results, Path('test_report.md'))
    #
    # csv_gen = CSVReportGenerator()
    # csv_gen.generate(results, Path('test_report.csv'))
    pass


if __name__ == "__main__":
    main()