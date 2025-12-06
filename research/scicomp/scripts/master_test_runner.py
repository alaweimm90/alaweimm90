#!/usr/bin/env python3
"""
SciComp Master Test Runner

Orchestrates comprehensive testing across all SciComp modules and frameworks.
Provides unified testing interface with detailed reporting and analysis.

Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse

class Colors:
    """Enhanced color scheme for master test output."""
    BERKELEY_BLUE = '\033[38;5;18m'
    CALIFORNIA_GOLD = '\033[38;5;178m'
    SUCCESS = '\033[38;5;46m'
    WARNING = '\033[38;5;208m'
    ERROR = '\033[38;5;196m'
    INFO = '\033[38;5;39m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

class MasterTestRunner:
    """Master test coordinator for SciComp."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
        # Test frameworks available
        self.test_frameworks = {
            'validation': {
                'script': 'validate_framework.py',
                'description': 'Core functionality validation',
                'critical': True,
                'timeout': 300
            },
            'consistency': {
                'script': 'consistency_check.py',
                'description': 'Repository consistency verification',
                'critical': True,
                'timeout': 120
            },
            'comprehensive': {
                'script': 'comprehensive_test_suite.py',
                'description': 'Deep analysis and syntax validation',
                'critical': False,
                'timeout': 600
            },
            'security': {
                'script': 'security_audit.py',
                'description': 'Security vulnerability scanning',
                'critical': False,
                'timeout': 180
            },
            'performance': {
                'script': 'performance_regression_tests.py',
                'description': 'Performance regression testing',
                'critical': False,
                'timeout': 900
            },
            'integration': {
                'script': 'integration_tests.py',
                'description': 'Module integration testing',
                'critical': True,
                'timeout': 450
            }
        }
    
    def print_master_header(self):
        """Print master test runner header."""
        print(f"\n{Colors.BERKELEY_BLUE}{'='*100}{Colors.RESET}")
        print(f"{Colors.BERKELEY_BLUE}{Colors.BOLD}🧪 SciComp Master Test Runner 🧪{Colors.RESET}")
        print(f"{Colors.BERKELEY_BLUE}{'='*100}{Colors.RESET}")
        print(f"{Colors.CALIFORNIA_GOLD}University of California, Berkeley{Colors.RESET}")
        print(f"{Colors.CALIFORNIA_GOLD}Comprehensive Quality Assurance & Deployment Readiness Testing{Colors.RESET}")
        print(f"{Colors.BERKELEY_BLUE}{'='*100}{Colors.RESET}")
        print(f"{Colors.DIM}Test Suite Version: 2.0 | Framework: SciComp | Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}\n")
    
    def run_test_framework(self, framework_name: str, framework_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific test framework."""
        script_path = self.repo_root / "scripts" / framework_config['script']
        
        if not script_path.exists():
            return {
                'status': 'SKIPPED',
                'reason': f'Script not found: {framework_config["script"]}',
                'execution_time': 0,
                'output': '',
                'error': f'Missing script: {script_path}'
            }
        
        print(f"{Colors.INFO}🔬 Running {framework_config['description']}...{Colors.RESET}")
        print(f"{Colors.DIM}   Script: {framework_config['script']}{Colors.RESET}")
        
        start_time = time.time()
        
        try:
            # Run the test script
            result = subprocess.run(
                [sys.executable, str(script_path), '--repo-root', str(self.repo_root)],
                capture_output=True,
                text=True,
                timeout=framework_config['timeout'],
                cwd=self.repo_root
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Determine status based on return code and output
            status = 'PASSED' if result.returncode == 0 else 'FAILED'
            
            # Parse output for additional information
            output_lines = result.stdout.split('\n')
            error_lines = result.stderr.split('\n') if result.stderr else []
            
            # Extract key metrics from output
            metrics = self._extract_metrics_from_output(output_lines, framework_name)
            
            test_result = {
                'status': status,
                'execution_time': execution_time,
                'return_code': result.returncode,
                'output': result.stdout,
                'error': result.stderr,
                'metrics': metrics,
                'output_lines': len(output_lines),
                'critical': framework_config['critical']
            }
            
            # Print immediate feedback
            if status == 'PASSED':
                print(f"   {Colors.SUCCESS}✅ PASSED{Colors.RESET} ({execution_time:.1f}s)")
            else:
                print(f"   {Colors.ERROR}❌ FAILED{Colors.RESET} ({execution_time:.1f}s)")
                if result.stderr:
                    print(f"   {Colors.DIM}Error: {result.stderr.split(chr(10))[0][:100]}...{Colors.RESET}")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"   {Colors.WARNING}⏰ TIMEOUT{Colors.RESET} ({execution_time:.1f}s)")
            
            return {
                'status': 'TIMEOUT',
                'execution_time': execution_time,
                'return_code': -1,
                'output': '',
                'error': f'Test timed out after {framework_config["timeout"]}s',
                'metrics': {},
                'critical': framework_config['critical']
            }
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"   {Colors.ERROR}💥 ERROR{Colors.RESET} ({execution_time:.1f}s)")
            
            return {
                'status': 'ERROR',
                'execution_time': execution_time,
                'return_code': -1,
                'output': '',
                'error': str(e),
                'metrics': {},
                'critical': framework_config['critical']
            }
    
    def _extract_metrics_from_output(self, output_lines: List[str], framework_name: str) -> Dict[str, Any]:
        """Extract key metrics from test output."""
        metrics = {}
        
        try:
            # Framework-specific metric extraction
            if framework_name == 'validation':
                for line in output_lines:
                    if 'Success Rate:' in line:
                        rate = line.split('Success Rate:')[1].strip().rstrip('%')
                        metrics['success_rate'] = float(rate)
                    elif 'Total Tests:' in line:
                        count = line.split('Total Tests:')[1].strip()
                        metrics['total_tests'] = int(count)
                    elif 'PASSED' in line and 'FAILED' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'Passed:' and i+1 < len(parts):
                                metrics['passed_tests'] = int(parts[i+1])
                            elif part == 'Failed:' and i+1 < len(parts):
                                metrics['failed_tests'] = int(parts[i+1])
            
            elif framework_name == 'security':
                for line in output_lines:
                    if 'Security Score:' in line:
                        score = line.split('Security Score:')[1].strip().split('/')[0]
                        metrics['security_score'] = int(score)
                    elif 'HIGH Severity Issues:' in line:
                        count = line.split('HIGH Severity Issues:')[1].strip()
                        metrics['high_severity_issues'] = int(count)
                    elif 'Files Scanned:' in line:
                        count = line.split('Files Scanned:')[1].strip()
                        metrics['files_scanned'] = int(count)
            
            elif framework_name == 'performance':
                for line in output_lines:
                    if 'Performance Regressions:' in line:
                        count = line.split('Performance Regressions:')[1].strip()
                        metrics['regressions'] = int(count)
                    elif 'Successful Tests:' in line:
                        parts = line.split('Successful Tests:')[1].strip().split('/')
                        if len(parts) == 2:
                            metrics['successful_perf_tests'] = int(parts[0])
                            metrics['total_perf_tests'] = int(parts[1])
            
            elif framework_name == 'integration':
                for line in output_lines:
                    if 'Success Rate:' in line:
                        rate = line.split('Success Rate:')[1].strip().rstrip('%')
                        metrics['integration_success_rate'] = float(rate)
                    elif 'Critical Integration Failures:' in line:
                        count = line.split('Critical Integration Failures:')[1].strip()
                        metrics['critical_failures'] = int(count)
            
            elif framework_name == 'comprehensive':
                for line in output_lines:
                    if 'Success Rate:' in line:
                        rate = line.split('Success Rate:')[1].strip().rstrip('%')
                        metrics['comprehensive_success_rate'] = float(rate)
                    elif 'Total Tests:' in line:
                        count = line.split('Total Tests:')[1].strip()
                        metrics['comprehensive_total_tests'] = int(count)
            
        except (ValueError, IndexError, AttributeError):
            # If metric extraction fails, continue with empty metrics
            pass
        
        return metrics
    
    def run_all_tests(self, selected_frameworks: Optional[List[str]] = None, 
                     skip_non_critical: bool = False) -> Dict[str, Any]:
        """Run all selected test frameworks."""
        self.start_time = time.time()
        
        # Determine which frameworks to run
        frameworks_to_run = {}
        
        if selected_frameworks:
            for fw_name in selected_frameworks:
                if fw_name in self.test_frameworks:
                    frameworks_to_run[fw_name] = self.test_frameworks[fw_name]
        else:
            frameworks_to_run = self.test_frameworks.copy()
        
        # Filter out non-critical if requested
        if skip_non_critical:
            frameworks_to_run = {
                name: config for name, config in frameworks_to_run.items()
                if config['critical']
            }
        
        print(f"{Colors.CALIFORNIA_GOLD}Running {len(frameworks_to_run)} test framework(s):{Colors.RESET}")
        for name, config in frameworks_to_run.items():
            critical_marker = "🚨 CRITICAL" if config['critical'] else "📊 STANDARD"
            print(f"   • {config['description']} {critical_marker}")
        print()
        
        # Run each framework
        for framework_name, framework_config in frameworks_to_run.items():
            print(f"\n{Colors.BOLD}{'─'*80}{Colors.RESET}")
            result = self.run_test_framework(framework_name, framework_config)
            self.test_results[framework_name] = result
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        return self.generate_master_report()
    
    def generate_master_report(self) -> Dict[str, Any]:
        """Generate comprehensive master test report."""
        total_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Calculate summary statistics
        total_frameworks = len(self.test_results)
        passed_frameworks = sum(1 for r in self.test_results.values() if r['status'] == 'PASSED')
        failed_frameworks = sum(1 for r in self.test_results.values() if r['status'] == 'FAILED')
        critical_failures = sum(1 for r in self.test_results.values() 
                              if r['status'] in ['FAILED', 'ERROR', 'TIMEOUT'] and r['critical'])
        
        # Aggregate metrics
        aggregated_metrics = {}
        
        for fw_name, result in self.test_results.items():
            for metric_name, metric_value in result.get('metrics', {}).items():
                if metric_name not in aggregated_metrics:
                    aggregated_metrics[metric_name] = metric_value
                # For some metrics, we might want to average or sum - this is simplified
        
        # Determine overall status
        if critical_failures > 0:
            overall_status = 'CRITICAL_FAILURE'
        elif failed_frameworks > 0:
            overall_status = 'FAILED'
        elif total_frameworks == passed_frameworks:
            overall_status = 'PASSED'
        else:
            overall_status = 'PARTIAL'
        
        # Calculate deployment readiness score
        deployment_score = self._calculate_deployment_score()
        
        master_report = {
            'timestamp': datetime.now().isoformat(),
            'execution_time': total_time,
            'overall_status': overall_status,
            'deployment_readiness_score': deployment_score,
            'summary': {
                'total_frameworks': total_frameworks,
                'passed_frameworks': passed_frameworks,
                'failed_frameworks': failed_frameworks,
                'critical_failures': critical_failures,
                'framework_success_rate': (passed_frameworks / total_frameworks * 100) if total_frameworks > 0 else 0
            },
            'frameworks': self.test_results,
            'aggregated_metrics': aggregated_metrics,
            'recommendations': self._generate_recommendations()
        }
        
        return master_report
    
    def _calculate_deployment_score(self) -> int:
        """Calculate overall deployment readiness score (0-100)."""
        score = 100
        
        # Count total frameworks and successes for base score
        total_frameworks = len(self.test_results)
        passed_frameworks = sum(1 for r in self.test_results.values() if r['status'] == 'PASSED')
        
        if total_frameworks > 0:
            base_success_rate = passed_frameworks / total_frameworks
            score = int(60 + (base_success_rate * 40))  # Base score 60-100 based on pass rate
        
        # Adjust for critical vs non-critical failures
        critical_failures = 0
        non_critical_failures = 0
        
        for fw_name, result in self.test_results.items():
            if result['status'] in ['FAILED', 'ERROR', 'TIMEOUT']:
                if result.get('critical', False):
                    critical_failures += 1
                else:
                    non_critical_failures += 1
        
        # More nuanced scoring adjustments
        if critical_failures == 0:
            score += 10  # Bonus for no critical failures
        else:
            score -= min(30, critical_failures * 15)  # Bigger penalty for critical failures
        
        if non_critical_failures > 0:
            score -= min(15, non_critical_failures * 5)  # Smaller penalty for non-critical
        
        # Metrics-based adjustments (more forgiving)
        metrics = {}
        for result in self.test_results.values():
            metrics.update(result.get('metrics', {}))
        
        # Validation success rate bonus/penalty (more lenient)
        if 'success_rate' in metrics:
            if metrics['success_rate'] >= 85:
                score += 5
            elif metrics['success_rate'] >= 75:
                score += 2
            elif metrics['success_rate'] < 60:
                score -= 8
        
        # Security score bonus/penalty (more realistic)
        if 'security_score' in metrics:
            if metrics['security_score'] >= 80:
                score += 5
            elif metrics['security_score'] >= 60:
                score += 2
            elif metrics['security_score'] < 30:
                score -= 10
        
        # Performance regression penalty (more forgiving)
        if 'regressions' in metrics and metrics['regressions'] > 0:
            score -= min(10, metrics['regressions'] * 3)
        
        # Integration success bonus
        if 'integration_success_rate' in metrics and metrics['integration_success_rate'] >= 90:
            score += 3
        
        return max(50, min(100, score))  # Minimum score of 50 for any working system
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on test results."""
        recommendations = []
        
        # Analyze each framework
        for fw_name, result in self.test_results.items():
            if result['status'] == 'FAILED' and result['critical']:
                recommendations.append(f"🚨 URGENT: Fix critical failure in {fw_name} before deployment")
            
            elif result['status'] == 'FAILED':
                recommendations.append(f"⚠️ Consider fixing issues in {fw_name} for better quality")
            
            elif result['status'] == 'TIMEOUT':
                recommendations.append(f"🔧 Investigate performance issues in {fw_name}")
        
        # Metric-based recommendations
        metrics = {}
        for result in self.test_results.values():
            metrics.update(result.get('metrics', {}))
        
        if metrics.get('success_rate', 100) < 85:
            recommendations.append("📈 Improve core validation success rate (target: >85%)")
        
        if metrics.get('security_score', 100) < 75:
            recommendations.append("🔒 Address security vulnerabilities found in audit")
        
        if metrics.get('regressions', 0) > 0:
            recommendations.append("⚡ Fix performance regressions detected")
        
        if metrics.get('critical_failures', 0) > 0:
            recommendations.append("🔗 Resolve critical integration failures")
        
        # General recommendations
        if not recommendations:
            recommendations.append("✅ All tests passing - repository ready for deployment!")
        else:
            recommendations.append("📋 Run tests again after addressing issues above")
        
        return recommendations
    
    def print_master_summary(self, report: Dict[str, Any]):
        """Print comprehensive master test summary."""
        print(f"\n{Colors.BERKELEY_BLUE}{'='*100}{Colors.RESET}")
        print(f"{Colors.BERKELEY_BLUE}{Colors.BOLD}📊 MASTER TEST REPORT 📊{Colors.RESET}")
        print(f"{Colors.BERKELEY_BLUE}{'='*100}{Colors.RESET}")
        
        summary = report['summary']
        
        print(f"\n{Colors.CALIFORNIA_GOLD}{Colors.BOLD}📈 EXECUTION SUMMARY:{Colors.RESET}")
        print(f"   Total Execution Time: {report['execution_time']:.1f}s")
        print(f"   Test Frameworks Run: {summary['total_frameworks']}")
        print(f"   Framework Success Rate: {summary['framework_success_rate']:.1f}%")
        print(f"   Deployment Readiness Score: {Colors.BOLD}{report['deployment_readiness_score']}/100{Colors.RESET}")
        
        print(f"\n{Colors.CALIFORNIA_GOLD}{Colors.BOLD}🎯 FRAMEWORK RESULTS:{Colors.RESET}")
        for fw_name, result in report['frameworks'].items():
            status_color = Colors.SUCCESS if result['status'] == 'PASSED' else Colors.ERROR
            status_icon = '✅' if result['status'] == 'PASSED' else '❌'
            critical_marker = ' 🚨' if result.get('critical', False) else ''
            
            print(f"   {status_icon} {fw_name.title()}: {status_color}{result['status']}{Colors.RESET} " +
                  f"({result['execution_time']:.1f}s){critical_marker}")
            
            # Show key metrics for each framework
            metrics = result.get('metrics', {})
            if metrics:
                metric_strs = []
                for key, value in list(metrics.items())[:3]:  # Show first 3 metrics
                    if isinstance(value, float):
                        metric_strs.append(f"{key}: {value:.1f}")
                    else:
                        metric_strs.append(f"{key}: {value}")
                if metric_strs:
                    print(f"      {Colors.DIM}{' | '.join(metric_strs)}{Colors.RESET}")
        
        # Show aggregated metrics
        if report['aggregated_metrics']:
            print(f"\n{Colors.CALIFORNIA_GOLD}{Colors.BOLD}🔬 KEY METRICS:{Colors.RESET}")
            for metric, value in report['aggregated_metrics'].items():
                if isinstance(value, float):
                    print(f"   • {metric.replace('_', ' ').title()}: {value:.1f}")
                else:
                    print(f"   • {metric.replace('_', ' ').title()}: {value}")
        
        # Overall assessment
        print(f"\n{Colors.CALIFORNIA_GOLD}{Colors.BOLD}🏆 OVERALL ASSESSMENT:{Colors.RESET}")
        
        if report['overall_status'] == 'PASSED':
            print(f"{Colors.SUCCESS}{Colors.BOLD}🚀 EXCELLENT: Repository is deployment-ready!{Colors.RESET}")
            print(f"{Colors.BERKELEY_BLUE}🐻 Go Bears! All systems green for production! 🐻{Colors.RESET}")
        elif report['overall_status'] == 'PARTIAL':
            print(f"{Colors.WARNING}{Colors.BOLD}⚠️ GOOD: Most tests passing, minor issues detected{Colors.RESET}")
            print("   Consider addressing non-critical issues for optimal quality")
        elif report['overall_status'] == 'FAILED':
            print(f"{Colors.ERROR}{Colors.BOLD}❌ NEEDS WORK: Multiple test failures detected{Colors.RESET}")
            print("   Address failing tests before deployment")
        else:  # CRITICAL_FAILURE
            print(f"{Colors.ERROR}{Colors.BOLD}🚨 CRITICAL: Critical test failures detected{Colors.RESET}")
            print("   Must resolve critical issues before any deployment")
        
        # Show recommendations
        if report['recommendations']:
            print(f"\n{Colors.CALIFORNIA_GOLD}{Colors.BOLD}💡 RECOMMENDATIONS:{Colors.RESET}")
            for recommendation in report['recommendations']:
                print(f"   {recommendation}")
        
        print(f"\n{Colors.BERKELEY_BLUE}{'='*100}{Colors.RESET}")
        print(f"{Colors.CALIFORNIA_GOLD}SciComp Master Testing Complete - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
        print(f"{Colors.BERKELEY_BLUE}{'='*100}{Colors.RESET}\n")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='SciComp Master Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available test frameworks:
  validation     - Core functionality validation (critical)
  consistency    - Repository consistency verification (critical)  
  comprehensive  - Deep analysis and syntax validation
  security       - Security vulnerability scanning
  performance    - Performance regression testing
  integration    - Module integration testing (critical)

Examples:
  python master_test_runner.py                                    # Run all tests
  python master_test_runner.py --frameworks validation security   # Run specific tests
  python master_test_runner.py --critical-only                   # Run only critical tests
  python master_test_runner.py --export-report full_report.json  # Export detailed report
        """
    )
    
    parser.add_argument('--repo-root', default='.', 
                       help='Repository root directory (default: current directory)')
    parser.add_argument('--frameworks', nargs='*', 
                       choices=['validation', 'consistency', 'comprehensive', 'security', 'performance', 'integration'],
                       help='Specific frameworks to run (default: all)')
    parser.add_argument('--critical-only', action='store_true',
                       help='Run only critical test frameworks')
    parser.add_argument('--export-report', metavar='FILE',
                       help='Export detailed report to JSON file')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output, show summary only')
    
    args = parser.parse_args()
    
    # Initialize master test runner
    runner = MasterTestRunner(args.repo_root)
    
    if not args.quiet:
        runner.print_master_header()
    
    # Run tests
    report = runner.run_all_tests(
        selected_frameworks=args.frameworks,
        skip_non_critical=args.critical_only
    )
    
    # Print summary
    if not args.quiet:
        runner.print_master_summary(report)
    else:
        # Quiet mode - just show final status
        status = report['overall_status']
        score = report['deployment_readiness_score']
        print(f"Overall Status: {status}")
        print(f"Deployment Readiness Score: {score}/100")
    
    # Export report if requested
    if args.export_report:
        with open(args.export_report, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Detailed report exported to: {args.export_report}")
    
    # Exit with appropriate code
    if report['overall_status'] in ['CRITICAL_FAILURE', 'FAILED']:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()