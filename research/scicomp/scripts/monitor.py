#!/usr/bin/env python3
"""
SciComp Runtime Monitoring System

Continuous health monitoring and performance tracking for deployed SciComp instances.

Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
"""

import os
import sys
import time
import json
import psutil
import threading
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import warnings
import argparse

# Add Python modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Python"))

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    CLEAR_LINE = '\033[2K\r'

class HealthMonitor:
    """Real-time health monitoring for SciComp."""
    
    def __init__(self, interval=30, dashboard=True):
        self.repo_root = Path(__file__).parent.parent
        self.interval = interval
        self.dashboard = dashboard
        self.running = False
        
        # Metrics storage
        self.metrics = {
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'response_times': deque(maxlen=100),
            'error_count': 0,
            'warning_count': 0,
            'uptime': 0,
            'last_check': None
        }
        
        # Health thresholds
        self.thresholds = {
            'cpu_critical': 90,
            'cpu_warning': 70,
            'memory_critical': 85,
            'memory_warning': 70,
            'response_time_critical': 5.0,  # seconds
            'response_time_warning': 2.0
        }
        
        # Health status
        self.health_status = 'UNKNOWN'
        self.health_score = 0
        
    def check_system_resources(self):
        """Check system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics['cpu_usage'].append(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.metrics['memory_usage'].append(memory_percent)
            
            # Disk usage
            disk = psutil.disk_usage(str(self.repo_root))
            disk_percent = disk.percent
            
            return {
                'cpu': cpu_percent,
                'memory': memory_percent,
                'disk': disk_percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_free_gb': disk.free / (1024**3)
            }
        except Exception as e:
            self.metrics['error_count'] += 1
            return None
    
    def check_module_health(self):
        """Check if core modules are responsive."""
        module_status = {}
        response_times = []
        
        # Test core modules
        modules_to_test = [
            ('Quantum', 'Quantum.core.quantum_states', 'BellStates'),
            ('Linear Algebra', 'Linear_Algebra.core.matrix_operations', 'MatrixOperations'),
            ('Optimization', 'Optimization.unconstrained', 'GradientDescent'),
        ]
        
        for module_name, import_path, class_name in modules_to_test:
            start_time = time.time()
            try:
                module = __import__(import_path, fromlist=[class_name])
                if hasattr(module, class_name):
                    # Try to instantiate or use the class
                    if class_name == 'BellStates':
                        obj = getattr(module, class_name).phi_plus()
                    else:
                        obj = getattr(module, class_name)()
                    
                    response_time = time.time() - start_time
                    response_times.append(response_time)
                    module_status[module_name] = {
                        'status': 'OK',
                        'response_time': response_time
                    }
                else:
                    module_status[module_name] = {
                        'status': 'ERROR',
                        'error': f'{class_name} not found'
                    }
                    self.metrics['error_count'] += 1
                    
            except Exception as e:
                response_time = time.time() - start_time
                module_status[module_name] = {
                    'status': 'ERROR',
                    'error': str(e)[:50],
                    'response_time': response_time
                }
                self.metrics['error_count'] += 1
        
        # Store average response time
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            self.metrics['response_times'].append(avg_response_time)
        
        return module_status
    
    def check_dependencies(self):
        """Check if required dependencies are available."""
        dependency_status = {}
        
        critical_deps = ['numpy', 'scipy', 'matplotlib']
        optional_deps = ['tensorflow', 'torch', 'cupy', 'sympy']
        
        for dep in critical_deps:
            try:
                __import__(dep)
                dependency_status[dep] = 'OK'
            except ImportError:
                dependency_status[dep] = 'MISSING'
                self.metrics['error_count'] += 1
        
        for dep in optional_deps:
            try:
                __import__(dep)
                dependency_status[dep] = 'OK'
            except ImportError:
                dependency_status[dep] = 'OPTIONAL_MISSING'
                self.metrics['warning_count'] += 1
        
        return dependency_status
    
    def calculate_health_score(self, resources, modules, dependencies):
        """Calculate overall health score."""
        score = 100
        
        # Resource usage impacts
        if resources:
            if resources['cpu'] > self.thresholds['cpu_critical']:
                score -= 20
                self.health_status = 'CRITICAL'
            elif resources['cpu'] > self.thresholds['cpu_warning']:
                score -= 10
                if self.health_status != 'CRITICAL':
                    self.health_status = 'WARNING'
            
            if resources['memory'] > self.thresholds['memory_critical']:
                score -= 20
                self.health_status = 'CRITICAL'
            elif resources['memory'] > self.thresholds['memory_warning']:
                score -= 10
                if self.health_status != 'CRITICAL':
                    self.health_status = 'WARNING'
        
        # Module health impacts
        failed_modules = sum(1 for m in modules.values() if m['status'] == 'ERROR')
        score -= failed_modules * 15
        
        # Response time impacts
        if self.metrics['response_times']:
            avg_response = sum(self.metrics['response_times']) / len(self.metrics['response_times'])
            if avg_response > self.thresholds['response_time_critical']:
                score -= 15
                self.health_status = 'CRITICAL'
            elif avg_response > self.thresholds['response_time_warning']:
                score -= 5
                if self.health_status not in ['CRITICAL']:
                    self.health_status = 'WARNING'
        
        # Dependency impacts
        missing_critical = sum(1 for dep, status in dependencies.items() 
                             if status == 'MISSING' and dep in ['numpy', 'scipy', 'matplotlib'])
        score -= missing_critical * 25
        
        # Set health status based on final score
        if score >= 90 and self.health_status not in ['CRITICAL', 'WARNING']:
            self.health_status = 'HEALTHY'
        elif score >= 70 and self.health_status not in ['CRITICAL']:
            self.health_status = 'DEGRADED'
        elif score < 50:
            self.health_status = 'CRITICAL'
        
        self.health_score = max(0, min(100, score))
        return self.health_score
    
    def display_dashboard(self, resources, modules, dependencies):
        """Display monitoring dashboard."""
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Header
        print(f"{Colors.BLUE}{'='*80}{Colors.RESET}")
        print(f"{Colors.BLUE}{Colors.BOLD}📊 SciComp Health Monitor Dashboard 📊{Colors.RESET}")
        print(f"{Colors.BLUE}{'='*80}{Colors.RESET}")
        
        # Timestamp
        print(f"{Colors.CYAN}Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
        print(f"{Colors.CYAN}Uptime: {self.format_uptime()}{Colors.RESET}")
        print()
        
        # Health Status
        status_color = {
            'HEALTHY': Colors.GREEN,
            'DEGRADED': Colors.YELLOW,
            'WARNING': Colors.YELLOW,
            'CRITICAL': Colors.RED,
            'UNKNOWN': Colors.MAGENTA
        }.get(self.health_status, Colors.RESET)
        
        print(f"{Colors.BOLD}Health Status: {status_color}{self.health_status}{Colors.RESET}")
        print(f"{Colors.BOLD}Health Score: {self.get_score_color()}{self.health_score}/100{Colors.RESET}")
        print()
        
        # System Resources
        print(f"{Colors.BOLD}System Resources:{Colors.RESET}")
        if resources:
            cpu_color = self.get_metric_color(resources['cpu'], 
                                             self.thresholds['cpu_warning'],
                                             self.thresholds['cpu_critical'])
            mem_color = self.get_metric_color(resources['memory'],
                                             self.thresholds['memory_warning'],
                                             self.thresholds['memory_critical'])
            
            print(f"  CPU Usage: {cpu_color}{resources['cpu']:.1f}%{Colors.RESET}")
            print(f"  Memory Usage: {mem_color}{resources['memory']:.1f}%{Colors.RESET} "
                  f"({resources['memory_available_gb']:.1f} GB available)")
            print(f"  Disk Usage: {resources['disk']:.1f}% "
                  f"({resources['disk_free_gb']:.1f} GB free)")
        print()
        
        # Module Health
        print(f"{Colors.BOLD}Module Health:{Colors.RESET}")
        for module_name, status in modules.items():
            if status['status'] == 'OK':
                symbol = f"{Colors.GREEN}✓{Colors.RESET}"
                time_str = f"({status['response_time']:.3f}s)"
            else:
                symbol = f"{Colors.RED}✗{Colors.RESET}"
                time_str = f"({status.get('error', 'Unknown error')[:30]})"
            print(f"  {symbol} {module_name}: {time_str}")
        print()
        
        # Dependencies
        print(f"{Colors.BOLD}Dependencies:{Colors.RESET}")
        critical_deps = ['numpy', 'scipy', 'matplotlib']
        for dep, status in dependencies.items():
            if dep in critical_deps:
                if status == 'OK':
                    print(f"  {Colors.GREEN}✓{Colors.RESET} {dep}")
                else:
                    print(f"  {Colors.RED}✗{Colors.RESET} {dep} ({status})")
        print()
        
        # Metrics Summary
        print(f"{Colors.BOLD}Metrics Summary:{Colors.RESET}")
        print(f"  Errors: {Colors.RED if self.metrics['error_count'] > 0 else Colors.GREEN}"
              f"{self.metrics['error_count']}{Colors.RESET}")
        print(f"  Warnings: {Colors.YELLOW if self.metrics['warning_count'] > 0 else Colors.GREEN}"
              f"{self.metrics['warning_count']}{Colors.RESET}")
        
        if self.metrics['response_times']:
            avg_response = sum(self.metrics['response_times']) / len(self.metrics['response_times'])
            resp_color = self.get_metric_color(avg_response,
                                              self.thresholds['response_time_warning'],
                                              self.thresholds['response_time_critical'])
            print(f"  Avg Response Time: {resp_color}{avg_response:.3f}s{Colors.RESET}")
        
        print(f"\n{Colors.BLUE}{'='*80}{Colors.RESET}")
        print(f"{Colors.CYAN}Press Ctrl+C to stop monitoring{Colors.RESET}")
    
    def get_metric_color(self, value, warning_threshold, critical_threshold):
        """Get color based on metric value and thresholds."""
        if value >= critical_threshold:
            return Colors.RED
        elif value >= warning_threshold:
            return Colors.YELLOW
        else:
            return Colors.GREEN
    
    def get_score_color(self):
        """Get color based on health score."""
        if self.health_score >= 90:
            return Colors.GREEN
        elif self.health_score >= 70:
            return Colors.YELLOW
        else:
            return Colors.RED
    
    def format_uptime(self):
        """Format uptime for display."""
        uptime_seconds = self.metrics['uptime']
        hours = uptime_seconds // 3600
        minutes = (uptime_seconds % 3600) // 60
        seconds = uptime_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def run_health_check(self):
        """Run a single health check cycle."""
        # Check system resources
        resources = self.check_system_resources()
        
        # Check module health
        modules = self.check_module_health()
        
        # Check dependencies
        dependencies = self.check_dependencies()
        
        # Calculate health score
        self.calculate_health_score(resources, modules, dependencies)
        
        # Update metrics
        self.metrics['last_check'] = datetime.now()
        self.metrics['uptime'] += self.interval
        
        # Display dashboard or return results
        if self.dashboard:
            self.display_dashboard(resources, modules, dependencies)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'health_status': self.health_status,
            'health_score': self.health_score,
            'resources': resources,
            'modules': modules,
            'dependencies': dependencies,
            'metrics': {
                'error_count': self.metrics['error_count'],
                'warning_count': self.metrics['warning_count'],
                'uptime': self.metrics['uptime']
            }
        }
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        self.running = True
        
        print(f"{Colors.GREEN}Starting SciComp Health Monitor...{Colors.RESET}")
        print(f"Monitoring interval: {self.interval} seconds")
        print(f"Dashboard: {'Enabled' if self.dashboard else 'Disabled'}")
        time.sleep(2)
        
        try:
            while self.running:
                result = self.run_health_check()
                
                # Log results if not in dashboard mode
                if not self.dashboard:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Status: {result['health_status']} | "
                          f"Score: {result['health_score']}/100 | "
                          f"Errors: {result['metrics']['error_count']}")
                
                # Sleep until next check
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop monitoring gracefully."""
        self.running = False
        print(f"\n{Colors.YELLOW}Stopping health monitor...{Colors.RESET}")
        
        # Save final metrics
        self.save_metrics()
        
        print(f"{Colors.GREEN}Health monitor stopped{Colors.RESET}")
        print(f"Total uptime: {self.format_uptime()}")
        print(f"Total errors: {self.metrics['error_count']}")
        print(f"Total warnings: {self.metrics['warning_count']}")
    
    def save_metrics(self):
        """Save metrics to file."""
        metrics_file = self.repo_root / f"health_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'uptime': self.metrics['uptime'],
            'final_status': self.health_status,
            'final_score': self.health_score,
            'error_count': self.metrics['error_count'],
            'warning_count': self.metrics['warning_count'],
            'cpu_history': list(self.metrics['cpu_usage']),
            'memory_history': list(self.metrics['memory_usage']),
            'response_time_history': list(self.metrics['response_times'])
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"Metrics saved to: {metrics_file.name}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='SciComp Health Monitor')
    parser.add_argument('--interval', type=int, default=30,
                       help='Monitoring interval in seconds (default: 30)')
    parser.add_argument('--no-dashboard', action='store_true',
                       help='Disable dashboard display (log mode only)')
    parser.add_argument('--once', action='store_true',
                       help='Run health check once and exit')
    
    args = parser.parse_args()
    
    monitor = HealthMonitor(interval=args.interval, dashboard=not args.no_dashboard)
    
    if args.once:
        # Run single health check
        result = monitor.run_health_check()
        print(f"\nHealth Status: {result['health_status']}")
        print(f"Health Score: {result['health_score']}/100")
        sys.exit(0 if result['health_score'] >= 70 else 1)
    else:
        # Start continuous monitoring
        try:
            monitor.start_monitoring()
        except Exception as e:
            print(f"{Colors.RED}Monitor error: {str(e)}{Colors.RESET}")
            sys.exit(1)

if __name__ == "__main__":
    main()