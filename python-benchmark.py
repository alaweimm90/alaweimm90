#!/usr/bin/env python3
"""Quick performance benchmark for Python packages."""
import time
import sys

def benchmark_import(module: str) -> float:
    start = time.perf_counter()
    try:
        __import__(module)
        return time.perf_counter() - start
    except ImportError:
        return -1

if __name__ == "__main__":
    modules = ["equilibria", "mezan", "maglogic", "scicomp"]
    print("🐍 Python Import Times\n")
    for mod in modules:
        t = benchmark_import(mod)
        status = f"{t*1000:.1f}ms" if t > 0 else "not installed"
        print(f"  {mod}: {status}")
