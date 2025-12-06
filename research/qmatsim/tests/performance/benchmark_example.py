import pytest

def expensive_computation(n):
    return sum(range(n))

@pytest.mark.benchmark(group="example-group")
def test_expensive_computation_small(benchmark):
    benchmark(expensive_computation, 100)

@pytest.mark.benchmark(group="example-group")
def test_expensive_computation_medium(benchmark):
    benchmark(expensive_computation, 1000)

@pytest.mark.benchmark(group="example-group")
def test_expensive_computation_large(benchmark):
    benchmark(expensive_computation, 10000)
