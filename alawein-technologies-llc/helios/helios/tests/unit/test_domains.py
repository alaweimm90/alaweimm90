"""
Unit tests for HELIOS Domains.

Tests for:
- Domain registration
- Domain interfaces
- Domain-specific functionality
"""

import pytest


class TestDomainRegistry:
    """Test domain registration and access."""

    @pytest.mark.unit
    def test_domains_importable(self):
        """Test that DOMAINS can be imported."""
        try:
            from helios.domains import DOMAINS
            assert isinstance(DOMAINS, dict)
        except ImportError:
            pytest.skip("DOMAINS not available")

    @pytest.mark.unit
    def test_all_domains_registered(self, available_domains):
        """Test that all expected domains are registered."""
        try:
            from helios.domains import DOMAINS

            for domain_name in available_domains:
                assert domain_name in DOMAINS, f"{domain_name} not in DOMAINS"
        except ImportError:
            pytest.skip("DOMAINS not available")

    @pytest.mark.unit
    def test_domain_count(self):
        """Test that exactly 7 domains are registered."""
        try:
            from helios.domains import DOMAINS
            assert len(DOMAINS) == 7
        except ImportError:
            pytest.skip("DOMAINS not available")


class TestDomainInterfaces:
    """Test domain interface compliance."""

    @pytest.mark.unit
    def test_quantum_domain(self):
        """Test QuantumDomain interface."""
        try:
            from helios.domains.quantum import QuantumDomain

            domain = QuantumDomain()
            assert hasattr(domain, 'name')
            assert hasattr(domain, 'display_name')
            assert hasattr(domain, 'description')
            assert hasattr(domain, 'generate_benchmark_problems')
        except ImportError:
            pytest.skip("QuantumDomain not available")

    @pytest.mark.unit
    def test_materials_domain(self):
        """Test MaterialsDomain interface."""
        try:
            from helios.domains.materials import MaterialsDomain

            domain = MaterialsDomain()
            assert hasattr(domain, 'name')
            assert domain.name == 'materials'
        except ImportError:
            pytest.skip("MaterialsDomain not available")

    @pytest.mark.unit
    def test_optimization_domain(self):
        """Test OptimizationDomain interface."""
        try:
            from helios.domains.optimization import OptimizationDomain

            domain = OptimizationDomain()
            assert hasattr(domain, 'name')
            assert domain.name == 'optimization'
        except ImportError:
            pytest.skip("OptimizationDomain not available")

    @pytest.mark.unit
    def test_ml_domain(self):
        """Test MLDomain interface."""
        try:
            from helios.domains.ml import MLDomain

            domain = MLDomain()
            assert hasattr(domain, 'name')
            assert domain.name == 'ml'
        except ImportError:
            pytest.skip("MLDomain not available")

    @pytest.mark.unit
    def test_nas_domain(self):
        """Test NASDomain interface."""
        try:
            from helios.domains.nas import NASDomain

            domain = NASDomain()
            assert hasattr(domain, 'name')
            assert domain.name == 'nas'
        except ImportError:
            pytest.skip("NASDomain not available")

    @pytest.mark.unit
    def test_synthesis_domain(self):
        """Test SynthesisDomain interface."""
        try:
            from helios.domains.synthesis import SynthesisDomain

            domain = SynthesisDomain()
            assert hasattr(domain, 'name')
            assert domain.name == 'synthesis'
        except ImportError:
            pytest.skip("SynthesisDomain not available")

    @pytest.mark.unit
    def test_graph_domain(self):
        """Test GraphDomain interface."""
        try:
            from helios.domains.graph import GraphDomain

            domain = GraphDomain()
            assert hasattr(domain, 'name')
            assert domain.name == 'graph'
        except ImportError:
            pytest.skip("GraphDomain not available")


class TestDomainInstantiation:
    """Test domain instantiation."""

    @pytest.mark.unit
    def test_instantiate_domains(self, available_domains):
        """Test that all domains can be instantiated."""
        try:
            from helios.domains import DOMAINS

            for domain_name in available_domains:
                domain = DOMAINS[domain_name]()
                assert domain is not None
                assert domain.name == domain_name
        except ImportError:
            pytest.skip("DOMAINS not available")

    @pytest.mark.unit
    def test_domain_has_display_name(self, available_domains):
        """Test that domains have display names."""
        try:
            from helios.domains import DOMAINS

            for domain_name in available_domains:
                domain = DOMAINS[domain_name]()
                assert hasattr(domain, 'display_name')
                assert isinstance(domain.display_name, str)
                assert len(domain.display_name) > 0
        except ImportError:
            pytest.skip("DOMAINS not available")

    @pytest.mark.unit
    def test_domain_has_description(self, available_domains):
        """Test that domains have descriptions."""
        try:
            from helios.domains import DOMAINS

            for domain_name in available_domains:
                domain = DOMAINS[domain_name]()
                assert hasattr(domain, 'description')
                assert isinstance(domain.description, str)
        except ImportError:
            pytest.skip("DOMAINS not available")
