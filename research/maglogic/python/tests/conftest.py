"""
Pytest configuration and shared fixtures for MagLogic tests.

This module provides common fixtures and configuration for all test modules,
ensuring consistent test environments and reducing code duplication.

Author: Meshal Alawein
Email: meshal@berkeley.edu
License: MIT
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import matplotlib
import logging

# Configure matplotlib for testing
matplotlib.use('Agg')

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_ovf_2x2():
    """Create a simple 2x2 OVF file content for testing."""
    return """# OOMMF OVF 2.0
# Segment count: 1
# Begin: Segment
# Begin: Header
# Title: Test 2x2 magnetization
# meshtype: rectangular
# meshunit: m
# xnodes: 2
# ynodes: 2
# znodes: 1
# xstepsize: 1e-9
# ystepsize: 1e-9
# zstepsize: 1e-9
# xmin: 0
# ymin: 0
# zmin: 0
# xmax: 1e-9
# ymax: 1e-9
# zmax: 0
# valuedim: 3
# valueunit: A/m
# valuemultiplier: 8.6e5
# End: Header
# Begin: Data Text
1.0 0.0 0.0
0.0 1.0 0.0
-1.0 0.0 0.0
0.0 -1.0 0.0
# End: Data Text
# End: Segment
"""


@pytest.fixture
def sample_ovf_5x5():
    """Create a 5x5 OVF file with vortex pattern for testing."""
    content = """# OOMMF OVF 2.0
# Segment count: 1
# Begin: Segment
# Begin: Header
# Title: Test 5x5 vortex
# meshtype: rectangular
# meshunit: m
# xnodes: 5
# ynodes: 5
# znodes: 1
# xstepsize: 1e-9
# ystepsize: 1e-9
# zstepsize: 1e-9
# xmin: 0
# ymin: 0
# zmin: 0
# xmax: 4e-9
# ymax: 4e-9
# zmax: 0
# valuedim: 3
# valueunit: A/m
# valuemultiplier: 8.6e5
# End: Header
# Begin: Data Text
"""
    
    # Create vortex pattern
    size = 5
    center = size // 2
    
    for j in range(size):
        for i in range(size):
            x = i - center
            y = j - center
            r = np.sqrt(x**2 + y**2)
            
            if r > 0:
                # Vortex circulation
                mx = -y / (r + 0.5)
                my = x / (r + 0.5)
                mz = np.tanh(r - 1.5)
            else:
                mx, my, mz = 0, 0, 1
            
            # Normalize
            mag = np.sqrt(mx**2 + my**2 + mz**2)
            if mag > 0:
                mx /= mag
                my /= mag
                mz /= mag
            
            content += f"{mx:.6f} {my:.6f} {mz:.6f}\n"
    
    content += "# End: Data Text\n# End: Segment\n"
    return content


@pytest.fixture
def sample_odt_content():
    """Create sample ODT (OOMMF Data Table) content."""
    return """# ODT 1.0
# Table Start
# Columns: Time {s} Energy {J} mx {} my {} mz {}
# Units:   s       J       {}   {}   {}
0.0      -1.23456e-12  1.000  0.000  0.000
1.0e-12  -1.23457e-12  0.995  0.100  0.000
2.0e-12  -1.23458e-12  0.980  0.199  0.000
3.0e-12  -1.23459e-12  0.955  0.296  0.000
4.0e-12  -1.23460e-12  0.921  0.389  0.000
5.0e-12  -1.23461e-12  0.878  0.479  0.000
"""


@pytest.fixture
def sample_mumax3_table():
    """Create sample MuMax3 table.txt content."""
    return """# t (s)	E_total (J)	E_exch (J)	E_demag (J)	mx ()	my ()	mz ()
0	-1.234567e-12	-2.345678e-13	-9.876543e-13	1	0	0
1e-12	-1.234568e-12	-2.345679e-13	-9.876544e-13	0.999	0.01	0
2e-12	-1.234569e-12	-2.345680e-13	-9.876545e-13	0.998	0.02	0
3e-12	-1.234570e-12	-2.345681e-13	-9.876546e-13	0.997	0.03	0
4e-12	-1.234571e-12	-2.345682e-13	-9.876547e-13	0.996	0.04	0
"""


@pytest.fixture
def sample_material_params():
    """Standard material parameters for testing."""
    return {
        'Ms': 8.6e5,      # Saturation magnetization (A/m)
        'A': 1.3e-11,     # Exchange constant (J/m)
        'alpha': 0.008,   # Gilbert damping
        'K1': 0.0,        # Uniaxial anisotropy constant (J/m³)
        'gamma': 1.7608592e11  # Gyromagnetic ratio (rad⋅s⁻¹⋅T⁻¹)
    }


@pytest.fixture
def uniform_magnetization_2d():
    """Create uniform 2D magnetization data for testing."""
    size = 10
    mx = np.ones((size, size))
    my = np.zeros((size, size))
    mz = np.zeros((size, size))
    
    return {
        'mx': mx,
        'my': my,
        'mz': mz,
        'magnitude': np.ones((size, size)),
        'theta': np.zeros((size, size)),
        'phi': np.zeros((size, size)),
        'mx_norm': mx,
        'my_norm': my,
        'mz_norm': mz
    }


@pytest.fixture
def vortex_magnetization_2d():
    """Create vortex magnetization pattern for testing."""
    size = 21  # Odd size for centered vortex
    center = size // 2
    
    mx = np.zeros((size, size))
    my = np.zeros((size, size))
    mz = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            r = np.sqrt(x**2 + y**2)
            
            if r > 0:
                # Vortex circulation (clockwise)
                mx[i, j] = -y / (r + 1)
                my[i, j] = x / (r + 1)
                mz[i, j] = np.tanh(2 * (r - center/2))
            else:
                # Core
                mx[i, j] = 0
                my[i, j] = 0
                mz[i, j] = 1
    
    # Normalize
    magnitude = np.sqrt(mx**2 + my**2 + mz**2)
    magnitude = np.where(magnitude > 0, magnitude, 1)
    
    return {
        'mx': mx,
        'my': my,
        'mz': mz,
        'magnitude': magnitude,
        'theta': np.arccos(np.clip(mz / magnitude, -1, 1)),
        'phi': np.arctan2(my, mx),
        'mx_norm': mx / magnitude,
        'my_norm': my / magnitude,
        'mz_norm': mz / magnitude
    }


@pytest.fixture
def coordinates_2d():
    """Create 2D coordinate arrays for testing."""
    size = 10
    spacing = 1e-9  # 1 nm
    
    x = np.linspace(0, (size-1)*spacing, size)
    y = np.linspace(0, (size-1)*spacing, size)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.zeros_like(X)
    
    return {
        'x': X,
        'y': Y,
        'z': Z
    }


@pytest.fixture
def sample_mif_content():
    """Create sample MIF file content for OOMMF simulations."""
    return """# MagLogic Test MIF File
# Basic micromagnetic simulation setup

SetOptions {
    basename test_simulation
    scalar_output_format %.17g
    vector_field_output_format {text %#.17g}
}

# Material Parameters
Parameter Ms {{Ms}}
Parameter A {{A}}
Parameter alpha {{alpha}}

# Geometry
Parameter width {{width}}
Parameter height {{height}}
Parameter thickness {{thickness}}

Specify Oxs_BoxAtlas:ORCHEX {
    xrange {0 $width}
    yrange {0 $height}
    zrange {0 $thickness}
}

# Mesh
Parameter cell_size {{cell_size}}
Specify Oxs_RectangularMesh:mesh {
    cellsize {$cell_size $cell_size $thickness}
    ORCHEX :ORCHEX
}

# Material Properties
Specify Oxs_UniformExchange {
    A $A
}

Specify Oxs_Demag {}

# Applied Field (optional)
Specify Oxs_UniformField {
    field {0.0 0.0 0.0}
}

# Evolver  
Specify Oxs_CGEvolve:evolve {
    method conjugate_gradient
    gradient_reset_angle 87
}

# Driver
Specify Oxs_MinDriver {
    evolver :evolve
    stopping_mxHxm 0.01
    mesh :mesh
    Ms $Ms
    m0 { Oxs_RandomVectorField {
        min_norm 1.0
        max_norm 1.0
    } }
}

# Output
Destination table mmArchive
Destination mags mmArchive

Schedule DataTable table Stage 1
Schedule Oxs_MinDriver::Magnetization mags Stage 1
"""


@pytest.fixture
def sample_mumax3_content():
    """Create sample MuMax3 script content."""
    return """// MagLogic Test MuMax3 Script
// Basic micromagnetic simulation

// Set grid size
SetGridSize(64, 64, 1)
SetCellSize(2e-9, 2e-9, 5e-9)

// Material parameters
Msat = 860e3  // A/m
Aex = 13e-12  // J/m
alpha = 0.02

// Geometry (could be defined by shape function)
DefRegion(1, rect(64*2e-9, 64*2e-9))

// Set initial magnetization
m = uniform(1, 0, 0)

// Add some disorder
m = m.add(randomMag())
normalize(m)

// Relax the system
Run(1e-9)  // 1 ns

// Output
TableAdd(E_total)
TableSave()
Save(m)
"""


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "oommf: marks tests that require OOMMF installation"
    )
    config.addinivalue_line(
        "markers", "mumax3: marks tests that require MuMax3 installation"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"  
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark slow tests
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Ensure matplotlib figures are cleaned up after each test."""
    yield
    
    import matplotlib.pyplot as plt
    plt.close('all')


@pytest.fixture
def suppress_warnings():
    """Suppress common warnings during testing."""
    import warnings
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        yield


# Custom assertions for testing
def assert_magnetization_normalized(magnetization, tolerance=1e-10):
    """Assert that magnetization vectors are normalized."""
    mx = magnetization['mx']
    my = magnetization['my']
    mz = magnetization['mz']
    
    magnitude = np.sqrt(mx**2 + my**2 + mz**2)
    
    # Check that all magnitudes are close to 1
    np.testing.assert_allclose(magnitude, 1.0, atol=tolerance,
                              err_msg="Magnetization vectors are not normalized")


def assert_ovf_file_valid(filepath):
    """Assert that an OVF file has valid structure."""
    from maglogic.parsers import OOMMFParser
    
    parser = OOMMFParser()
    
    # This should not raise an exception
    data = parser.parse_ovf(filepath)
    
    # Check required fields
    assert 'magnetization' in data
    assert 'coordinates' in data
    assert 'metadata' in data
    assert 'header' in data
    
    # Check magnetization data
    mag = data['magnetization']
    assert 'mx' in mag
    assert 'my' in mag
    assert 'mz' in mag
    
    # Check that shapes are consistent
    assert mag['mx'].shape == mag['my'].shape == mag['mz'].shape


def assert_analysis_complete(analysis_results):
    """Assert that analysis results contain all expected sections."""
    expected_sections = [
        'domain_analysis',
        'energy_analysis', 
        'spatial_analysis',
        'topological_analysis',
        'texture_analysis'
    ]
    
    for section in expected_sections:
        assert section in analysis_results, f"Missing analysis section: {section}"
    
    # Check that each section has reasonable content
    assert isinstance(analysis_results['domain_analysis'], dict)
    assert isinstance(analysis_results['energy_analysis'], dict)
    assert isinstance(analysis_results['spatial_analysis'], dict)
    assert isinstance(analysis_results['topological_analysis'], dict)
    assert isinstance(analysis_results['texture_analysis'], dict)


# Make custom assertions available
pytest.assert_magnetization_normalized = assert_magnetization_normalized
pytest.assert_ovf_file_valid = assert_ovf_file_valid
pytest.assert_analysis_complete = assert_analysis_complete