"""
Test fixtures for AdversarialReview test suite
"""
import pytest


@pytest.fixture
def sample_paper_text():
    """Sample paper text for testing"""
    return """
    This is a sample research paper abstract. We conducted a study with 100 participants
    to investigate the effects of variable X on outcome Y. Results showed a significant
    correlation (p < 0.05). We conclude that X causes Y.
    """


@pytest.fixture
def short_paper_text():
    """Short paper text for testing"""
    return "A brief study on topic Z."


@pytest.fixture
def long_paper_text():
    """Longer paper text for testing"""
    return """
    Title: Advanced Machine Learning for Predicting Stock Prices

    Abstract:
    This paper presents a novel deep learning approach for predicting stock market prices
    with unprecedented accuracy. We trained a neural network on 5 years of historical
    data and achieved 95% accuracy in predicting next-day prices.

    Introduction:
    Stock market prediction has been a challenge for decades. Our approach solves this
    problem definitively using a 10-layer convolutional neural network.

    Methods:
    We collected data from Yahoo Finance for 500 stocks. We split data 80/20 for
    training and testing. The model was trained for 100 epochs.

    Results:
    Our model achieved 95% accuracy on the test set. All p-values were less than 0.05.
    We observed strong correlations across all stock categories.

    Discussion:
    These results prove that stock prices can be perfectly predicted using our method.
    This will revolutionize finance and make everyone rich.

    Conclusion:
    We have solved stock market prediction. Future work is unnecessary.
    """


@pytest.fixture
def paper_title():
    """Sample paper title"""
    return "Sample Research Paper: Testing Hypothesis X"


@pytest.fixture
def alternative_title():
    """Alternative paper title"""
    return "Advanced AI Methods for Universal Problem Solving"


@pytest.fixture
def nightmare_mode():
    """Nightmare review mode"""
    return "nightmare"


@pytest.fixture
def normal_mode():
    """Normal/standard review mode"""
    return "standard"


@pytest.fixture
def brutal_mode():
    """Brutal review mode"""
    return "brutal"


# Expected value ranges for different modes
@pytest.fixture
def expected_score_ranges():
    """Expected score ranges for different severities"""
    return {
        "CRITICAL": (0.0, 4.0),
        "MAJOR": (4.0, 6.0),
        "MINOR": (6.0, 10.0)
    }


@pytest.fixture
def valid_severity_levels():
    """Valid severity levels"""
    return ["CRITICAL", "MAJOR", "MINOR"]


@pytest.fixture
def valid_verdicts():
    """Valid verdict options"""
    return ["REJECT", "MAJOR_REVISION", "MINOR_REVISION", "ACCEPT"]


@pytest.fixture
def critic_dimensions():
    """Expected critic dimensions"""
    return {
        "Statistical Skeptic": "statistical_validity",
        "Methodology Maverick": "methodological_rigor",
        "Logic Enforcer": "logical_consistency",
        "History Hunter": "historical_context",
        "Ethics Enforcer": "ethical_implications",
        "Economic Realist": "economic_feasibility"
    }


@pytest.fixture
def minimum_issues_by_mode():
    """Minimum expected issues by mode"""
    return {
        "standard": 1,
        "brutal": 2,
        "nightmare": 3
    }


@pytest.fixture
def maximum_issues_by_mode():
    """Maximum expected issues by mode"""
    return {
        "standard": 4,
        "brutal": 6,
        "nightmare": 8
    }
