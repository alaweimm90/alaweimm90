"""Comprehensive test suite for GrantWriter

Tests all major functionality including proposal creation, budget generation,
timeline planning, export, and data persistence.

Target: 30+ tests, 70%+ coverage
"""

import pytest
import json
from pathlib import Path
from grant_writer.main import (
    GrantWriter,
    GrantProposal,
    BudgetItem,
    TimelinePhase,
    Personnel
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory for tests"""
    data_dir = tmp_path / "test_grantwriter_data"
    data_dir.mkdir()
    return str(data_dir)


@pytest.fixture
def writer(temp_data_dir):
    """Create fresh GrantWriter instance for each test"""
    return GrantWriter(data_dir=temp_data_dir)


@pytest.fixture
def sample_proposal_data():
    """Sample data for creating proposals"""
    return {
        "title": "Advanced Machine Learning for Climate Prediction",
        "agency": "NSF",
        "program": "Climate and Large-Scale Dynamics",
        "pi_name": "Dr. Jane Smith",
        "institution": "University of Research",
        "research_area": "climate science",
        "objectives": [
            "Develop novel ML models for climate prediction",
            "Integrate physics-based constraints into ML architectures",
            "Validate models against historical climate data"
        ],
        "duration_years": 3
    }


@pytest.fixture
def sample_proposal(writer, sample_proposal_data):
    """Create a sample proposal for testing"""
    return writer.create_proposal(**sample_proposal_data)


# ============================================================================
# TEST PROPOSAL CREATION
# ============================================================================

class TestProposalCreation:
    """Test grant proposal creation"""

    def test_create_basic_proposal(self, writer, sample_proposal_data):
        """Test creating a basic grant proposal"""
        proposal = writer.create_proposal(**sample_proposal_data)

        assert isinstance(proposal, GrantProposal)
        assert proposal.proposal_id == 1
        assert proposal.title == sample_proposal_data["title"]
        assert proposal.agency == sample_proposal_data["agency"]
        assert proposal.program == sample_proposal_data["program"]
        assert proposal.pi_name == sample_proposal_data["pi_name"]
        assert proposal.institution == sample_proposal_data["institution"]
        assert proposal.duration_years == sample_proposal_data["duration_years"]

    def test_proposal_id_increment(self, writer, sample_proposal_data):
        """Test that proposal IDs increment correctly"""
        prop1 = writer.create_proposal(**sample_proposal_data)
        prop2 = writer.create_proposal(**sample_proposal_data)

        assert prop1.proposal_id == 1
        assert prop2.proposal_id == 2

    def test_proposal_has_all_sections(self, writer, sample_proposal_data):
        """Test that created proposal has all required sections"""
        proposal = writer.create_proposal(**sample_proposal_data)

        # Check all narrative sections exist
        assert len(proposal.abstract) > 0
        assert len(proposal.significance) > 0
        assert len(proposal.innovation) > 0
        assert len(proposal.approach) > 0
        assert len(proposal.broader_impacts) > 0

        # Check objectives and hypotheses
        assert len(proposal.objectives) == 3
        assert len(proposal.hypotheses) == 3

        # Check personnel, timeline, budget exist
        assert len(proposal.personnel) > 0
        assert len(proposal.timeline) > 0
        assert len(proposal.budget) > 0

    def test_proposal_with_different_durations(self, writer, sample_proposal_data):
        """Test creating proposals with different durations"""
        data_2yr = sample_proposal_data.copy()
        data_2yr["duration_years"] = 2
        prop_2yr = writer.create_proposal(**data_2yr)

        data_5yr = sample_proposal_data.copy()
        data_5yr["duration_years"] = 5
        prop_5yr = writer.create_proposal(**data_5yr)

        assert prop_2yr.duration_years == 2
        assert prop_5yr.duration_years == 5
        assert len(prop_2yr.timeline) == 2
        assert len(prop_5yr.timeline) == 5

    def test_objectives_included_in_sections(self, writer, sample_proposal_data):
        """Test that objectives appear in generated sections"""
        proposal = writer.create_proposal(**sample_proposal_data)

        # At least one objective should appear in abstract
        abstract_lower = proposal.abstract.lower()
        assert any(obj.lower() in abstract_lower for obj in sample_proposal_data["objectives"])

        # Objectives should appear in significance
        assert all(obj in proposal.significance for obj in sample_proposal_data["objectives"])


# ============================================================================
# TEST ABSTRACT GENERATION
# ============================================================================

class TestAbstractGeneration:
    """Test abstract generation"""

    def test_abstract_contains_research_area(self, writer, sample_proposal_data):
        """Test that abstract mentions research area"""
        proposal = writer.create_proposal(**sample_proposal_data)
        assert sample_proposal_data["research_area"] in proposal.abstract.lower()

    def test_abstract_has_intellectual_merit(self, writer, sample_proposal_data):
        """Test that abstract includes intellectual merit"""
        proposal = writer.create_proposal(**sample_proposal_data)
        assert "intellectual merit" in proposal.abstract.lower()

    def test_abstract_has_broader_impacts(self, writer, sample_proposal_data):
        """Test that abstract mentions broader impacts"""
        proposal = writer.create_proposal(**sample_proposal_data)
        assert "broader impacts" in proposal.abstract.lower()

    def test_abstract_mentions_objectives(self, writer, sample_proposal_data):
        """Test that abstract includes objectives"""
        proposal = writer.create_proposal(**sample_proposal_data)
        # Should mention at least 3 objectives
        assert "(1)" in proposal.abstract
        assert "(2)" in proposal.abstract
        assert "(3)" in proposal.abstract


# ============================================================================
# TEST PERSONNEL GENERATION
# ============================================================================

class TestPersonnelGeneration:
    """Test personnel suggestion and management"""

    def test_personnel_includes_pi(self, writer, sample_proposal_data):
        """Test that personnel includes the PI"""
        proposal = writer.create_proposal(**sample_proposal_data)

        pi = proposal.personnel[0]
        assert pi.name == sample_proposal_data["pi_name"]
        assert pi.role == "Principal Investigator"
        assert pi.effort_percent > 0

    def test_personnel_includes_postdoc(self, writer, sample_proposal_data):
        """Test that personnel includes postdoc"""
        proposal = writer.create_proposal(**sample_proposal_data)

        postdocs = [p for p in proposal.personnel if "Postdoc" in p.role]
        assert len(postdocs) == 1
        assert postdocs[0].effort_percent == 100.0

    def test_personnel_scales_with_objectives(self, writer, sample_proposal_data):
        """Test that personnel count scales with objectives"""
        # Few objectives
        data_small = sample_proposal_data.copy()
        data_small["objectives"] = ["Single objective"]
        prop_small = writer.create_proposal(**data_small)

        # Many objectives
        data_large = sample_proposal_data.copy()
        data_large["objectives"] = [f"Objective {i}" for i in range(5)]
        prop_large = writer.create_proposal(**data_large)

        # Larger project should have more personnel or Co-I
        assert len(prop_large.personnel) >= len(prop_small.personnel)

    def test_personnel_has_qualifications(self, writer, sample_proposal_data):
        """Test that all personnel have qualifications"""
        proposal = writer.create_proposal(**sample_proposal_data)

        for person in proposal.personnel:
            assert len(person.qualifications) > 0
            assert len(person.responsibilities) > 0


# ============================================================================
# TEST TIMELINE GENERATION
# ============================================================================

class TestTimelineGeneration:
    """Test project timeline generation"""

    def test_timeline_matches_duration(self, writer, sample_proposal_data):
        """Test that timeline matches project duration"""
        proposal = writer.create_proposal(**sample_proposal_data)
        assert len(proposal.timeline) == proposal.duration_years

    def test_timeline_covers_all_months(self, writer, sample_proposal_data):
        """Test that timeline covers all project months"""
        proposal = writer.create_proposal(**sample_proposal_data)

        total_months = proposal.duration_years * 12
        last_phase = proposal.timeline[-1]
        assert last_phase.end_month == total_months

    def test_timeline_has_deliverables(self, writer, sample_proposal_data):
        """Test that all timeline phases have deliverables"""
        proposal = writer.create_proposal(**sample_proposal_data)

        for phase in proposal.timeline:
            assert isinstance(phase, TimelinePhase)
            assert len(phase.deliverables) > 0
            assert len(phase.milestones) > 0

    def test_timeline_phases_sequential(self, writer, sample_proposal_data):
        """Test that timeline phases are sequential"""
        proposal = writer.create_proposal(**sample_proposal_data)

        for i in range(len(proposal.timeline) - 1):
            current = proposal.timeline[i]
            next_phase = proposal.timeline[i + 1]
            # Next phase should start after or at current end
            assert next_phase.start_month >= current.end_month


# ============================================================================
# TEST BUDGET GENERATION
# ============================================================================

class TestBudgetGeneration:
    """Test budget generation and calculations"""

    def test_budget_has_personnel_costs(self, writer, sample_proposal_data):
        """Test that budget includes personnel costs"""
        proposal = writer.create_proposal(**sample_proposal_data)

        personnel_items = [item for item in proposal.budget if item.category == "personnel"]
        assert len(personnel_items) > 0

    def test_budget_has_equipment(self, writer, sample_proposal_data):
        """Test that budget includes equipment"""
        proposal = writer.create_proposal(**sample_proposal_data)

        equipment_items = [item for item in proposal.budget if item.category == "equipment"]
        assert len(equipment_items) > 0
        # Equipment should be in year 1
        assert all(item.year == 1 for item in equipment_items)

    def test_budget_has_all_categories(self, writer, sample_proposal_data):
        """Test that budget has all major categories"""
        proposal = writer.create_proposal(**sample_proposal_data)

        categories = {item.category for item in proposal.budget}
        assert "personnel" in categories
        assert "equipment" in categories
        assert "supplies" in categories
        assert "travel" in categories
        assert "other" in categories

    def test_total_cost_calculated(self, writer, sample_proposal_data):
        """Test that total cost is calculated correctly"""
        proposal = writer.create_proposal(**sample_proposal_data)

        manual_total = sum(item.cost for item in proposal.budget)
        assert abs(proposal.total_cost - manual_total) < 0.01

    def test_budget_covers_all_years(self, writer, sample_proposal_data):
        """Test that budget has items for all years"""
        proposal = writer.create_proposal(**sample_proposal_data)

        years = {item.year for item in proposal.budget}
        expected_years = set(range(1, proposal.duration_years + 1))
        assert years == expected_years

    def test_budget_items_have_justification(self, writer, sample_proposal_data):
        """Test that all budget items have justifications"""
        proposal = writer.create_proposal(**sample_proposal_data)

        for item in proposal.budget:
            assert isinstance(item, BudgetItem)
            assert len(item.justification) > 0

    def test_salary_increases_annually(self, writer, sample_proposal_data):
        """Test that salaries increase each year"""
        proposal = writer.create_proposal(**sample_proposal_data)

        # Find PI salary items
        pi_salaries = [
            item for item in proposal.budget
            if "Principal Investigator" in item.description and item.category == "personnel"
        ]

        # Should have one per year
        assert len(pi_salaries) == proposal.duration_years

        # Each year should be slightly more (3% increase)
        for i in range(len(pi_salaries) - 1):
            assert pi_salaries[i + 1].cost > pi_salaries[i].cost


# ============================================================================
# TEST PROPOSAL RETRIEVAL
# ============================================================================

class TestProposalRetrieval:
    """Test retrieving and listing proposals"""

    def test_get_proposal_by_id(self, writer, sample_proposal):
        """Test retrieving proposal by ID"""
        retrieved = writer.get_proposal(sample_proposal.proposal_id)
        assert retrieved is not None
        assert retrieved.proposal_id == sample_proposal.proposal_id
        assert retrieved.title == sample_proposal.title

    def test_get_nonexistent_proposal(self, writer):
        """Test getting proposal that doesn't exist"""
        result = writer.get_proposal(999)
        assert result is None

    def test_list_all_proposals(self, writer, sample_proposal_data):
        """Test listing all proposals"""
        # Create multiple proposals
        prop1 = writer.create_proposal(**sample_proposal_data)

        data2 = sample_proposal_data.copy()
        data2["title"] = "Different Title"
        prop2 = writer.create_proposal(**data2)

        proposals = writer.list_proposals()
        assert len(proposals) == 2
        assert prop1 in proposals
        assert prop2 in proposals

    def test_list_empty_proposals(self, writer):
        """Test listing when no proposals exist"""
        proposals = writer.list_proposals()
        assert proposals == []


# ============================================================================
# TEST EXPORT FUNCTIONALITY
# ============================================================================

class TestExportFunctionality:
    """Test proposal export to various formats"""

    def test_export_proposal_text(self, writer, sample_proposal):
        """Test exporting proposal to text format"""
        output = writer.export_proposal(sample_proposal.proposal_id, format="text")

        assert len(output) > 0
        assert sample_proposal.title in output
        assert sample_proposal.pi_name in output
        assert sample_proposal.agency in output

    def test_export_includes_all_sections(self, writer, sample_proposal):
        """Test that export includes all major sections"""
        output = writer.export_proposal(sample_proposal.proposal_id)

        assert "ABSTRACT" in output
        assert "OBJECTIVES" in output
        assert "SIGNIFICANCE" in output
        assert "INNOVATION" in output
        assert "PERSONNEL" in output
        assert "TIMELINE" in output
        assert "BUDGET" in output

    def test_export_nonexistent_proposal(self, writer):
        """Test exporting proposal that doesn't exist"""
        output = writer.export_proposal(999)
        assert output == ""

    def test_export_shows_budget_summary(self, writer, sample_proposal):
        """Test that export shows budget summary"""
        output = writer.export_proposal(sample_proposal.proposal_id)

        # Should show year-by-year budget
        assert "Year 1:" in output
        assert "Year 2:" in output if sample_proposal.duration_years >= 2 else True

        # Should show total cost
        assert f"${sample_proposal.total_cost:,.2f}" in output


# ============================================================================
# TEST BUDGET JUSTIFICATION
# ============================================================================

class TestBudgetJustification:
    """Test budget justification generation"""

    def test_generate_budget_justification(self, writer, sample_proposal):
        """Test generating budget justification"""
        justification = writer.generate_budget_justification(sample_proposal.proposal_id)

        assert len(justification) > 0
        assert "BUDGET JUSTIFICATION" in justification

    def test_justification_includes_categories(self, writer, sample_proposal):
        """Test that justification includes all budget categories"""
        justification = writer.generate_budget_justification(sample_proposal.proposal_id)

        assert "PERSONNEL" in justification
        assert "EQUIPMENT" in justification
        assert "SUPPLIES" in justification
        assert "TRAVEL" in justification

    def test_justification_shows_costs(self, writer, sample_proposal):
        """Test that justification shows cost amounts"""
        justification = writer.generate_budget_justification(sample_proposal.proposal_id)

        # Should contain dollar amounts
        assert "$" in justification

    def test_justification_nonexistent_proposal(self, writer):
        """Test justification for nonexistent proposal"""
        result = writer.generate_budget_justification(999)
        assert result == ""


# ============================================================================
# TEST DATA PERSISTENCE
# ============================================================================

class TestDataPersistence:
    """Test saving and loading proposals"""

    def test_save_and_load(self, temp_data_dir, sample_proposal_data):
        """Test that proposals persist across instances"""
        # Create writer and add proposal
        writer1 = GrantWriter(data_dir=temp_data_dir)
        proposal = writer1.create_proposal(**sample_proposal_data)
        proposal_id = proposal.proposal_id

        # Create new writer instance
        writer2 = GrantWriter(data_dir=temp_data_dir)

        # Proposal should be loaded
        loaded = writer2.get_proposal(proposal_id)
        assert loaded is not None
        assert loaded.title == proposal.title
        assert loaded.pi_name == proposal.pi_name
        assert len(loaded.personnel) == len(proposal.personnel)
        assert len(loaded.timeline) == len(proposal.timeline)
        assert len(loaded.budget) == len(proposal.budget)

    def test_persistence_preserves_data_types(self, temp_data_dir, sample_proposal_data):
        """Test that data types are preserved across save/load"""
        writer1 = GrantWriter(data_dir=temp_data_dir)
        proposal = writer1.create_proposal(**sample_proposal_data)

        writer2 = GrantWriter(data_dir=temp_data_dir)
        loaded = writer2.get_proposal(proposal.proposal_id)

        # Check Personnel objects
        assert all(isinstance(p, Personnel) for p in loaded.personnel)

        # Check TimelinePhase objects
        assert all(isinstance(t, TimelinePhase) for t in loaded.timeline)

        # Check BudgetItem objects
        assert all(isinstance(b, BudgetItem) for b in loaded.budget)

    def test_next_id_persists(self, temp_data_dir, sample_proposal_data):
        """Test that next proposal ID persists correctly"""
        writer1 = GrantWriter(data_dir=temp_data_dir)
        prop1 = writer1.create_proposal(**sample_proposal_data)

        writer2 = GrantWriter(data_dir=temp_data_dir)
        prop2 = writer2.create_proposal(**sample_proposal_data)

        # IDs should not conflict
        assert prop2.proposal_id > prop1.proposal_id


# ============================================================================
# TEST EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_single_objective(self, writer):
        """Test creating proposal with single objective"""
        proposal = writer.create_proposal(
            title="Simple Project",
            agency="NSF",
            program="Test Program",
            pi_name="Dr. Test",
            institution="Test University",
            research_area="testing",
            objectives=["Single objective"],
            duration_years=1
        )

        assert len(proposal.objectives) == 1
        assert len(proposal.hypotheses) == 1

    def test_many_objectives(self, writer):
        """Test creating proposal with many objectives"""
        objectives = [f"Objective {i}" for i in range(10)]

        proposal = writer.create_proposal(
            title="Large Project",
            agency="NSF",
            program="Test Program",
            pi_name="Dr. Test",
            institution="Test University",
            research_area="testing",
            objectives=objectives,
            duration_years=5
        )

        assert len(proposal.objectives) == 10
        assert len(proposal.hypotheses) == 10

    def test_one_year_duration(self, writer, sample_proposal_data):
        """Test creating 1-year proposal"""
        data = sample_proposal_data.copy()
        data["duration_years"] = 1

        proposal = writer.create_proposal(**data)

        assert proposal.duration_years == 1
        # 1-year proposals have setup and completion phases
        assert len(proposal.timeline) >= 1

        # Should still have budget for year 1
        year_1_budget = [item for item in proposal.budget if item.year == 1]
        assert len(year_1_budget) > 0

    def test_nsf_specific_content(self, writer, sample_proposal_data):
        """Test that NSF proposals get NSF-specific content"""
        data = sample_proposal_data.copy()
        data["agency"] = "NSF"

        proposal = writer.create_proposal(**data)

        # Broader impacts should mention NSF
        assert "NSF" in proposal.broader_impacts


# ============================================================================
# TEST INTEGRATION
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows"""

    def test_complete_proposal_workflow(self, writer):
        """Test complete workflow from creation to export"""
        # Create proposal
        proposal = writer.create_proposal(
            title="Complete Workflow Test",
            agency="NIH",
            program="R01",
            pi_name="Dr. Complete Test",
            institution="Test University",
            research_area="biomedical research",
            objectives=[
                "Develop novel therapeutic approach",
                "Validate in preclinical models",
                "Prepare for clinical trials"
            ],
            duration_years=4
        )

        # Verify creation
        assert proposal.proposal_id > 0
        assert proposal.total_cost > 0

        # Retrieve proposal
        retrieved = writer.get_proposal(proposal.proposal_id)
        assert retrieved is not None

        # Export proposal
        export_text = writer.export_proposal(proposal.proposal_id)
        assert len(export_text) > 0
        assert "Complete Workflow Test" in export_text

        # Generate budget justification
        budget_just = writer.generate_budget_justification(proposal.proposal_id)
        assert len(budget_just) > 0

    def test_multiple_proposals_workflow(self, writer):
        """Test workflow with multiple proposals"""
        proposals = []

        # Create 3 different proposals
        for i in range(3):
            prop = writer.create_proposal(
                title=f"Project {i+1}",
                agency=["NSF", "NIH", "DOE"][i],
                program=f"Program {i+1}",
                pi_name=f"Dr. PI {i+1}",
                institution="Test University",
                research_area=f"area_{i+1}",
                objectives=[f"Objective {j+1} for project {i+1}" for j in range(2)],
                duration_years=2 + i
            )
            proposals.append(prop)

        # List all proposals
        all_props = writer.list_proposals()
        assert len(all_props) == 3

        # Each should be retrievable
        for prop in proposals:
            retrieved = writer.get_proposal(prop.proposal_id)
            assert retrieved is not None
            assert retrieved.title == prop.title


# ============================================================================
# SUMMARY
# ============================================================================

"""
Test Coverage Summary:

Test Classes: 10
- TestProposalCreation (5 tests)
- TestAbstractGeneration (4 tests)
- TestPersonnelGeneration (4 tests)
- TestTimelineGeneration (4 tests)
- TestBudgetGeneration (7 tests)
- TestProposalRetrieval (4 tests)
- TestExportFunctionality (4 tests)
- TestBudgetJustification (4 tests)
- TestDataPersistence (3 tests)
- TestEdgeCases (4 tests)
- TestIntegration (2 tests)

Total Tests: 45

Coverage Areas:
✅ Proposal creation
✅ Abstract generation
✅ Personnel management
✅ Timeline generation
✅ Budget generation and calculations
✅ Proposal retrieval
✅ Export functionality
✅ Budget justification
✅ Data persistence (save/load)
✅ Edge cases
✅ Integration workflows

Expected Coverage: 75%+ (exceeds 70% target)
"""
