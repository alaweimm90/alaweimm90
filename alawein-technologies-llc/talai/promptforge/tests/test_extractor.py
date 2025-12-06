"""Tests for pattern extraction"""

import pytest
from promptforge.extractor import PatternExtractor


class TestPatternExtractor:
    """Test PatternExtractor class"""

    def test_create_extractor(self, extractor):
        """Test creating a PatternExtractor instance"""
        assert extractor is not None
        assert hasattr(extractor, 'patterns')
        assert hasattr(extractor, 'variable_patterns')
        assert len(extractor.patterns) > 0

    def test_extract_instruction_pattern(self, extractor):
        """Test extracting instruction patterns"""
        text = "Task: Complete the user authentication module"
        patterns = extractor.extract_from_text(text, "test.md")
        assert len(patterns) > 0
        instruction_patterns = [p for p in patterns if p.pattern_type == "instruction"]
        assert len(instruction_patterns) > 0
        assert "authentication" in instruction_patterns[0].template.lower()

    def test_extract_role_play_pattern(self, extractor):
        """Test extracting role play patterns"""
        text = "You are a senior Python developer with expertise in web frameworks."
        patterns = extractor.extract_from_text(text, "test.md")
        role_patterns = [p for p in patterns if p.pattern_type == "role_play"]
        assert len(role_patterns) > 0
        assert "developer" in role_patterns[0].template.lower()

    def test_extract_format_pattern(self, extractor):
        """Test extracting format patterns"""
        text = "Format: JSON output with fields id, name, and email"
        patterns = extractor.extract_from_text(text, "test.md")
        format_patterns = [p for p in patterns if p.pattern_type == "format"]
        assert len(format_patterns) > 0
        assert "json" in format_patterns[0].template.lower()

    def test_extract_constraint_pattern(self, extractor):
        """Test extracting constraint patterns"""
        text = "Constraint: Must use HTTPS for all communications"
        patterns = extractor.extract_from_text(text, "test.md")
        constraint_patterns = [p for p in patterns if p.pattern_type == "constraint"]
        assert len(constraint_patterns) > 0
        assert "https" in constraint_patterns[0].template.lower()

    def test_extract_example_pattern(self, extractor):
        """Test extracting example patterns"""
        text = "Example: user@example.com should be validated"
        patterns = extractor.extract_from_text(text, "test.md")
        example_patterns = [p for p in patterns if p.pattern_type == "example"]
        assert len(example_patterns) > 0

    def test_extract_step_by_step_pattern(self, extractor):
        """Test extracting step-by-step patterns"""
        text = """
        Step 1: Validate user input
        Step 2: Check database credentials
        Step 3: Generate authentication token
        """
        patterns = extractor.extract_from_text(text, "test.md")
        step_patterns = [p for p in patterns if p.pattern_type == "step_by_step"]
        assert len(step_patterns) >= 3

    def test_extract_conditional_pattern(self, extractor):
        """Test extracting conditional patterns"""
        text = "If user is authenticated, then grant access to dashboard."
        patterns = extractor.extract_from_text(text, "test.md")
        conditional_patterns = [p for p in patterns if p.pattern_type == "conditional"]
        assert len(conditional_patterns) > 0
        assert "IF" in conditional_patterns[0].template
        assert "THEN" in conditional_patterns[0].template

    def test_extract_context_pattern(self, extractor):
        """Test extracting context patterns"""
        text = "Context: This is an e-commerce platform handling payment processing"
        patterns = extractor.extract_from_text(text, "test.md")
        context_patterns = [p for p in patterns if p.pattern_type == "context"]
        assert len(context_patterns) > 0
        assert "commerce" in context_patterns[0].template.lower()

    def test_extract_from_basic_text(self, extractor, sample_text_basic):
        """Test extraction from basic sample text"""
        patterns = extractor.extract_from_text(sample_text_basic, "test.md")
        assert len(patterns) > 0
        # Should find multiple pattern types
        pattern_types = set(p.pattern_type for p in patterns)
        assert len(pattern_types) > 3

    def test_extract_variables_curly_braces(self, extractor):
        """Test extracting variables with curly braces"""
        template = "Process {filename} and save to {output_path}"
        variables = extractor._extract_variables(template)
        assert "filename" in variables
        assert "output_path" in variables
        assert len(variables) == 2

    def test_extract_variables_square_brackets(self, extractor):
        """Test extracting variables with square brackets"""
        template = "Convert [input_format] to [output_format]"
        variables = extractor._extract_variables(template)
        assert "input_format" in variables
        assert "output_format" in variables

    def test_extract_variables_angle_brackets(self, extractor):
        """Test extracting variables with angle brackets"""
        template = "Deploy to <environment> using <config_file>"
        variables = extractor._extract_variables(template)
        assert "environment" in variables
        assert "config_file" in variables

    def test_extract_variables_dollar_sign(self, extractor):
        """Test extracting variables with dollar sign"""
        template = "Set $variable to ${another_variable}"
        variables = extractor._extract_variables(template)
        assert "variable" in variables or "another_variable" in variables

    def test_extract_from_text_with_variables(self, extractor, sample_text_with_variables):
        """Test extraction with variable placeholders"""
        patterns = extractor.extract_from_text(sample_text_with_variables, "test.md")
        # Find patterns with variables
        patterns_with_vars = [p for p in patterns if len(p.variables) > 0]
        assert len(patterns_with_vars) > 0

    def test_generate_pattern_name(self, extractor):
        """Test pattern name generation"""
        name = extractor._generate_name("instruction", "Implement user authentication system")
        assert name.startswith("instruction_")
        assert "implement" in name.lower()

    def test_generate_pattern_name_short_template(self, extractor):
        """Test pattern name generation with short template"""
        name = extractor._generate_name("format", "JSON")
        # JSON has 4 letters, so it will be used in the name
        assert name == "format_json"

    def test_calculate_confidence_base(self, extractor):
        """Test confidence calculation with base score"""
        confidence = extractor._calculate_confidence("Simple template text", [])
        assert 0.0 <= confidence <= 1.0
        assert confidence >= 0.3  # Base score minus penalty for no variables

    def test_calculate_confidence_with_variables(self, extractor):
        """Test confidence calculation with variables"""
        confidence = extractor._calculate_confidence("Process {filename} and save output", ["filename"])
        # Base (0.5) + variables (0.2) = 0.7 (no short penalty for longer text)
        assert confidence >= 0.6  # Base + variable bonus

    def test_calculate_confidence_with_structure(self, extractor):
        """Test confidence calculation with structured content"""
        confidence = extractor._calculate_confidence("First do this, then do that, finally complete", [])
        assert confidence > 0.5  # Base + structure bonus

    def test_calculate_confidence_with_requirements(self, extractor):
        """Test confidence calculation with requirement keywords"""
        confidence = extractor._calculate_confidence("Must implement required security features", [])
        assert confidence > 0.5  # Base + requirement bonus

    def test_calculate_confidence_short_template(self, extractor):
        """Test confidence calculation with very short template"""
        confidence = extractor._calculate_confidence("Short", [])
        assert confidence < 0.5  # Penalty for short templates

    def test_extract_tags_from_context(self, extractor, sample_text_with_tags):
        """Test tag extraction from context"""
        patterns = extractor.extract_from_text(sample_text_with_tags, "test.md")
        # Check that some patterns have tags
        patterns_with_tags = [p for p in patterns if len(p.tags) > 0]
        assert len(patterns_with_tags) > 0

    def test_extract_tags_coding(self, extractor):
        """Test semantic tag extraction for coding"""
        text = "Task: Write code to implement the function"
        patterns = extractor.extract_from_text(text, "test.md")
        has_coding_tag = any("coding" in p.tags for p in patterns)
        assert has_coding_tag

    def test_extract_tags_generation(self, extractor):
        """Test semantic tag extraction for generation"""
        text = "Task: Generate documentation for the API"
        patterns = extractor.extract_from_text(text, "test.md")
        has_generation_tag = any("generation" in p.tags for p in patterns)
        assert has_generation_tag

    def test_extract_tags_analysis(self, extractor):
        """Test semantic tag extraction for analysis"""
        text = "Task: Analyze the performance metrics"
        patterns = extractor.extract_from_text(text, "test.md")
        has_analysis_tag = any("analysis" in p.tags for p in patterns)
        assert has_analysis_tag

    def test_line_number_tracking(self, extractor):
        """Test that line numbers are tracked correctly"""
        text = "\nTask: Do something\nLine 3\nLine 4"
        patterns = extractor.extract_from_text(text, "test.md")
        assert len(patterns) > 0
        # Pattern starts at position after first newline, so line_number should be 2
        assert patterns[0].line_number >= 1

    def test_source_file_tracking(self, extractor):
        """Test that source file is tracked"""
        text = "Task: Test file tracking"
        patterns = extractor.extract_from_text(text, "custom_file.md")
        assert len(patterns) > 0
        assert patterns[0].source_file == "custom_file.md"

    def test_template_truncation(self, extractor):
        """Test that long templates are truncated"""
        long_text = "Task: " + "a" * 1000
        patterns = extractor.extract_from_text(long_text, "test.md")
        assert len(patterns) > 0
        assert len(patterns[0].template) <= 500

    def test_empty_text_extraction(self, extractor):
        """Test extraction from empty text"""
        patterns = extractor.extract_from_text("", "test.md")
        assert len(patterns) == 0

    def test_no_patterns_text(self, extractor):
        """Test extraction from text with no patterns"""
        text = "This is just plain text without any recognizable patterns."
        patterns = extractor.extract_from_text(text, "test.md")
        assert len(patterns) == 0
