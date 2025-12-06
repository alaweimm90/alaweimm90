"""
Test suite for PatternExtractor class
"""
import pytest
from promptforge_lite.main import PatternExtractor, PromptPattern


class TestPatternExtractorInit:
    """Test PatternExtractor initialization"""

    def test_extractor_init(self, extractor):
        """Test that extractor initializes with correct patterns"""
        assert extractor is not None
        assert "instruction" in extractor.patterns
        assert "role_play" in extractor.patterns
        assert "format" in extractor.patterns
        assert "constraint" in extractor.patterns
        assert "example" in extractor.patterns
        assert "step_by_step" in extractor.patterns
        assert "conditional" in extractor.patterns
        assert "context" in extractor.patterns

    def test_variable_patterns_exist(self, extractor):
        """Test that variable patterns are defined"""
        assert len(extractor.variable_patterns) == 4


class TestExtractFromText:
    """Test extract_from_text method"""

    def test_extract_instruction_pattern(self, extractor, instruction_text):
        """Test extraction of instruction patterns"""
        patterns = extractor.extract_from_text(instruction_text, "test.md")
        assert len(patterns) > 0
        instruction_patterns = [p for p in patterns if p.pattern_type == "instruction"]
        assert len(instruction_patterns) > 0
        assert instruction_patterns[0].source_file == "test.md"

    def test_extract_role_play_pattern(self, extractor, role_play_text):
        """Test extraction of role play patterns"""
        patterns = extractor.extract_from_text(role_play_text, "test.md")
        role_patterns = [p for p in patterns if p.pattern_type == "role_play"]
        assert len(role_patterns) > 0
        assert "architect" in role_patterns[0].template.lower()

    def test_extract_format_pattern(self, extractor, format_text):
        """Test extraction of format patterns"""
        patterns = extractor.extract_from_text(format_text, "test.md")
        format_patterns = [p for p in patterns if p.pattern_type == "format"]
        assert len(format_patterns) > 0

    def test_extract_constraint_pattern(self, extractor, constraint_text):
        """Test extraction of constraint patterns"""
        patterns = extractor.extract_from_text(constraint_text, "test.md")
        constraint_patterns = [p for p in patterns if p.pattern_type == "constraint"]
        assert len(constraint_patterns) > 0

    def test_extract_step_by_step_pattern(self, extractor, step_by_step_text):
        """Test extraction of step by step patterns"""
        patterns = extractor.extract_from_text(step_by_step_text, "test.md")
        step_patterns = [p for p in patterns if p.pattern_type == "step_by_step"]
        assert len(step_patterns) >= 3  # Should find multiple steps

    def test_extract_conditional_pattern(self, extractor, conditional_text):
        """Test extraction of conditional patterns"""
        patterns = extractor.extract_from_text(conditional_text, "test.md")
        conditional_patterns = [p for p in patterns if p.pattern_type == "conditional"]
        assert len(conditional_patterns) > 0
        # Check IF...THEN format
        assert "IF" in conditional_patterns[0].template
        assert "THEN" in conditional_patterns[0].template

    def test_extract_context_pattern(self, extractor, context_text):
        """Test extraction of context patterns"""
        patterns = extractor.extract_from_text(context_text, "test.md")
        context_patterns = [p for p in patterns if p.pattern_type == "context"]
        assert len(context_patterns) > 0

    def test_extract_multiple_patterns(self, extractor, multi_pattern_text):
        """Test extraction of multiple pattern types from one text"""
        patterns = extractor.extract_from_text(multi_pattern_text, "test.md")
        assert len(patterns) >= 5  # Should find multiple different patterns

        pattern_types = set(p.pattern_type for p in patterns)
        assert "instruction" in pattern_types or "step_by_step" in pattern_types
        assert len(pattern_types) >= 3  # At least 3 different pattern types

    def test_extract_with_empty_text(self, extractor):
        """Test extraction from empty text"""
        patterns = extractor.extract_from_text("", "test.md")
        assert len(patterns) == 0

    def test_line_number_extraction(self, extractor, instruction_text):
        """Test that line numbers are correctly extracted"""
        patterns = extractor.extract_from_text(instruction_text, "test.md")
        assert all(p.line_number > 0 for p in patterns)

    def test_template_truncation(self, extractor, long_template_text):
        """Test that long templates are truncated to 500 chars"""
        patterns = extractor.extract_from_text(long_template_text, "test.md")
        for pattern in patterns:
            assert len(pattern.template) <= 500


class TestExtractVariables:
    """Test _extract_variables method"""

    def test_extract_curly_braces(self, extractor):
        """Test extraction of {variable} format"""
        template = "Process {input_file} and save to {output_file}"
        variables = extractor._extract_variables(template)
        assert "input_file" in variables
        assert "output_file" in variables

    def test_extract_square_brackets(self, extractor):
        """Test extraction of [variable] format"""
        template = "Use [config] with [settings]"
        variables = extractor._extract_variables(template)
        assert "config" in variables
        assert "settings" in variables

    def test_extract_angle_brackets(self, extractor):
        """Test extraction of <variable> format"""
        template = "Apply <template> to <data>"
        variables = extractor._extract_variables(template)
        assert "template" in variables
        assert "data" in variables

    def test_extract_dollar_sign(self, extractor):
        """Test extraction of $variable and ${variable} format"""
        template = "Use $config and ${environment} settings"
        variables = extractor._extract_variables(template)
        assert "config" in variables
        assert "environment" in variables

    def test_extract_mixed_variables(self, extractor, variables_text):
        """Test extraction of multiple variable formats"""
        variables = extractor._extract_variables(variables_text)
        assert len(variables) >= 4
        assert sorted(variables) == variables  # Should be sorted

    def test_no_variables(self, extractor):
        """Test text with no variables"""
        template = "This is a simple template with no variables"
        variables = extractor._extract_variables(template)
        assert len(variables) == 0


class TestGenerateName:
    """Test _generate_name method"""

    def test_generate_name_with_words(self, extractor):
        """Test name generation with valid words"""
        template = "Analyze the code quality metrics"
        name = extractor._generate_name("instruction", template)
        assert name.startswith("instruction_")
        assert "analyze" in name.lower()

    def test_generate_name_fallback(self, extractor):
        """Test name generation fallback for short text"""
        template = "Do it"
        name = extractor._generate_name("instruction", template)
        assert name == "instruction_pattern"

    def test_generate_name_multiple_words(self, extractor):
        """Test that name uses first 3 significant words"""
        template = "Create comprehensive detailed thorough extensive analysis report"
        name = extractor._generate_name("instruction", template)
        word_count = len(name.split("_")) - 1  # Subtract pattern type
        assert word_count <= 3


class TestCalculateConfidence:
    """Test _calculate_confidence method"""

    def test_base_confidence(self, extractor):
        """Test base confidence score"""
        template = "This is a simple template"
        confidence = extractor._calculate_confidence(template, [])
        assert 0.0 <= confidence <= 1.0

    def test_confidence_with_variables(self, extractor):
        """Test confidence boost with variables"""
        template = "Process {input} and generate {output}"
        variables = ["input", "output"]
        confidence = extractor._calculate_confidence(template, variables)
        confidence_no_vars = extractor._calculate_confidence(template, [])
        assert confidence > confidence_no_vars

    def test_confidence_with_structured_keywords(self, extractor):
        """Test confidence boost with structured keywords"""
        template = "First, analyze the data, then generate report, finally validate"
        confidence = extractor._calculate_confidence(template, [])
        assert confidence >= 0.6

    def test_confidence_with_instruction_keywords(self, extractor):
        """Test confidence boost with instruction keywords"""
        template = "You must validate the input and should check for errors"
        confidence = extractor._calculate_confidence(template, [])
        assert confidence >= 0.6

    def test_confidence_penalty_short(self, extractor):
        """Test confidence penalty for very short templates"""
        template = "Do this"
        confidence = extractor._calculate_confidence(template, [])
        assert confidence < 0.5

    def test_confidence_clamping(self, extractor):
        """Test that confidence is clamped between 0 and 1"""
        template = "Must should required first then finally step" + " word" * 20
        variables = ["var1", "var2", "var3"]
        confidence = extractor._calculate_confidence(template, variables)
        assert 0.0 <= confidence <= 1.0


class TestExtractTags:
    """Test _extract_tags method"""

    def test_extract_hashtags(self, extractor, tagged_text):
        """Test extraction of hashtags"""
        patterns = extractor.extract_from_text(tagged_text, "test.md")
        assert len(patterns) > 0
        # Check that some tags were extracted
        has_tags = any(len(p.tags) > 0 for p in patterns)
        assert has_tags

    def test_semantic_tag_coding(self, extractor):
        """Test semantic tag for coding content"""
        text = "Task: Write a code function for processing"
        tags = extractor._extract_tags(text, 0, len(text))
        assert "coding" in tags

    def test_semantic_tag_generation(self, extractor):
        """Test semantic tag for generation content"""
        text = "Task: Generate a report and write documentation"
        tags = extractor._extract_tags(text, 0, len(text))
        assert "generation" in tags

    def test_semantic_tag_analysis(self, extractor):
        """Test semantic tag for analysis content"""
        text = "Task: Analyze the system and review the results"
        tags = extractor._extract_tags(text, 0, len(text))
        assert "analysis" in tags

    def test_tags_limit(self, extractor):
        """Test that tags are limited to 5"""
        text = "Task: Test #tag1 #tag2 #tag3 #tag4 #tag5 #tag6 #tag7 #tag8"
        tags = extractor._extract_tags(text, 0, len(text))
        assert len(tags) <= 5

    def test_tags_sorted(self, extractor):
        """Test that tags are sorted"""
        text = "Task: Process #zebra #alpha #mike #bravo"
        tags = extractor._extract_tags(text, 0, len(text))
        assert tags == sorted(tags)


class TestExtractExample:
    """Test _extract_example method"""

    def test_extract_explicit_example(self, extractor):
        """Test extraction of explicit example with e.g."""
        text = "Task: Do something\n\ne.g. like this example here"
        example = extractor._extract_example(text, 0, 20)
        assert "like this" in example.lower() or "task" in example.lower()

    def test_extract_example_fallback(self, extractor):
        """Test fallback to match content"""
        text = "Task: Create a comprehensive test suite"
        example = extractor._extract_example(text, 0, len(text))
        assert len(example) > 0

    def test_example_truncation(self, extractor):
        """Test that examples are truncated to 200 chars"""
        long_text = "Task: " + "A" * 300
        patterns = extractor.extract_from_text(long_text, "test.md")
        for pattern in patterns:
            assert len(pattern.example) <= 200
