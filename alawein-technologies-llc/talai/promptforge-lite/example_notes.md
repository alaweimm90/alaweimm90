# Prompt Engineering Notes

## Code Analysis Prompts

Instruction: Analyze the following {language} code for {issue_type} issues and provide detailed recommendations.

Example: e.g., Python code for security vulnerabilities

Constraint: Must include severity ratings (CRITICAL, HIGH, MEDIUM, LOW)

Format: Output should be structured as JSON with fields for issue, severity, line_number, and recommendation.

## Role-Based Prompts

Role: You are an expert software architect with 15 years of experience in distributed systems.

Task: Review the system design and identify scalability bottlenecks.

Context: The system currently handles 10K requests per second but needs to scale to 100K RPS.

## Step-by-Step Instructions

Step 1: Parse the input code and build an abstract syntax tree
Step 2: Identify all external dependencies and API calls
Step 3: Check for common security vulnerabilities (SQL injection, XSS, etc.)
Step 4: Generate a security report with prioritized findings

## Conditional Patterns

If the code contains database queries, then check for SQL injection vulnerabilities.

When analyzing web applications, ensure CSRF protection is implemented.

Given user-submitted content, verify proper input sanitization and output encoding.

## Writing Prompts

Goal: Generate a technical blog post about {topic} that is {length} words long.

Requirements:
- Must include code examples
- Should target intermediate developers
- Must cite at least 3 authoritative sources

Structure:
1. Introduction with hook
2. Problem statement
3. Solution explanation
4. Code examples
5. Conclusion with call-to-action

#writing #technical #blogging

## Data Analysis

Objective: Analyze the dataset <dataset_path> and generate insights about ${metric_name}.

Format:
- Summary statistics (mean, median, std dev)
- Visualizations (histogram, scatter plot)
- Outlier detection results
- Correlation analysis

Tags: #data-science #analysis #statistics
