# LLM Works User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [The Arena - Interactive Testing](#the-arena)
3. [The Bench - Standardized Benchmarks](#the-bench)
4. [Dashboard - Results & Analytics](#dashboard)
5. [Settings - Configuration](#settings)
6. [Accessibility Features](#accessibility)
7. [Keyboard Shortcuts](#keyboard-shortcuts)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#frequently-asked-questions)

## Getting Started

### Welcome to LLM Works

LLM Works is an open-source platform for evaluating Large Language Models (LLMs) through interactive testing and rigorous benchmarking. Our platform provides two main evaluation modes:

- **The Arena**: Interactive, comparative testing between models
- **The Bench**: Standardized benchmark suites (MMLU, TruthfulQA, GSM8K)

### First Steps

1. **Access the Platform**: Navigate to [llmworks.dev](https://llmworks.dev)
2. **Configure Models**: Go to Settings → Models to add your API keys
3. **Choose Evaluation Type**: Select Arena for interactive testing or Bench for standardized benchmarks
4. **Review Results**: Check the Dashboard for detailed analytics and insights

### System Requirements

- **Browser**: Modern browser with JavaScript enabled (Chrome 80+, Firefox 75+, Safari 13+, Edge 80+)
- **Internet Connection**: Required for model API calls
- **Local Storage**: ~5MB for preferences and cached data

## The Arena

The Arena provides interactive, head-to-head model comparisons through structured debates, creative challenges, and explanation tasks.

### Arena Modes

#### 1. Debate Mode
Models argue opposing sides of a topic with a judge model evaluating arguments.

**How to Use:**
1. Select 2+ models for comparison
2. Choose or enter a debate topic
3. Configure debate parameters (rounds, time limits)
4. Select a judge model
5. Start the debate and review real-time results

**Best Topics:**
- Ethical dilemmas: "Should AI be regulated?"
- Scientific questions: "Is nuclear energy sustainable?"
- Philosophical debates: "Does free will exist?"

#### 2. Creative Sandbox
Models collaborate on creative tasks with iterative refinement.

**How to Use:**
1. Select models for Creator and Refiner roles
2. Provide a creative brief (story, poem, business plan)
3. Set refinement rounds
4. Review collaborative outputs
5. Compare final results

**Example Tasks:**
- Write a sci-fi short story
- Create a marketing campaign
- Design a mobile app concept

#### 3. Explanation Challenge
Models explain complex topics, optimized for different audiences.

**How to Use:**
1. Choose models for Expert and Student roles
2. Enter a complex topic
3. Set audience level (elementary, high school, college, expert)
4. Configure explanation parameters
5. Evaluate clarity and accuracy

### Arena Results

Results include:
- **Win/Loss Records**: Head-to-head comparison outcomes
- **Elo Ratings**: Dynamic scoring based on performance
- **Category Breakdown**: Performance by reasoning, creativity, factual accuracy
- **Detailed Transcripts**: Full conversation logs with timestamps
- **Judge Explanations**: Reasoning behind decisions

### Tips for Arena Testing

✅ **Do:**
- Use diverse topics to test different capabilities
- Configure appropriate judge models (usually GPT-4 or Claude)
- Test multiple rounds for statistical significance
- Review judge explanations for insights

❌ **Avoid:**
- Single-round comparisons (insufficient data)
- Biased or leading prompts
- Using weak models as judges
- Ignoring category-specific performance

## The Bench

The Bench provides standardized benchmark testing with established academic datasets.

### Available Benchmarks

#### MMLU (Massive Multitask Language Understanding)
- **Subjects**: 57 academic disciplines
- **Questions**: 15,459 multiple choice
- **Time**: ~45 minutes
- **Best for**: General knowledge and reasoning

#### TruthfulQA
- **Focus**: Truthfulness and misinformation resistance
- **Questions**: 817 questions
- **Time**: ~20 minutes
- **Best for**: Factual accuracy and reliability

#### GSM8K
- **Focus**: Grade school math problems
- **Questions**: 1,319 problems
- **Time**: ~30 minutes
- **Best for**: Mathematical reasoning and multi-step problem solving

### Running Benchmarks

1. **Select Benchmark**: Choose from MMLU, TruthfulQA, or GSM8K
2. **Configure Models**: Add 1-4 models for comparison
3. **Customize Settings**:
   - Subject filters (MMLU only)
   - Few-shot examples (0-5)
   - Temperature settings
   - Token limits
4. **Start Evaluation**: Monitor progress in real-time
5. **Review Results**: Access detailed analytics and export options

### Benchmark Results

Results provide:
- **Overall Scores**: Percentage accuracy across all questions
- **Subject Breakdown**: Performance by category/subject
- **Statistical Analysis**: Confidence intervals and significance tests
- **Performance Metrics**: Latency, token usage, cost analysis
- **Comparison Charts**: Side-by-side model comparisons
- **Error Analysis**: Common failure patterns and examples

### Custom Benchmarks

Create your own evaluation suites:

1. **Navigate to Custom Tests**: Bench → Custom Tests tab
2. **Upload Dataset**: JSON/CSV format with questions and answers
3. **Configure Evaluation**:
   - Scoring methodology
   - Multiple choice vs. free form
   - Custom judges for subjective tasks
4. **Run and Analyze**: Same interface as standard benchmarks

## Dashboard

The Dashboard provides comprehensive analytics and monitoring for all evaluations.

### Dashboard Sections

#### Overview Cards
- **Recent Evaluations**: Latest 10 evaluation results
- **Performance Summary**: Win rates and average scores
- **System Status**: API connectivity and health checks
- **Quick Actions**: Start new evaluations, view reports

#### Analytics Charts
- **Performance Trends**: Model performance over time
- **Cost Analysis**: Token usage and API costs
- **Latency Metrics**: Response time distributions
- **Usage Patterns**: Evaluation frequency and types

#### Results Tables
- **Filterable Results**: Sort by date, type, models, scores
- **Export Options**: JSON, CSV, PDF reports
- **Detailed Views**: Drill down into specific evaluations
- **Comparison Tools**: Side-by-side model analysis

### Performance Monitoring

The dashboard includes real-time performance monitoring:

- **Core Web Vitals**: LCP, FID, CLS metrics
- **Service Worker Status**: Offline functionality status
- **Cache Performance**: Hit rates and storage usage
- **Network Connectivity**: Online/offline status

## Settings

### Model Configuration

Add and manage model connections:

1. **Provider Setup**:
   - OpenAI: API key from platform.openai.com
   - Anthropic: API key from console.anthropic.com
   - Google: Vertex AI or Gemini API credentials
   - Custom: Any OpenAI-compatible endpoint

2. **Model Parameters**:
   - Temperature: Creativity vs. consistency (0.0-2.0)
   - Max Tokens: Response length limit
   - Top P: Nucleus sampling parameter
   - Frequency/Presence Penalty: Repetition control

3. **Testing Connection**: Verify API access before evaluation

### System Preferences

#### General Settings
- **Theme**: Light, dark, or system preference
- **Language**: Interface language selection
- **Timezone**: Result timestamps and scheduling
- **Auto-save**: Automatic configuration backup

#### Notifications
- **Evaluation Complete**: Email/browser notifications
- **Model Errors**: API failure alerts
- **Weekly Reports**: Performance summaries
- **System Updates**: Platform announcements

#### Privacy & Data
- **Anonymous Analytics**: Usage data sharing (opt-out available)
- **Data Retention**: How long to keep evaluation results
- **Export/Import**: Backup and restore configurations
- **Account Deletion**: Complete data removal options

## Accessibility Features

LLM Works includes comprehensive accessibility features for users with disabilities.

### Accessibility Toolbar

Click the accessibility icon (♿) in the bottom-right corner to access:

#### Vision Support
- **High Contrast**: Enhanced color contrast for low vision
- **Large Text**: 1.25x to 2x text scaling
- **Text Size Slider**: Fine-grained text control
- **Color Blindness Filters**: Protanopia, deuteranopia, tritanopia simulation

#### Motor Support
- **Reduced Motion**: Disable animations for vestibular disorders
- **Large Cursor**: Normal, large, or extra-large cursor options
- **Enhanced Focus**: Improved keyboard focus indicators

#### Cognitive Support
- **Screen Reader Mode**: Optimized layout for screen readers
- **Sound Feedback**: Audio confirmation for interactions
- **Simplified Interface**: Reduced visual complexity

### Screen Reader Compatibility

Fully compatible with:
- **NVDA** (Windows)
- **JAWS** (Windows)
- **VoiceOver** (macOS/iOS)
- **TalkBack** (Android)
- **Orca** (Linux)

Features include:
- Live region announcements
- Proper heading hierarchy
- Descriptive labels and instructions
- Keyboard navigation support
- Focus management

## Keyboard Shortcuts

### Global Navigation
- `Alt + H`: Home page
- `Alt + A`: Arena evaluation
- `Alt + B`: Bench evaluation  
- `Alt + D`: Dashboard
- `Alt + S`: Settings

### Accessibility
- `Alt + C`: Toggle high contrast
- `Alt + L`: Toggle large text
- `Alt + M`: Toggle reduced motion
- `Alt + R`: Toggle screen reader mode
- `Alt + T`: Toggle accessibility toolbar

### General
- `Alt + K`: View all keyboard shortcuts
- `Tab/Shift+Tab`: Navigate between elements
- `Enter/Space`: Activate buttons and links
- `Escape`: Close modals and dialogs
- `/`: Focus search field (if available)

### Evaluation Controls
- `Ctrl + E`: Start new evaluation
- `Ctrl + P`: Pause/resume current evaluation
- `Ctrl + X`: Stop evaluation
- `Ctrl + V`: View results

### Forms
- `Ctrl + Enter`: Submit forms
- `Ctrl + R`: Reset forms
- `Arrow Keys`: Navigate select/radio options

## Best Practices

### Choosing Models for Evaluation

#### For Arena Testing:
- **Balanced Comparison**: Choose models with similar capabilities
- **Diverse Strengths**: Test models with different architectural approaches
- **Clear Objectives**: Define what you're testing (creativity, reasoning, factual accuracy)

#### For Benchmark Testing:
- **Baseline Models**: Include well-known models for reference
- **Version Consistency**: Use specific model versions for reproducibility
- **Cost Consideration**: Balance evaluation depth with API costs

### Evaluation Design

#### Prompt Engineering:
- **Clear Instructions**: Be specific about task requirements
- **Neutral Phrasing**: Avoid leading or biased language
- **Consistent Format**: Use standardized prompt templates

#### Statistical Validity:
- **Multiple Runs**: Test several times for reliable results
- **Sample Size**: Use sufficient test cases for statistical significance
- **Control Variables**: Keep parameters consistent across models

### Result Interpretation

#### Understanding Scores:
- **Context Matters**: Consider the evaluation context and goals
- **Statistical Significance**: Look for meaningful differences, not just score variations
- **Category Analysis**: Examine performance breakdowns by task type

#### Making Decisions:
- **Multi-metric Evaluation**: Don't rely on single scores
- **Cost-Benefit Analysis**: Consider API costs vs. performance gains
- **Use Case Alignment**: Choose models that match your specific requirements

## Troubleshooting

### Common Issues

#### API Connection Problems
**Problem**: Model requests failing or timing out
**Solutions**:
- Verify API key is correct and active
- Check API quota and billing status
- Test with a simple request in Settings
- Ensure firewall/proxy allows API requests

#### Evaluation Stuck or Slow
**Problem**: Evaluations not progressing or very slow
**Solutions**:
- Check internet connection stability
- Reduce concurrent evaluations in Settings
- Lower temperature for faster, more deterministic responses
- Contact API provider for service status

#### Unexpected Results
**Problem**: Results don't match expectations
**Solutions**:
- Review prompt phrasing for clarity
- Check model parameters (temperature, max tokens)
- Verify judge model configuration
- Run multiple iterations for consistency

#### Interface Issues
**Problem**: UI not loading or responding properly
**Solutions**:
- Clear browser cache and cookies
- Disable browser extensions temporarily
- Try incognito/private browsing mode
- Check browser console for JavaScript errors

### Performance Optimization

#### For Faster Evaluations:
- Use lower temperature settings (0.1-0.3)
- Reduce max token limits when possible
- Limit concurrent evaluations
- Choose faster models for judge roles

#### For Better Accuracy:
- Increase few-shot examples (3-5)
- Use higher quality judge models
- Run multiple evaluation rounds
- Enable detailed error reporting

### Getting Help

1. **Documentation**: Check this guide and API reference
2. **GitHub Issues**: Report bugs at [github.com/llmworks/llm-works](https://github.com/alaweimm90/aegis-ai-evaluator)
3. **Community**: Join discussions and ask questions
4. **Email Support**: Contact team@llmworks.dev for enterprise users

## Frequently Asked Questions

### General Questions

**Q: Is LLM Works free to use?**
A: Yes, LLM Works is open-source and free. You only pay for the API calls to model providers (OpenAI, Anthropic, etc.).

**Q: Do you store my API keys?**
A: API keys are stored locally in your browser and never transmitted to our servers. All model requests go directly to the providers.

**Q: Can I use custom models?**
A: Yes, any OpenAI-compatible API endpoint can be added in Settings → Models.

**Q: How accurate are the evaluations?**
A: Accuracy depends on the judge model and evaluation design. We recommend using GPT-4 or Claude-3 as judges and running multiple iterations.

### Technical Questions

**Q: What browsers are supported?**
A: Modern browsers including Chrome 80+, Firefox 75+, Safari 13+, and Edge 80+.

**Q: Can I run evaluations offline?**
A: No, evaluations require internet connectivity to access model APIs. However, the interface works offline for reviewing cached results.

**Q: How do I export results?**
A: Use the Dashboard → Export button to download results in JSON, CSV, or PDF format.

**Q: Can I integrate LLM Works with my application?**
A: Yes, we provide a REST API and WebSocket interface. See the API Reference for details.

### Billing and Usage

**Q: How much do evaluations cost?**
A: Costs depend on the model providers you use. A typical Arena evaluation costs $0.10-0.50, while benchmark runs cost $2-10 depending on the models and dataset size.

**Q: How can I reduce costs?**
A: Use smaller models for initial testing, reduce max token limits, lower temperature settings, and filter benchmark subsets.

**Q: Do you offer enterprise plans?**
A: We offer enterprise support and on-premises deployment. Contact team@llmworks.dev for details.

### Privacy and Security

**Q: What data do you collect?**
A: We collect anonymous usage analytics (opt-out available) and error reports. No model inputs, outputs, or API keys are transmitted to our servers.

**Q: Is my data secure?**
A: Yes, all sensitive data stays in your browser. We use HTTPS for all connections and follow security best practices.

**Q: Can I delete my data?**
A: Yes, use Settings → Privacy → Delete All Data to remove all local data.

---

For additional help, visit our [GitHub repository](https://github.com/alaweimm90/aegis-ai-evaluator) or contact [team@llmworks.dev](mailto:team@llmworks.dev).