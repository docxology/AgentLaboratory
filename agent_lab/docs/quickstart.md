# Quick Start Guide

This guide will help you get up and running with Agent Laboratory quickly. It covers the basic steps to set up and run a simple research experiment.

## Prerequisites

Before you begin, make sure you have:

- Installed Agent Laboratory (see the [Installation Guide](./installation.md))
- Configured your API keys (OpenAI, DeepSeek, etc.)

## Running Your First Experiment

### Step 1: Create a Configuration File

Create a file named `my_experiment.yaml` with the following content:

```yaml
# Research configuration
research:
  topic: "Introduction to Transformers in NLP"
  title: "Understanding Transformer Architectures"
  description: "A brief exploration of transformer architectures in natural language processing"
  
# Model configuration
model:
  provider: "openai"
  name: "gpt-4o-mini"
  api_key: "${OPENAI_API_KEY}"  # Will use your environment variable
  temperature: 0.2
  max_tokens: 4096  # Maximum token limit for responses

# Experiment configuration
experiment:
  human_in_loop: false
  compile_latex: true
  max_steps: 50
  phases:
    - literature_review
    - plan_formulation
    - report_writing

# Task notes
notes:
  literature_review:
    - "Focus on the original transformer paper and key follow-up works"
    - "Include practical applications of transformers"
  plan_formulation:
    - "Keep the scope focused on core architecture components"
  report_writing:
    - "Include diagrams of the attention mechanism"
```

### Step 2: Run the Experiment

Run your experiment with the following command:

```bash
agent-lab --config my_experiment.yaml
```

This will start the Agent Laboratory workflow, which will:

1. Run the literature review phase
2. Formulate a research plan
3. Write a final report

### Step 3: Monitor Progress

During execution, you'll see output indicating the progress of the experiment:

```
INFO: Starting research on "Introduction to Transformers in NLP"
INFO: Beginning phase: literature_review
INFO: Agent Professor performing inference...
INFO: Agent PhD Student performing inference...
...
```

If you enabled `human_in_loop: true`, you'll be prompted at certain points to review and provide feedback.

### Step 4: Review Results

When the experiment completes, you'll find the results in the `outputs` directory (by default):

```
outputs/
├── Introduction_to_Transformers_in_NLP/
    ├── logs/
    │   └── agent_logs.txt
    ├── phases/
    │   ├── literature_review/
    │   ├── plan_formulation/
    │   └── report_writing/
    ├── final_report.md
    ├── final_report.pdf
    └── visualization.html
```

## Basic Configuration Options

Here are some key configuration options you can adjust:

### Research Topic

```yaml
research:
  topic: "Your research topic here"
  title: "Optional title for your research"
  description: "Optional longer description"
```

### Model Selection

```yaml
model:
  provider: "openai"  # Options: openai, deepseek, anthropic, gemini
  name: "gpt-4o-mini"  # Model name
  temperature: 0.2  # Creativity level (0.0-1.0)
  max_tokens: 4096  # Maximum token limit for responses
```

### Workflow Phases

```yaml
experiment:
  phases:
    - literature_review
    - plan_formulation
    - data_preparation
    - running_experiments
    - results_interpretation
    - report_writing
```

You can include all phases or select only the ones you need.

### Human Interaction

```yaml
experiment:
  human_in_loop: true  # Enable interactive mode
```

When enabled, you'll be prompted to review agent outputs and provide feedback during the experiment.

## Command-Line Options

Some useful command-line options:

```bash
# Run a specific phase only
agent-lab --config my_experiment.yaml --phase literature_review

# Enable debug mode for verbose output
agent-lab --config my_experiment.yaml --debug

# Specify a custom output directory
agent-lab --config my_experiment.yaml --output-dir ./my_results
```

## Next Steps

- Explore the [Configuration Guide](./configuration.md) for detailed configuration options
- See the [Example Workflow](./example_workflow.md) for a complete walkthrough
- Learn about the [Architecture](./architecture.md) to understand how Agent Laboratory works 