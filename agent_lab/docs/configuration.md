# Configuration Guide

This document describes the configuration options for the Agent Laboratory.

## Configuration File Format

Agent Laboratory uses YAML files for configuration. The configuration file is divided into several sections:

- `research`: Research-related configuration
- `model`: LLM model configuration
- `experiment`: Experiment settings
- `notes`: Phase-specific notes for agents
- `output`: Output and reporting configuration

## Full Configuration Example

```yaml
# Research configuration
research:
  topic: "POMDP implementation for Active Inference"
  title: "Implementing POMDP Models for Active Inference"
  description: "This experiment explores how to implement POMDP models for agent-based Active Inference"
  
# Model configuration
model:
  provider: "openai"  # Options: openai, deepseek, anthropic, gemini
  name: "gpt-4o-mini"
  api_key: "${OPENAI_API_KEY}"  # Environment variable reference
  deepseek_api_key: "${DEEPSEEK_API_KEY}"  # Optional
  anthropic_api_key: "${ANTHROPIC_API_KEY}"  # Optional
  gemini_api_key: "${GEMINI_API_KEY}"  # Optional
  temperature: 0.7  # Optional, default is 0.0
  max_tokens: 4096  # Optional, maximum response length

# Experiment configuration
experiment:
  human_in_loop: false
  compile_latex: true
  lab_index: 0
  max_steps: 100
  phases:
    - literature_review
    - plan_formulation
    - data_preparation
    - running_experiments
    - results_interpretation
    - report_writing

# Task notes
notes:
  literature_review:
    - "Focus on recent papers about Active Inference and POMDP models"
    - "Include both theoretical and implementation papers"
  plan_formulation:
    - "Include both a baseline and an advanced implementation"
    - "Consider computational efficiency in the design"
  data_preparation:
    - "Prepare synthetic data for testing the models"
  running_experiments:
    - "Compare performance against standard POMDP approaches"
  results_interpretation:
    - "Analyze differences in computational complexity"
  report_writing:
    - "Include clear diagrams of the implementation architecture"

# Output configuration
output:
  base_dir: "outputs"
  save_intermediate: true
  visualizations: true
  report_format: "pdf"
```

## Configuration Sections

### Research Configuration

The `research` section defines the core research topic and metadata:

| Field        | Type   | Description                                      | Required |
|--------------|--------|--------------------------------------------------|----------|
| topic        | string | The main research topic/question                 | Yes      |
| title        | string | A title for the research project                 | No       |
| description  | string | A longer description of the research             | No       |

### Model Configuration

The `model` section configures the language model(s) to use:

| Field            | Type   | Description                               | Required |
|------------------|--------|-------------------------------------------|----------|
| provider         | string | LLM provider (openai, deepseek, etc.)     | Yes      |
| name             | string | Model name (e.g., gpt-4o-mini)            | Yes      |
| api_key          | string | API key for the primary provider          | Yes      |
| deepseek_api_key | string | DeepSeek API key if using DeepSeek models | No       |
| anthropic_api_key| string | Anthropic API key if using Claude         | No       |
| gemini_api_key   | string | Google API key if using Gemini            | No       |
| temperature      | float  | Sampling temperature (0.0-1.0)            | No       |
| max_tokens       | int    | Maximum response length (up to 4096)      | No       |

### Experiment Configuration

The `experiment` section defines how the experiment will run:

| Field         | Type    | Description                                  | Required |
|---------------|---------|----------------------------------------------|----------|
| human_in_loop | boolean | Enable human oversight/interaction           | No       |
| compile_latex | boolean | Whether to compile LaTeX reports to PDF      | No       |
| lab_index     | integer | Index for parallel runs                      | No       |
| max_steps     | integer | Maximum number of agent steps                | No       |
| phases        | array   | List of phases to run (order matters)        | No       |

### Notes Configuration

The `notes` section provides phase-specific guidance to agents:

```yaml
notes:
  phase_name:  # Must match a valid phase name
    - "Note 1 for this phase"
    - "Note 2 for this phase"
```

### Output Configuration

The `output` section configures result storage and formats:

| Field            | Type    | Description                                | Required |
|------------------|---------|--------------------------------------------|----------|
| base_dir         | string  | Base directory for outputs                 | No       |
| save_intermediate| boolean | Save intermediate results                  | No       |
| visualizations   | boolean | Generate visualizations                    | No       |
| report_format    | string  | Output format (pdf, html, md)              | No       |

## Environment Variable Substitution

You can use environment variables in your configuration with the `${VARIABLE_NAME}` syntax:

```yaml
model:
  api_key: "${OPENAI_API_KEY}"
```

## Command-Line Overrides

Some configuration options can be overridden via command-line arguments:

```bash
agent-lab --config experiment.yaml --output-dir custom/output/path --api-key your-api-key
```

## Multiple Configuration Files

You can split your configuration across multiple files and combine them:

```bash
agent-lab --config base.yaml --config experiment_specific.yaml
```

Later files override earlier ones when there are conflicts. 