# Agent Laboratory (Refactored Implementation)

This directory contains the refactored implementation of the Agent Laboratory, designed to provide improved modularity, better logging, enhanced visualization, and more comprehensive output tracking.

## Directory Structure

- `agent_lab/core/`: Core functionality
  - `laboratory_workflow.py`: Main workflow coordinator
  - `llm_interface.py`: Interface for LLM providers

- `agent_lab/agents/`: Agent implementations
  - `base_agent.py`: Base class for all agents
  - `professor_agent.py`: Professor agent implementation

- `agent_lab/config/`: Configuration handling
  - `config_loader.py`: YAML configuration loading and validation

- `agent_lab/logging/`: Enhanced logging
  - `research_logger.py`: Comprehensive research logger

- `agent_lab/visualization/`: Visualization tools
  - `experiment_visualizer.py`: Experiment visualization and reporting
  - `templates/`: HTML templates for reports

- `agent_lab/utils/`: Utility functions
  - Various utilities for file handling, formatting, etc.

## Key Improvements

1. **Modular Structure**: Code is organized into logical modules for better maintainability
2. **Enhanced Logging**: Comprehensive logging of all agent dialogs, interactions, and outputs
3. **Visualization Tools**: Tools for visualizing agent interactions and experiment progress
4. **Improved Debugging**: Better error handling, state tracking, and reproducibility
5. **Compatibility Layer**: Backward compatibility with the original codebase
6. **Centralized Configuration**: Unified configuration management

## Usage

### Command-line Usage

```bash
# Run with a configuration file
python -m agent_lab.main --config experiment_configs/POMDP_ActiveInference.yaml

# Run a specific phase
python -m agent_lab.main --config experiment_configs/POMDP_ActiveInference.yaml --phase plan-formulation

# Enable debug mode
python -m agent_lab.main --config experiment_configs/POMDP_ActiveInference.yaml --debug
```

### Backward Compatibility

```bash
# Use the compatibility wrapper
python ai_lab_repo_wrapper.py --yaml-location experiment_configs/POMDP_ActiveInference.yaml
```

### Library Usage

```python
from agent_lab.config.config_loader import load_config
from agent_lab.core.laboratory_workflow import LaboratoryWorkflow

# Load configuration
config = load_config("experiment_configs/POMDP_ActiveInference.yaml")

# Create and run workflow
workflow = LaboratoryWorkflow(config)
workflow.perform_research()
```

## Experiment Outputs

Experiment outputs are organized in directories with the following structure:

```
research_outputs/
└── research_topic_timestamp_lab0/
    ├── logs/              # Log files
    ├── reports/           # HTML reports and documentation
    ├── metadata/          # Configuration and metrics
    ├── agent_dialogs/     # Agent dialog records by phase
    ├── artifacts/         # Research artifacts by phase
    ├── source/            # Experiment source code
    ├── visualizations/    # Charts and visualizations
    └── states/            # Experiment state snapshots
```

## Example Configuration

```yaml
# Research topic
research-topic: "POMDP implementation for Active Inference"

# Model to use
llm-backend: "gpt-4o-mini"

# API key
api-key: "your-api-key-here"

# Human-in-the-loop mode
copilot_mode: false

# Compile LaTeX
compile-latex: true

# Task notes
task-notes:
  plan-formulation:
    - "Note 1 for plan formulation"
    - "Note 2 for plan formulation"
  literature-review:
    - "Note 1 for literature review"
``` 