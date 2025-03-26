# Agent Laboratory (Refactored Version)

This is a refactored version of Agent Laboratory with improved structure, better logging, enhanced visualization, and more comprehensive output tracking.

## Key Features of the Refactored Implementation

1. **Modular Structure**: Code is organized into logical modules for better maintainability
2. **Enhanced Logging**: Comprehensive logging of all agent dialogs, interactions, and outputs
3. **Visualization Tools**: Tools for visualizing agent interactions and experiment progress
4. **Improved Debugging**: Better error handling, state tracking, and reproducibility
5. **Detailed Outputs**: All experiment artifacts, including intermediary steps, are saved in a structured manner
6. **HTML Reports**: Automatically generated HTML reports for each experiment
7. **Backward Compatibility**: Fully compatible with the original codebase and YAML configurations

## Installation

### Method 1: Direct Installation (Recommended)

```bash
# Install the package
pip install -e .

# Run the agent lab
agent-lab --config experiment_configs/POMDP_ActiveInference.yaml
```

### Method 2: Module-based Usage

```bash
# Run as a module
python -m agent_lab.main --config experiment_configs/POMDP_ActiveInference.yaml
```

### Method 3: Backward Compatibility Wrapper

```bash
# Use the compatibility wrapper
python ai_lab_repo_wrapper.py --yaml-location experiment_configs/POMDP_ActiveInference.yaml
```

## Directory Structure

```
agent_lab/               # Main package directory
├── core/                # Core functionality
├── agents/              # Agent implementations
├── config/              # Configuration handling
├── logging/             # Enhanced logging
├── visualization/       # Visualization tools
└── utils/               # Utilities

# Supporting files
setup.py                 # Package installation
ai_lab_repo_wrapper.py   # Backward compatibility wrapper
```

## Migration Guide

### For Users

If you're currently using the original `ai_lab_repo.py` script, you can:

1. **Use the Compatibility Wrapper**: The `ai_lab_repo_wrapper.py` script accepts the same command-line arguments as the original script.

   ```bash
   python ai_lab_repo_wrapper.py --yaml-location experiment_configs/POMDP_ActiveInference.yaml
   ```

2. **Switch to the New CLI**: The new command-line interface provides additional options for debugging and running specific phases.

   ```bash
   python -m agent_lab.main --config experiment_configs/POMDP_ActiveInference.yaml
   ```

3. **Install as a Package**: For the best experience, install the package and use the `agent-lab` command.

   ```bash
   pip install -e .
   agent-lab --config experiment_configs/POMDP_ActiveInference.yaml
   ```

Your existing YAML configurations will work without changes.

### For Developers

If you're developing with the codebase:

1. **Modular Development**: Add new agents in the `agent_lab/agents/` directory.

2. **Extending Functionality**: Add new visualizations in `agent_lab/visualization/` or new logging capabilities in `agent_lab/logging/`.

3. **Configuration**: Extend the configuration options in `agent_lab/config/config_loader.py`.

4. **LLM Integration**: Add support for new models in `agent_lab/core/llm_interface.py`.

## Configuration

Example YAML configuration:

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

## Command-line Arguments

```
usage: agent-lab [-h] [--config CONFIG] [--yaml-location YAML_LOCATION]
                 [--output-dir OUTPUT_DIR] [--lab-index LAB_INDEX]
                 [--load-state LOAD_STATE] [--debug]
                 [--phase {literature-review,plan-formulation,data-preparation,running-experiments,results-interpretation,report-writing}]

Agent Laboratory: LLM-powered research assistant workflow

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to experiment configuration YAML file
  --yaml-location YAML_LOCATION
                        Legacy path to experiment configuration YAML file (for backward compatibility)
  --output-dir OUTPUT_DIR
                        Base directory for research outputs
  --lab-index LAB_INDEX
                        Index of this laboratory instance (for parallel runs)
  --load-state LOAD_STATE
                        Path to state file to load
  --debug               Enable debug mode
  --phase {literature-review,plan-formulation,data-preparation,running-experiments,results-interpretation,report-writing}
                        Run only a specific phase of the workflow
```

## Contributing

Contributions are welcome! Please see the [contributing guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 