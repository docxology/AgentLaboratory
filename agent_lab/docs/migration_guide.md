# Migration Guide

This guide will help you transition from the original Agent Laboratory implementation to the refactored version.

## Command-Line Interface Changes

### Original CLI Arguments vs. New CLI Arguments

| Original Argument      | New Argument           | Notes                                       |
|------------------------|-----------------------|---------------------------------------------|
| `--yaml-location`      | `--config`            | Path to experiment configuration YAML file  |
| `--llm-backend`        | *removed*             | Now specified in the configuration file     |
| `--api-key`            | `--api-key`           | Same usage                                  |
| `--deepseek-api-key`   | *moved to config*     | Now specified in the configuration file     |
| `--copilot-mode`       | *moved to config*     | Now specified as `human_in_loop` in config  |
| `--compile-latex`      | *moved to config*     | Now specified in the configuration file     |
| `--output-dir`         | `--output-dir`        | Same usage                                  |
| `--lab-index`          | *moved to config*     | Now specified in the configuration file     |
| *none*                 | `--debug`             | Enables debug logging                       |
| *none*                 | `--phase`             | Run only a specific phase of the workflow   |
| *none*                 | `--resume`            | Resume from a saved state                   |

### Examples

#### Original Command:
```bash
python ai_lab_repo.py --yaml-location experiment_configs/POMDP_ActiveInference.yaml --llm-backend gpt-4o-mini --copilot-mode true
```

#### New Command:
```bash
python -m agent_lab.main --config experiment_configs/POMDP_ActiveInference.yaml --debug
```

Or using the installed package:
```bash
agent-lab --config experiment_configs/POMDP_ActiveInference.yaml --debug
```

## Configuration File Structure

The configuration file structure has been updated to be more consistent and flexible. Here's a comparison:

### Original YAML Structure:
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

### New YAML Structure:
```yaml
# Research configuration
research:
  topic: "POMDP implementation for Active Inference"
  
# Model configuration
model:
  provider: "openai"
  name: "gpt-4o-mini"
  api_key: "your-api-key-here"
  deepseek_api_key: "your-deepseek-key-here"  # Optional

# Experiment configuration
experiment:
  human_in_loop: false
  compile_latex: true
  lab_index: 0

# Task notes
notes:
  plan_formulation:
    - "Note 1 for plan formulation"
    - "Note 2 for plan formulation"
  literature_review:
    - "Note 1 for literature review"
```

## Using the Compatibility Wrapper

For backward compatibility, you can continue using the original command-line interface through the wrapper script:

```bash
python ai_lab_repo_wrapper.py --yaml-location experiment_configs/POMDP_ActiveInference.yaml --llm-backend gpt-4o-mini
```

The wrapper will automatically translate your command-line arguments to the new format.

## API Changes

If you're developing with the Agent Laboratory, here are some key API changes:

1. `LaboratoryWorkflow.run()` is now `LaboratoryWorkflow.perform_research()`
2. Agent `step()` method is now `inference()`
3. Agent prompt generation is now split into `context()`, `phase_prompt()`, and `role_description()`

## Directory Structure Changes

The codebase has been reorganized from flat files into a modular structure:

| Original File        | New Location                              |
|----------------------|-------------------------------------------|
| agents.py            | Split into multiple files in agent_lab/agents/ |
| mlesolver.py         | agent_lab/agents/solvers/mlesolver.py     |
| papersolver.py       | agent_lab/agents/solvers/papersolver.py   |
| tools.py             | agent_lab/utils/legacy_tools.py           |
| utils.py             | agent_lab/utils/legacy_utils.py           |
| common_imports.py    | agent_lab/utils/common_imports.py         |
| app.py               | agent_lab/io/app.py                       |

## Enhancements in the Refactored Version

The refactored implementation includes several enhancements:

1. **Better Logging**: More comprehensive logging of all interactions
2. **Enhanced Visualizations**: Tools for visualizing agent interactions
3. **Improved Debugging**: Support for debugging specific phases
4. **State Management**: Better state tracking and resumability
5. **Modular Structure**: Easier to extend with new agents and functionality
6. **Standardized Interfaces**: Cleaner API for agent interactions
7. **HTML Reports**: Automatically generated reports for experiment results 