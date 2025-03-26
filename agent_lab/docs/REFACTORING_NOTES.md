# Agent Laboratory Refactoring Notes

This document provides an overview of the refactoring performed on the Agent Laboratory codebase.

## Refactoring Goals

The main goals of this refactoring were:

1. **Improved Modularity**: Break down monolithic files into smaller, purpose-specific modules
2. **Better Organization**: Organize code into a clear directory structure
3. **Enhanced Maintainability**: Make the codebase easier to understand and extend
4. **Backward Compatibility**: Ensure existing scripts and workflows continue to work

## Directory Structure

The refactored codebase follows this structure:

```
agent_lab/
├── agents/             # Agent implementations
│   ├── solvers/        # Specialized solver agents (MLESolver, PaperSolver)
│   └── ...             # Other agent implementations
├── config/             # Configuration management
├── core/               # Core functionality
│   ├── laboratory_workflow.py  # Main workflow implementation
│   ├── llm_interface.py        # LLM integration
│   └── base_agent.py           # Base agent class
├── io/                 # Input/output utilities
├── logging/            # Logging functionality
├── utils/              # General utilities
├── visualization/      # Visualization tools
└── docs/               # Documentation
```

## File Migrations

The following files were moved from the root directory to their new locations:

| Original File        | New Location                              |
|----------------------|-------------------------------------------|
| agents.py            | Split into multiple files in agent_lab/agents/ |
| mlesolver.py         | agent_lab/agents/solvers/mlesolver.py     |
| papersolver.py       | agent_lab/agents/solvers/papersolver.py   |
| tools.py             | agent_lab/utils/legacy_tools.py           |
| utils.py             | agent_lab/utils/legacy_utils.py           |
| common_imports.py    | agent_lab/utils/common_imports.py         |
| app.py               | agent_lab/io/app.py                       |

## Key Components

### LaboratoryWorkflow

The main entry point for running experiments. It manages the overall flow, including:
- Setting up agents
- Running experiment steps
- Saving results

### LLM Interface

Provides a unified interface for interacting with various language models:
- OpenAI models (gpt-4o, o1, etc.)
- DeepSeek models
- Other LLM providers can be added

### Agent Hierarchy

All agents inherit from the BaseAgent class, which provides:
- Common agent functionality
- Memory management
- Dialog formatting

Specialized agents include:
- ProfessorAgent
- PostdocAgent
- MLEngineerAgent
- SWEngineerAgent
- PhDStudentAgent
- ReviewersAgent

### Solvers

Specialized components for specific tasks:
- MLESolver: For machine learning experiments
- PaperSolver: For report generation

## Backward Compatibility

The `ai_lab_repo_wrapper.py` script maintains backward compatibility with the original interface. It:
1. Accepts the same command-line arguments as the original script
2. Uses the refactored implementation when available
3. Falls back to the original implementation if needed

## Usage

### Original Method (Still Supported)

```bash
python ai_lab_repo_wrapper.py --yaml-location experiment_configs/POMDP_ActiveInference.yaml
```

### New Method

```bash
# As a module
python -m agent_lab.main --config experiment_configs/POMDP_ActiveInference.yaml

# As an installed package
agent-lab --config experiment_configs/POMDP_ActiveInference.yaml
```

## Next Steps

1. Complete the migration of functionality from deprecated files
2. Add comprehensive documentation for all modules
3. Implement additional tests
4. Enhance visualization tools 