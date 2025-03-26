# Agent Laboratory Architecture

This document provides an overview of the architecture for the Agent Laboratory framework.

## System Overview

The Agent Laboratory is a framework for conducting research on multi-agent systems. It provides a structured way to define, run, and analyze experiments with various agent types, configurations, and tasks.

The system is designed to be modular, extensible, and easy to use, with a focus on reproducibility and analysis of results.

## Directory Structure

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
├── docs/               # Documentation
├── utils/              # General utilities
└── visualization/      # Visualization tools
```

## Core Components

### Workflow

The `LaboratoryWorkflow` class serves as the main entry point for running experiments. It manages the overall flow of the experiment, including setting up agents, running steps, and saving results.

```mermaid
classDiagram
    class LaboratoryWorkflow {
        +dict config
        +LLMInterface llm
        +dict agents
        +int step_count
        +__init__(config)
        +perform_research()
        +run_step()
        +create_agent(agent_type)
        +save_state()
        +initialize_output_directory()
        +setup_logging()
        +from_yaml(yaml_path)
        +from_args(args)
    }
    
    LaboratoryWorkflow *-- LLMInterface : uses
    LaboratoryWorkflow *-- BaseAgent : manages
```

### LLM Interface

The `LLMInterface` class provides a unified interface for interacting with various language models (e.g., OpenAI, DeepSeek).

```mermaid
classDiagram
    class LLMInterface {
        +str model
        +str api_key
        +str deepseek_api_key
        +__init__(model, api_key, deepseek_api_key)
        +completion(prompt, system_message)
        +get_model_provider()
    }
```

### Base Agent

The `BaseAgent` class serves as the foundation for all agent implementations. It provides common functionality for managing memory, generating prompts, and saving dialog.

```mermaid
classDiagram
    class BaseAgent {
        +LLMInterface llm
        +dict config
        +str name
        +list memory
        +__init__(llm, config, name)
        +inference(query, phase, feedback)
        +add_to_memory(entry)
        +get_memory()
        +save_dialog()
        +context(phase)
        +phase_prompt(phase)
        +role_description()
    }
    
    BaseAgent *-- LLMInterface : uses
```

## Agents

Various agent implementations extend the `BaseAgent` class to provide specialized functionality.

```mermaid
classDiagram
    BaseAgent <|-- ProfessorAgent
    BaseAgent <|-- PostdocAgent
    BaseAgent <|-- MLEngineerAgent
    BaseAgent <|-- SWEngineerAgent
    BaseAgent <|-- PhDStudentAgent
    BaseAgent <|-- ReviewersAgent
    
    class BaseAgent {
        +LLMInterface llm
        +dict config
        +str name
        +list memory
        +inference()
    }
    
    class ProfessorAgent {
        +context(phase)
        +phase_prompt(phase)
        +role_description()
        +command_descriptions(phase)
    }
    
    class PhDStudentAgent {
        +context(phase)
        +phase_prompt(phase)
        +role_description()
        +command_descriptions(phase)
    }
    
    class ReviewersAgent {
        +inference(plan, report)
    }
```

## Solver Components

```mermaid
classDiagram
    class MLESolver {
        +solve()
        +run_code()
        +reflect_code()
        +process_command()
    }
    
    class PaperSolver {
        +solve()
        +gen_initial_report()
        +process_command()
    }
    
    class Command {
        +execute_command()
        +matches_command()
        +parse_command()
    }
    
    MLESolver --> Command
    PaperSolver --> Command
```

## Configuration

The configuration module provides functionality for loading and validating configuration from various sources (YAML files, command-line arguments, environment variables).

```mermaid
flowchart TD
    A[YAML File] --> D[Configuration]
    B[Command-line Arguments] --> D
    C[Environment Variables] --> D
    D --> E[LaboratoryWorkflow]
```

## Data Flow

The following diagram illustrates the flow of data through the system during a typical experiment.

```mermaid
sequenceDiagram
    participant User
    participant LaboratoryWorkflow
    participant Agents
    participant LLMInterface
    participant OutputFiles
    
    User->>LaboratoryWorkflow: Run experiment
    LaboratoryWorkflow->>LaboratoryWorkflow: Load configuration
    LaboratoryWorkflow->>LaboratoryWorkflow: Setup logging
    LaboratoryWorkflow->>LaboratoryWorkflow: Initialize output directory
    LaboratoryWorkflow->>Agents: Create agents
    
    loop for each phase
        LaboratoryWorkflow->>Agents: Run phase
        Agents->>LLMInterface: Generate completion
        LLMInterface-->>Agents: Completion result
        Agents->>Agents: Process result
        Agents->>Agents: Update memory
        Agents-->>LaboratoryWorkflow: Phase result
        LaboratoryWorkflow->>OutputFiles: Save state
    end
    
    LaboratoryWorkflow->>OutputFiles: Save final results
    LaboratoryWorkflow-->>User: Experiment complete
```

## Compatibility Layer

To ensure backward compatibility with the original implementation, the system includes a compatibility layer.

```mermaid
flowchart TD
    A[Original Entry Point<br>ai_lab_repo_wrapper.py] --> B[Compatibility Layer]
    B --> C[Refactored Implementation]
    D[New Entry Point<br>agent_lab.main] --> C
```

## Command-Line Interface

The Agent Laboratory can be invoked in multiple ways:

1. **Using the compatibility wrapper**:
   ```
   python ai_lab_repo_wrapper.py --yaml-location experiment_configs/POMDP_ActiveInference.yaml
   ```

2. **Using the new module directly**:
   ```
   python -m agent_lab.main --config experiment_configs/POMDP_ActiveInference.yaml
   ```

3. **Using the installed package**:
   ```
   agent-lab --config experiment_configs/POMDP_ActiveInference.yaml
   ```

Note that the command-line arguments changed in the refactored version:
- Original: `--yaml-location` → New: `--config`
- Original: `--llm-backend` → New: Model specified in config
- Additional new options: `--debug`, `--phase`, `--resume`

## Extensibility

The Agent Laboratory is designed to be easily extensible. New agent types, LLM interfaces, and visualization tools can be added without modifying the core components.

```mermaid
flowchart TD
    subgraph Core Components
        A[Laboratory Workflow] --> B[LLM Interface]
        A --> C[Base Agent]
    end
    
    subgraph Extensions
        D[Custom Agent Types] --> C
        E[New LLM Providers] --> B
        F[Visualization Tools] --> A
    end
```

## Workflow Phases

The research workflow is divided into several distinct phases:

1. **Literature Review**
   - Gathering and analyzing relevant research papers
   - Identifying key concepts and approaches
   - Summarizing state of the art

2. **Plan Formulation**
   - Defining research objectives
   - Outlining experiments
   - Selecting methodologies

3. **Data Preparation**
   - Creating or obtaining datasets
   - Preprocessing data
   - Setting up experimental environments

4. **Running Experiments**
   - Executing experimental code
   - Collecting results
   - Monitoring progress

5. **Results Interpretation**
   - Analyzing experimental outcomes
   - Comparing with baselines
   - Drawing conclusions

6. **Report Writing**
   - Documenting methodology
   - Presenting results
   - Discussing implications

Each phase involves specific agents with roles tailored to the requirements of that phase.

## Output and Visualization

The system generates various outputs:

- **Dialog Logs**: Records of all agent interactions
- **Experiment Artifacts**: Code, data, and results
- **Visualization**: Interactive graphs of agent interactions
- **Reports**: Final research reports in LaTeX/PDF format

The visualization tools help analyze:
- Agent interaction patterns
- Time spent in each phase
- Information flow between agents
- Decision points and branches

## Error Handling and Recovery

The system includes mechanisms for:
- Saving state at regular intervals
- Resuming from a saved state
- Handling errors gracefully
- Providing debug information

## Future Extensions

Planned extensions to the architecture include:
- Support for distributed agent execution
- Integration with external tools and services
- Enhanced visualization capabilities
- More specialized agent types
- Improved memory and reasoning models

## Technical Implementation Notes

- Python 3.10+ is required
- Dependencies are managed through pip/requirements.txt
- Configuration uses YAML with environment variable substitution
- Logging leverages Python's standard logging module with enhancements
- Visualization is implemented using HTML/JavaScript for interactivity 