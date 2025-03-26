# Installation Guide

This guide covers how to install and set up the Agent Laboratory framework for your research workflows.

## Prerequisites

Before installing Agent Laboratory, ensure you have the following prerequisites:

- **Python**: Version 3.10 or higher
- **Pip**: For installing Python packages
- **Git**: For cloning the repository (optional)
- **LLM API Keys**: You'll need API keys for at least one of the supported LLM providers:
  - OpenAI API Key
  - DeepSeek API Key (optional)
  - Anthropic API Key (optional)
  - Google Gemini API Key (optional)

## Installation Methods

### Method 1: Direct Installation (Recommended)

This method installs Agent Laboratory as a package, making it available system-wide.

```bash
# Clone the repository
git clone https://github.com/yourusername/AgentLaboratory.git
cd AgentLaboratory

# Install the package in development mode
pip install -e .

# Test the installation
agent-lab --help
```

### Method 2: Module-based Usage

If you prefer not to install the package, you can run it directly as a module:

```bash
# Clone the repository
git clone https://github.com/yourusername/AgentLaboratory.git
cd AgentLaboratory

# Install dependencies
pip install -r requirements.txt

# Run as a module
python -m agent_lab.main --help
```

### Method 3: Using a Virtual Environment (Recommended for Development)

For development or to avoid package conflicts, using a virtual environment is recommended:

```bash
# Clone the repository
git clone https://github.com/yourusername/AgentLaboratory.git
cd AgentLaboratory

# Create a virtual environment
python -m venv venv_agent_lab

# Activate the virtual environment
# On Windows:
venv_agent_lab\Scripts\activate
# On macOS and Linux:
source venv_agent_lab/bin/activate

# Install in development mode
pip install -e .

# Test the installation
agent-lab --help
```

## Setting Up API Keys

Agent Laboratory requires API keys for the LLM providers you want to use. These can be provided in two ways:

### Option 1: Environment Variables (Recommended)

Set environment variables for your API keys:

```bash
# On macOS/Linux:
export OPENAI_API_KEY=your_openai_key_here
export DEEPSEEK_API_KEY=your_deepseek_key_here
export ANTHROPIC_API_KEY=your_anthropic_key_here
export GEMINI_API_KEY=your_gemini_key_here

# On Windows:
set OPENAI_API_KEY=your_openai_key_here
set DEEPSEEK_API_KEY=your_deepseek_key_here
set ANTHROPIC_API_KEY=your_anthropic_key_here
set GEMINI_API_KEY=your_gemini_key_here
```

You can add these to your `.bashrc`, `.zshrc`, or other shell startup file to make them persistent.

### Option 2: Using a .env File

Create a `.env` file in the root directory of the repository:

```
OPENAI_API_KEY=your_openai_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here
```

The framework will automatically load these environment variables when it starts.

### Option 3: Configuration File

Specify the API keys directly in your experiment configuration file:

```yaml
model:
  provider: "openai"
  name: "gpt-4o-mini"
  api_key: "your_api_key_here"
  deepseek_api_key: "your_deepseek_key_here"  # Optional
  anthropic_api_key: "your_anthropic_key_here"  # Optional
  gemini_api_key: "your_gemini_key_here"  # Optional
  max_tokens: 4096  # Maximum token limit for responses
```

## Verifying Installation

To verify that the installation is working correctly:

```bash
# Run a simple test
agent-lab --config experiment_configs/POMDP_ActiveInference.yaml --debug
```

If everything is set up correctly, you should see the framework start and begin loading the configuration.

## Troubleshooting

### Common Issues

1. **Command not found: agent-lab**
   - Ensure the package is installed correctly: `pip install -e .`
   - Check that your Python scripts directory is in your PATH

2. **ModuleNotFoundError**
   - Install required dependencies: `pip install -r requirements.txt`

3. **API Key errors**
   - Verify your API keys are set correctly in environment variables or config
   - Check that the keys have the necessary permissions for the LLM provider

4. **Python version issues**
   - Ensure you're using Python 3.10 or higher: `python --version`

## Next Steps

After installation, you can:

- Review the [Quick Start Guide](./quickstart.md) to run your first experiment
- Learn about [Configuration Options](./configuration.md) to customize your research workflows
- Explore the [Example Workflow](./example_workflow.md) for a detailed walkthrough 