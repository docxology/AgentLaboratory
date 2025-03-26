"""
Configuration loader for the Agent Laboratory.

This module provides utilities for loading and managing configuration from various sources,
including YAML files, command-line arguments, and environment variables.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict

# Setup logger
logger = logging.getLogger(__name__)

# Default values
DEFAULT_RESEARCH_PHASES = [
    "plan-formulation",
    "literature-review",
    "data-preparation",
    "running-experiments",
    "results-interpretation",
    "report-writing"
]

DEFAULT_LLM_BACKEND = "gpt-4o-mini"
DEFAULT_COMPILE_LATEX = True
DEFAULT_COPILOT_MODE = False


@dataclass
class ConfigValue:
    """A configuration value with metadata."""
    value: Any
    description: str
    required: bool = False
    source: str = "default"


@dataclass
class AgentLabConfig:
    """Configuration class for the Agent Laboratory."""
    
    # Research configuration
    research_topic: str = ""
    
    # Model configuration
    llm_backend: str = DEFAULT_LLM_BACKEND
    api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    
    # Task configuration
    notes: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime configuration
    copilot_mode: bool = DEFAULT_COPILOT_MODE
    compile_latex: bool = DEFAULT_COMPILE_LATEX
    max_steps: int = 10
    output_dir: str = "research_outputs"
    lab_index: int = 0
    
    # State management
    load_state: Optional[str] = None
    
    # Debug configuration
    debug: bool = False


class ConfigLoader:
    """Loader for Agent Laboratory configuration."""
    
    def load_from_yaml(self, yaml_path: str) -> AgentLabConfig:
        """Load configuration from a YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            AgentLabConfig: The loaded configuration
            
        Raises:
            FileNotFoundError: If the YAML file does not exist
            yaml.YAMLError: If the YAML file is invalid
        """
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
            
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
            
        # Create config with defaults
        config = AgentLabConfig()
        
        # Update with YAML data
        for key, value in yaml_data.items():
            if hasattr(config, key):
                setattr(config, key, value)
                
        return config
    
    def load_from_args(self, args: Any) -> AgentLabConfig:
        """Load configuration from command-line arguments.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            AgentLabConfig: The loaded configuration
        """
        # Start with default config
        config = AgentLabConfig()
        
        # Load from YAML file if provided
        if hasattr(args, 'yaml_location') and args.yaml_location:
            yaml_config = self.load_from_yaml(args.yaml_location)
            # Update config with YAML values
            for key, value in asdict(yaml_config).items():
                setattr(config, key, value)
        
        # Override with command-line arguments
        if hasattr(args, 'llm_backend') and args.llm_backend:
            config.llm_backend = args.llm_backend
            
        if hasattr(args, 'api_key') and args.api_key:
            config.api_key = args.api_key
            
        if hasattr(args, 'deepseek_api_key') and args.deepseek_api_key:
            config.deepseek_api_key = args.deepseek_api_key
            
        if hasattr(args, 'copilot_mode'):
            if isinstance(args.copilot_mode, bool):
                config.copilot_mode = args.copilot_mode
            elif isinstance(args.copilot_mode, str):
                config.copilot_mode = args.copilot_mode.lower() == 'true'
                
        if hasattr(args, 'compile_latex'):
            if isinstance(args.compile_latex, bool):
                config.compile_latex = args.compile_latex
            elif isinstance(args.compile_latex, str):
                config.compile_latex = args.compile_latex.lower() == 'true'
                
        if hasattr(args, 'output_dir') and args.output_dir:
            config.output_dir = args.output_dir
            
        if hasattr(args, 'lab_index') and args.lab_index is not None:
            config.lab_index = int(args.lab_index)
            
        if hasattr(args, 'load_state') and args.load_state:
            config.load_state = args.load_state
            
        if hasattr(args, 'debug') and args.debug:
            config.debug = args.debug
            
        return config
    
    def load_from_env(self) -> AgentLabConfig:
        """Load configuration from environment variables.
        
        Returns:
            AgentLabConfig: The loaded configuration
        """
        config = AgentLabConfig()
        
        # Override with environment variables
        if 'AGENT_LAB_LLM_BACKEND' in os.environ:
            config.llm_backend = os.environ['AGENT_LAB_LLM_BACKEND']
            
        if 'OPENAI_API_KEY' in os.environ:
            config.api_key = os.environ['OPENAI_API_KEY']
            
        if 'DEEPSEEK_API_KEY' in os.environ:
            config.deepseek_api_key = os.environ['DEEPSEEK_API_KEY']
            
        if 'AGENT_LAB_COPILOT_MODE' in os.environ:
            config.copilot_mode = os.environ['AGENT_LAB_COPILOT_MODE'].lower() == 'true'
            
        if 'AGENT_LAB_COMPILE_LATEX' in os.environ:
            config.compile_latex = os.environ['AGENT_LAB_COMPILE_LATEX'].lower() == 'true'
            
        if 'AGENT_LAB_OUTPUT_DIR' in os.environ:
            config.output_dir = os.environ['AGENT_LAB_OUTPUT_DIR']
            
        if 'AGENT_LAB_LAB_INDEX' in os.environ:
            config.lab_index = int(os.environ['AGENT_LAB_LAB_INDEX'])
            
        if 'AGENT_LAB_DEBUG' in os.environ:
            config.debug = os.environ['AGENT_LAB_DEBUG'].lower() == 'true'
            
        return config


def load_config(source: Union[str, Any, None] = None) -> AgentLabConfig:
    """Convenience function to load configuration from various sources.
    
    Args:
        source: Source of configuration (file path, args object, or None for env)
        
    Returns:
        AgentLabConfig: The loaded configuration
    """
    loader = ConfigLoader()
    
    if source is None:
        return loader.load_from_env()
    elif isinstance(source, str):
        return loader.load_from_yaml(source)
    else:
        return loader.load_from_args(source)


def load_config_from_file(file_path: str) -> AgentLabConfig:
    """
    Load configuration from a YAML file.
    
    Args:
        file_path: Path to the YAML configuration file
        
    Returns:
        AgentLabConfig: The loaded configuration
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML file is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file '{file_path}' not found")
    
    logger.info(f"Loading configuration from {file_path}")
    
    with open(file_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return load_config(config_data)


def validate_config(config: AgentLabConfig) -> List[str]:
    """
    Validate the configuration and return a list of validation errors.
    
    Returns:
        List[str]: List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check required fields
    if not config.research_topic:
        errors.append("Research topic is required")
    
    if not config.api_key and not config.deepseek_api_key:
        errors.append("Either OpenAI API key or DeepSeek API key is required")
    
    return errors


def save_config_to_file(config: AgentLabConfig, file_path: str) -> None:
    """
    Save the current configuration to a YAML file.
    
    Args:
        config: The configuration to save
        file_path: Path to save the YAML configuration file
        
    Raises:
        IOError: If the file cannot be written
    """
    if not config:
        logger.warning("Saving empty configuration")
    
    config_dict = asdict(config)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    with open(file_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    logger.info(f"Configuration saved to {file_path}") 