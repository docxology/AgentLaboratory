"""Configuration loader module for the Agent Laboratory.

This module provides functions for loading configuration from various sources
such as YAML files, command-line arguments, and environment variables.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
import argparse


def load_config(
    source: Optional[Union[str, argparse.Namespace]] = None
) -> Dict[str, Any]:
    """Load configuration from a source.
    
    Args:
        source: Source to load configuration from. Can be:
            - A string path to a YAML file
            - An argparse.Namespace object
            - None, in which case environment variables are used
            
    Returns:
        Dict[str, Any]: The loaded configuration
    """
    if source is None:
        return load_from_env()
    elif isinstance(source, str):
        return load_from_yaml(source)
    else:
        return load_from_args(source)


def load_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        yaml_path: Path to the YAML file
        
    Returns:
        Dict[str, Any]: The loaded configuration
        
    Raises:
        FileNotFoundError: If the YAML file does not exist
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def load_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Load configuration from command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dict[str, Any]: The loaded configuration
    """
    config = {}
    
    # Extract configuration from arguments
    if hasattr(args, 'yaml_location') and args.yaml_location:
        # Load base configuration from YAML if provided
        config.update(load_from_yaml(args.yaml_location))
    
    # Override with command-line arguments
    for key, value in vars(args).items():
        if value is not None:
            # Convert dashed keys to underscores
            config_key = key.replace('-', '_')
            config[config_key] = value
    
    return config


def load_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables.
    
    Returns:
        Dict[str, Any]: The loaded configuration
    """
    config = {}
    
    # API keys
    if openai_api_key := os.environ.get('OPENAI_API_KEY'):
        config['api_key'] = openai_api_key
    
    if deepseek_api_key := os.environ.get('DEEPSEEK_API_KEY'):
        config['deepseek_api_key'] = deepseek_api_key
    
    # LLM backend
    if llm_backend := os.environ.get('AGENT_LAB_LLM_BACKEND'):
        config['llm_backend'] = llm_backend
    
    # Debug mode
    if debug := os.environ.get('AGENT_LAB_DEBUG'):
        config['debug'] = debug.lower() in ('true', '1', 'yes')
    
    # Output directory
    if output_dir := os.environ.get('AGENT_LAB_OUTPUT_DIR'):
        config['output_dir'] = output_dir
    
    # Research topic
    if research_topic := os.environ.get('AGENT_LAB_RESEARCH_TOPIC'):
        config['research_topic'] = research_topic
    
    return config 