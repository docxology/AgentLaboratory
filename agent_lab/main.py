#!/usr/bin/env python3
"""
Main entry point for the Agent Laboratory.

This module provides the main entry point for running the Agent Laboratory workflow,
either from the command line or as a library.
"""

import os
import sys
import argparse
import logging
import yaml
from typing import Dict, List, Any, Optional

# Import dotenv to load environment variables from .env file
from dotenv import load_dotenv

from agent_lab.core.workflow import LaboratoryWorkflow
from agent_lab.config.loader import load_config

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


def configure_logging(debug: bool = False):
    """Configure logging for the application.
    
    Args:
        debug: Whether to enable debug logging
    """
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Set specific loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.INFO)
    logging.getLogger('agent_lab').setLevel(log_level)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Agent Laboratory - Autonomous Research Framework')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--yaml-location', type=str, help='Path to YAML configuration file (alias for --config)')
    parser.add_argument('--resume', type=str, help='Resume from state file')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--api-key', type=str, help='API key for LLM provider')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--phase', type=str, help='Run a specific phase')
    
    return parser.parse_args()


def process_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Process and normalize the configuration.
    
    Args:
        config: Raw configuration dictionary
    
    Returns:
        Dict[str, Any]: Processed configuration dictionary
    """
    # Convert dash-case keys to underscore_case
    processed_config = {}
    for key, value in config.items():
        # Convert dashed keys to underscores
        processed_key = key.replace("-", "_")
        processed_config[processed_key] = value
    
    # Ensure API key is set
    if "api_key" not in processed_config and os.environ.get("OPENAI_API_KEY") is None:
        # Check if we have the dash version
        if "api-key" in config:
            processed_config["api_key"] = config["api-key"]
    
    # Ensure deepseek API key is set
    if "deepseek_api_key" not in processed_config and os.environ.get("DEEPSEEK_API_KEY") is None:
        # Check if we have the dash version
        if "deepseek-api-key" in config:
            processed_config["deepseek_api_key"] = config["deepseek-api-key"]
    
    # Ensure notes are properly processed
    if "task_notes" not in processed_config and "task-notes" in config:
        processed_config["task_notes"] = config["task-notes"]
    
    # Convert string boolean values to actual booleans
    for key in ["copilot_mode", "compile_latex", "debug"]:
        if key in processed_config and isinstance(processed_config[key], str):
            processed_config[key] = processed_config[key].lower() in ["true", "yes", "1", "t"]
    
    return processed_config


def run_workflow(config: Dict[str, Any]) -> None:
    """Run the Agent Laboratory workflow.
    
    Args:
        config: Configuration dictionary
    """
    # Process the configuration
    processed_config = process_config(config)
    
    # Create the workflow
    workflow = LaboratoryWorkflow(processed_config)
    
    # Run the workflow using the run method
    workflow.run()


def main() -> None:
    """Main entry point for the application."""
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging
    configure_logging(args.debug)
    
    # Load configuration
    config_path = args.config or args.yaml_location
    if not config_path:
        logging.error("No configuration file specified. Use --config or --yaml-location")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Override with command-line arguments
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    if args.api_key:
        if 'llm_config' not in config:
            config['llm_config'] = {}
        config['llm_config']['api_key'] = args.api_key
    
    # Check for required configuration
    if not config.get('llm_config', {}).get('api_key'):
        # Try to get from environment
        api_key = os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')
        if api_key:
            if 'llm_config' not in config:
                config['llm_config'] = {}
            config['llm_config']['api_key'] = api_key
            logging.info(f"Using API key from environment variables")
        else:
            logging.error("No API key found. Please specify with --api-key or set OPENAI_API_KEY environment variable")
            sys.exit(1)
    
    # Create workflow
    workflow = LaboratoryWorkflow(config)
    
    # Resume if specified
    if args.resume:
        workflow.load_state(args.resume)
        logging.info(f"Resumed from state file: {args.resume}")
    
    # Run workflow
    if args.phase:
        # Run specific phase
        logging.info(f"Running phase: {args.phase}")
        if args.phase == "literature-review":
            workflow.run_literature_review()
        elif args.phase == "plan-formulation":
            workflow.run_plan_formulation()
        elif args.phase == "data-preparation":
            workflow.run_data_preparation()
        elif args.phase == "code-implementation":
            workflow.run_code_implementation()
        elif args.phase == "running-experiments":
            workflow.run_experiments()
        elif args.phase == "results-interpretation":
            workflow.run_results_interpretation()
        elif args.phase == "report-writing":
            workflow.run_report_writing()
        else:
            logging.error(f"Unknown phase: {args.phase}")
            sys.exit(1)
    else:
        # Run full workflow
        # Use the run method instead of run_workflow
        workflow.run()
    
    logging.info("Agent Laboratory workflow completed successfully")


def run_from_config(
    config_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """Run the Agent Laboratory workflow from a configuration.
    
    Args:
        config_path: Path to the configuration YAML file
        config: Configuration dictionary
        **kwargs: Additional configuration parameters
    """
    # Load environment variables from .env file
    load_dotenv()
    
    if config_path is not None:
        workflow_config = load_config(config_path)
    elif config is not None:
        workflow_config = config.copy()
    else:
        workflow_config = {}
    
    # Add any additional parameters
    workflow_config.update(kwargs)
    
    # Run the workflow
    run_workflow(workflow_config)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Store the config file path in the config itself
        config["config_file"] = config_path
        
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 