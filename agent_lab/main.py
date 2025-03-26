#!/usr/bin/env python3
"""
Main entry point for the Agent Laboratory.

This module provides the command-line interface for running
research experiments in the Agent Laboratory.
"""

import os
import sys
import argparse
import yaml
import logging
import traceback
from typing import Dict, Any, Optional, List, Union

from agent_lab.config.config_loader import load_config
from agent_lab.core.laboratory_workflow import LaboratoryWorkflow


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Agent Laboratory: LLM-powered research assistant workflow"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to experiment configuration YAML file"
    )
    
    parser.add_argument(
        "--yaml-location", 
        type=str, 
        help="Legacy path to experiment configuration YAML file (for backward compatibility)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="research_outputs",
        help="Base directory for research outputs"
    )
    
    parser.add_argument(
        "--lab-index", 
        type=int, 
        default=0,
        help="Index of this laboratory instance (for parallel runs)"
    )
    
    parser.add_argument(
        "--load-state", 
        type=str,
        help="Path to state file to load"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--phase", 
        type=str,
        choices=[
            "literature-review",
            "plan-formulation",
            "data-preparation",
            "running-experiments",
            "results-interpretation",
            "report-writing"
        ],
        help="Run only a specific phase of the workflow"
    )
    
    return parser.parse_args()


def validate_config_path(args):
    """
    Validate and determine the configuration file path.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Path to the configuration file
    
    Raises:
        FileNotFoundError: If no configuration file is found
    """
    config_path = None
    
    # Check --config argument
    if args.config:
        if os.path.exists(args.config):
            return args.config
        else:
            print(f"Error: Specified config file not found: {args.config}")
    
    # Check --yaml-location argument (for backward compatibility)
    if args.yaml_location:
        if os.path.exists(args.yaml_location):
            return args.yaml_location
        else:
            print(f"Error: Specified yaml file not found: {args.yaml_location}")
    
    # Check for common default file names
    common_names = [
        "config.yaml",
        "experiment.yaml",
        "POMDP_ActiveInference.yaml",
        "experiment_configs/POMDP_ActiveInference.yaml"
    ]
    
    for name in common_names:
        if os.path.exists(name):
            print(f"Using default configuration file: {name}")
            return name
    
    raise FileNotFoundError(
        "No configuration file found. Please provide a file path using --config"
    )


def run_workflow(args):
    """
    Run the Agent Laboratory workflow.
    
    Args:
        args: Command-line arguments
    """
    try:
        # Determine configuration file path
        config_path = validate_config_path(args)
        
        # Load configuration
        config = load_config(config_path)
        
        # Override with command-line arguments
        if args.output_dir:
            config.output_dir = args.output_dir
        
        if args.lab_index:
            config.lab_index = args.lab_index
        
        if args.debug:
            config.debug = True
        
        # Initialize workflow
        workflow = LaboratoryWorkflow(
            config=config,
            load_state_path=args.load_state,
            debug=args.debug
        )
        
        # Run specified phase or full workflow
        if args.phase:
            print(f"Running single phase: {args.phase}")
            workflow.run_phase(args.phase)
        else:
            print(f"Running full workflow for: {config.research_topic}")
            workflow.perform_research()
        
        print("Workflow completed successfully.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running workflow: {e}")
        if args.debug:
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_arguments()
    run_workflow(args)


if __name__ == "__main__":
    main() 