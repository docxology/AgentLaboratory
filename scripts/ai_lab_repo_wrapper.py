#!/usr/bin/env python3
"""Wrapper script for backward compatibility with the original ai_lab_repo.py.

This script maintains the original command-line interface while using the refactored
implementation. The original ai_lab_repo.py is no longer required as the implementation
has been completely refactored into the agent_lab package.
"""

import os
import sys
import argparse
import traceback

def main():
    """Main entry point for the wrapper script."""
    parser = argparse.ArgumentParser(description="Agent Laboratory - Research Assistant Workflow")
    parser.add_argument("--yaml-location", type=str, help="Path to experiment configuration YAML file")
    parser.add_argument("--llm-backend", type=str, default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--deepseek-api-key", type=str, help="DeepSeek API key")
    parser.add_argument("--copilot-mode", type=str, default="false", help="Enable human-in-the-loop mode")
    parser.add_argument("--compile-latex", type=str, default="true", help="Compile LaTeX to PDF")
    parser.add_argument("--output-dir", type=str, help="Base directory for research outputs")
    parser.add_argument("--lab-index", type=int, default=0, help="Index of the laboratory instance for parallel runs")
    
    args = parser.parse_args()
    
    # Verify that the YAML file exists if provided
    if args.yaml_location and not os.path.isfile(args.yaml_location):
        print(f"Error: YAML file not found at {args.yaml_location}")
        sys.exit(1)
    
    try:
        # Import the refactored implementation
        from agent_lab.core.workflow import LaboratoryWorkflow
        from agent_lab.config.loader import load_config
        
        # Use the refactored implementation
        print("Using refactored Agent Laboratory implementation...")
        
        # Load configuration from the YAML file
        config = load_config(args.yaml_location)
        
        # Override with command-line arguments
        if args.llm_backend:
            config['llm_backend'] = args.llm_backend
        if args.api_key:
            config['api_key'] = args.api_key
        if args.deepseek_api_key:
            config['deepseek_api_key'] = args.deepseek_api_key
        if args.copilot_mode:
            config['copilot_mode'] = args.copilot_mode.lower() in ['true', 'yes', '1', 't']
        if args.compile_latex:
            config['compile_latex'] = args.compile_latex.lower() in ['true', 'yes', '1', 't']
        if args.output_dir:
            config['output_dir'] = args.output_dir
        if args.lab_index:
            config['lab_index'] = args.lab_index
            
        # Create and run workflow
        workflow = LaboratoryWorkflow(config)
        workflow.run()
        
    except ImportError as e:
        print(f"Error: Refactored implementation not found: {e}")
        print("Please make sure the agent_lab package is properly installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running the implementation: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 