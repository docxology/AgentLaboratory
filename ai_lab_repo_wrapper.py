#!/usr/bin/env python3
"""Wrapper script for backward compatibility with the original ai_lab_repo.py.

This script maintains the original command-line interface while using the refactored
implementation underneath if available. If the refactored implementation is not
available, it falls back to the original implementation.
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
        # Try to import the refactored implementation
        from agent_lab.core.laboratory_workflow import LaboratoryWorkflow
        from agent_lab.config.config_loader import ConfigLoader
        
        # Use the refactored implementation
        config = ConfigLoader().load_from_args(args)
        workflow = LaboratoryWorkflow(config)
        workflow.perform_research()
        
    except ImportError:
        # Fall back to the original implementation
        print("Refactored implementation not found. Falling back to original implementation.")
        try:
            import ai_lab_repo
            ai_lab_repo.main(args)
        except Exception as e:
            print(f"Error running original implementation: {e}")
            traceback.print_exc()
            sys.exit(1)
    except Exception as e:
        print(f"Error running refactored implementation: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 