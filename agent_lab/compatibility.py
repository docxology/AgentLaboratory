"""
Compatibility module for the Agent Laboratory.

This module provides compatibility functions and wrappers to ensure that the
refactored code can work seamlessly with the existing codebase.
"""

import os
import sys
import yaml
import importlib.util
from types import ModuleType
from typing import Dict, Any, Optional, List, Callable

from agent_lab.core.laboratory_workflow import LaboratoryWorkflow
from agent_lab.config.config_loader import load_config


def run_from_original_entry_point(yaml_path: str, **kwargs) -> None:
    """
    Run the laboratory workflow from the original entry point.
    
    Args:
        yaml_path: Path to the YAML configuration file
        **kwargs: Additional keyword arguments to override configuration
    """
    # Load configuration
    config = load_config(yaml_path)
    
    # Override with keyword arguments
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif key.replace('_', '-') in config._raw_data:
            config._raw_data[key.replace('_', '-')] = value
    
    # Initialize and run workflow
    workflow = LaboratoryWorkflow(config=config)
    workflow.perform_research()


def get_original_functions() -> Dict[str, Callable]:
    """
    Get compatibility functions for the original codebase.
    
    Returns:
        Dictionary mapping original function names to compatibility wrappers
    """
    return {
        "parse_yaml": _parse_yaml_compat,
        "parse_arguments": _parse_arguments_compat
    }


class YamlDataHolder:
    """Compatibility class to mimic the original YAML data structure."""
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize with YAML data.
        
        Args:
            data: Dictionary of YAML data
        """
        self.data = data
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to data."""
        return self.data.get(key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value with a default."""
        return self.data.get(key, default)
    
    def __contains__(self, key: str) -> bool:
        """Check if key is in data."""
        return key in self.data


def _parse_yaml_compat(yaml_path: str) -> YamlDataHolder:
    """
    Compatibility wrapper for the original parse_yaml function.
    
    Args:
        yaml_path: Path to the YAML file
        
    Returns:
        YamlDataHolder containing the YAML data
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        try:
            data = yaml.safe_load(f)
            return YamlDataHolder(data)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML: {e}")


def _parse_arguments_compat() -> Dict[str, Any]:
    """
    Compatibility wrapper for the original parse_arguments function.
    
    Returns:
        Dictionary of parsed arguments
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent Laboratory (Compatibility Wrapper)")
    
    parser.add_argument(
        "--yaml-location", 
        type=str, 
        help="Path to the experiment configuration YAML file"
    )
    
    parser.add_argument(
        "--llm-backend", 
        type=str, 
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)"
    )
    
    parser.add_argument(
        "--api-key", 
        type=str, 
        help="OpenAI API key"
    )
    
    parser.add_argument(
        "--deepseek-api-key", 
        type=str, 
        help="DeepSeek API key"
    )
    
    parser.add_argument(
        "--copilot-mode", 
        type=str, 
        default="false",
        help="Enable human-in-the-loop mode (default: false)"
    )
    
    parser.add_argument(
        "--compile-latex", 
        type=str, 
        default="true",
        help="Compile LaTeX to PDF (default: true)"
    )
    
    args = parser.parse_args()
    return vars(args)


def inject_compatibility(module_name: str) -> None:
    """
    Inject compatibility functions into the specified module.
    
    Args:
        module_name: Name of the module to inject into
    """
    # Find the module
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        # Try to find the module file
        for path in sys.path:
            module_path = os.path.join(path, f"{module_name}.py")
            if os.path.exists(module_path):
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                break
        else:
            raise ImportError(f"Module {module_name} not found")
    
    # Inject compatibility functions
    compat_funcs = get_original_functions()
    for func_name, func in compat_funcs.items():
        setattr(module, func_name, func)


# Auto-inject if this module is imported directly into the original codebase
if __name__ != "__main__":
    caller_frame = sys._getframe(1)
    caller_module = caller_frame.f_globals.get("__name__", "")
    
    if caller_module == "ai_lab_repo":
        inject_compatibility(caller_module) 