"""
Agent Laboratory - LLM-powered research assistant workflow.

This package provides a structured workflow for conducting research with LLM-powered agents,
including comprehensive logging, visualization, and output management.
"""

__version__ = "0.1.0"

# Core imports
from agent_lab.core.laboratory_workflow import LaboratoryWorkflow
from agent_lab.core.llm_interface import LLMInterface

# Agent imports
from agent_lab.agents.base_agent import BaseAgent
from agent_lab.agents.professor_agent import ProfessorAgent

# Configuration imports
from agent_lab.config.config_loader import ConfigLoader, load_config, AgentLabConfig

# Logging imports
from agent_lab.logging.research_logger import ResearchLogger

# Visualization imports
from agent_lab.visualization.experiment_visualizer import ExperimentVisualizer

# Main entry point
from agent_lab.main import main

__all__ = [
    "LaboratoryWorkflow",
    "LLMInterface",
    "BaseAgent",
    "ProfessorAgent",
    "ConfigLoader",
    "load_config",
    "AgentLabConfig",
    "ResearchLogger",
    "ExperimentVisualizer",
    "main",
]
