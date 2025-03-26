"""
Laboratory workflow for the Agent Laboratory.

This module defines the main workflow for the Agent Laboratory.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Type, Union

from agent_lab.core.llm_interface import LLMInterface
from agent_lab.core.base_agent import BaseAgent
from agent_lab.agents import ProfessorAgent
from agent_lab.config.config_loader import AgentLabConfig
from agent_lab.logging.research_logger import ResearchLogger
from agent_lab.visualization.experiment_visualizer import ExperimentVisualizer

class LaboratoryWorkflow:
    """Main workflow manager for the Agent Laboratory."""
    
    DEFAULT_PHASES = [
        "plan-formulation",
        "literature-review", 
        "experiment-design",
        "data-collection",
        "data-analysis", 
        "result-interpretation",
        "report-writing"
    ]
    
    def __init__(
        self,
        config: AgentLabConfig,
        logger: Optional[ResearchLogger] = None
    ):
        """Initialize the laboratory workflow.
        
        Args:
            config: Configuration for the workflow
            logger: Optional research logger
        """
        self.config = config
        
        # Create logger if not provided
        if logger is None:
            self.logger = ResearchLogger(
                output_dir=config.output_dir,
                research_topic=config.research_topic,
                debug=config.debug,
                lab_index=config.lab_index
            )
        else:
            self.logger = logger
        
        # Initialize LLM interface
        self.llm = LLMInterface(
            model=config.llm_backend,
            api_key=config.api_key,
            deepseek_api_key=config.deepseek_api_key
        )
        
        # Initialize experiment visualizer
        self.visualizer = ExperimentVisualizer(self.logger)
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Initialize state
        self.current_phase = None
        self.phase_results = {}
        
        self.logger.log_info("Laboratory workflow initialized")
    
    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """Initialize the agents for the workflow.
        
        Returns:
            Dict[str, BaseAgent]: Dictionary of agent name to agent instance
        """
        agents = {}
        
        # Create the professor agent for research planning and report writing
        professor_agent = ProfessorAgent(
            model=self.config.llm_backend,
            notes=self.config.notes,
            max_steps=self.config.max_steps,
            api_key=self.config.api_key,
            logger=self.logger
        )
        agents["professor"] = professor_agent
        
        # Add more agents here as they are implemented
        
        self.logger.log_info(f"Initialized {len(agents)} agents")
        return agents
    
    def _save_state(self) -> str:
        """Save the current state of the workflow.
        
        Returns:
            str: Path to the saved state file
        """
        state_dir = self.logger.get_state_dir()
        os.makedirs(state_dir, exist_ok=True)
        
        state = {
            "config": {
                "research_topic": self.config.research_topic,
                "llm_backend": self.config.llm_backend,
                "output_dir": self.config.output_dir,
                "lab_index": self.config.lab_index,
                "max_steps": self.config.max_steps,
                "debug": self.config.debug
            },
            "current_phase": self.current_phase,
            "phase_results": self.phase_results
        }
        
        # Save agent states
        for agent_name, agent in self.agents.items():
            agent.save_state(state_dir)
        
        state_file = os.path.join(state_dir, "workflow_state.json")
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        
        self.logger.log_info(f"Workflow state saved to {state_file}")
        return state_file
    
    def _load_state(self, state_file: str) -> None:
        """Load the workflow state from a file.
        
        Args:
            state_file: Path to the state file
            
        Raises:
            FileNotFoundError: If the state file does not exist
        """
        if not os.path.isfile(state_file):
            raise FileNotFoundError(f"State file not found: {state_file}")
        
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        # Restore workflow state
        self.current_phase = state.get("current_phase")
        self.phase_results = state.get("phase_results", {})
        
        # Load agent states if available
        state_dir = os.path.dirname(state_file)
        for agent_name, agent in self.agents.items():
            agent_state_file = os.path.join(state_dir, f"{agent.__class__.__name__}_state.json")
            if os.path.isfile(agent_state_file):
                agent.load_state(agent_state_file)
        
        self.logger.log_info(f"Workflow state loaded from {state_file}")
    
    def perform_research(self, specific_phase: Optional[str] = None) -> Dict[str, Any]:
        """Perform the research workflow.
        
        Args:
            specific_phase: Optional specific phase to run
            
        Returns:
            Dict[str, Any]: Results of the research workflow
        """
        start_time = time.time()
        self.logger.log_info(f"Starting research on topic: {self.config.research_topic}")
        
        # Load state if specified
        if self.config.load_state:
            try:
                self._load_state(self.config.load_state)
                self.logger.log_info(f"Resumed from state file: {self.config.load_state}")
            except Exception as e:
                self.logger.log_error(f"Failed to load state: {e}")
        
        try:
            # Run either a specific phase or the full workflow
            if specific_phase:
                if specific_phase not in self.DEFAULT_PHASES:
                    self.logger.log_warning(f"Unknown phase: {specific_phase}, valid phases: {self.DEFAULT_PHASES}")
                    return {"status": "error", "message": f"Unknown phase: {specific_phase}"}
                
                self.logger.log_info(f"Running specific phase: {specific_phase}")
                result = self._run_phase(specific_phase)
                self.phase_results[specific_phase] = result
            else:
                self.logger.log_info("Running full research workflow")
                for phase in self.DEFAULT_PHASES:
                    # Skip phases that have already been completed (in case of resuming)
                    if phase in self.phase_results:
                        self.logger.log_info(f"Skipping already completed phase: {phase}")
                        continue
                    
                    result = self._run_phase(phase)
                    self.phase_results[phase] = result
                    
                    # Save state after each phase
                    self._save_state()
            
            # Generate visualizations and report
            self._generate_visualizations()
            
            # Log completion
            end_time = time.time()
            duration = end_time - start_time
            self.logger.log_info(f"Research completed in {duration:.2f} seconds")
            self.logger.log_metric("total_duration", duration)
            
            return {
                "status": "success",
                "topic": self.config.research_topic,
                "phases": self.phase_results,
                "duration": duration
            }
            
        except Exception as e:
            self.logger.log_error(f"Error during research: {e}")
            
            # Try to save state for recovery
            try:
                self._save_state()
            except Exception as save_error:
                self.logger.log_error(f"Failed to save state after error: {save_error}")
            
            return {
                "status": "error",
                "message": str(e),
                "phases": self.phase_results
            }
    
    def _run_phase(self, phase: str) -> Dict[str, Any]:
        """Run a specific research phase.
        
        Args:
            phase: The phase to run
            
        Returns:
            Dict[str, Any]: Results of the phase
        """
        self.current_phase = phase
        self.logger.log_phase_start(phase)
        
        start_time = time.time()
        
        # Determine which agent should handle this phase
        agent = self._get_agent_for_phase(phase)
        
        # Run the agent for this phase
        try:
            result = agent.run(phase)
            status = "success"
        except Exception as e:
            self.logger.log_error(f"Error in phase {phase}: {e}")
            result = {"error": str(e)}
            status = "error"
        
        # Calculate phase duration
        end_time = time.time()
        duration = end_time - start_time
        
        # Log phase completion
        self.logger.log_phase_end(phase, status=status)
        self.logger.log_phase_metric(phase, "duration", duration)
        
        return {
            "status": status,
            "result": result,
            "duration": duration
        }
    
    def _get_agent_for_phase(self, phase: str) -> BaseAgent:
        """Get the appropriate agent for a phase.
        
        Args:
            phase: The phase to get an agent for
            
        Returns:
            BaseAgent: The agent for the phase
            
        Raises:
            ValueError: If no agent is available for the phase
        """
        # Simple phase-to-agent mapping
        # For now, the professor agent handles everything
        if phase in ["plan-formulation", "report-writing"]:
            return self.agents["professor"]
        
        # Default to professor agent for all phases until more agents are implemented
        return self.agents["professor"]
    
    def _generate_visualizations(self) -> None:
        """Generate visualizations for the research workflow."""
        self.logger.log_info("Generating visualizations")
        
        try:
            # Generate phase duration chart
            self.visualizer.plot_phase_durations()
            
            # Generate phase steps chart
            self.visualizer.plot_phase_steps()
            
            # Generate HTML report
            self.visualizer.generate_html_report()
            
            self.logger.log_info("Visualizations generated successfully")
        except Exception as e:
            self.logger.log_error(f"Error generating visualizations: {e}")
    
    def handle_human_feedback(self, feedback: str, phase: Optional[str] = None) -> Dict[str, Any]:
        """Handle human feedback during the research process.
        
        Args:
            feedback: Human feedback text
            phase: Optional phase the feedback is for (defaults to current phase)
            
        Returns:
            Dict[str, Any]: Result of processing the feedback
        """
        if not phase:
            phase = self.current_phase or self.DEFAULT_PHASES[0]
        
        self.logger.log_info(f"Received human feedback for phase {phase}")
        self.logger.log_human_feedback(feedback, agent_name="human")
        
        # Get the appropriate agent for the phase
        agent = self._get_agent_for_phase(phase)
        
        # Process feedback with the agent
        try:
            result = agent.run(phase, feedback=feedback)
            status = "success"
        except Exception as e:
            self.logger.log_error(f"Error processing feedback: {e}")
            result = {"error": str(e)}
            status = "error"
        
        return {
            "status": status,
            "phase": phase,
            "result": result
        }
    
    def log_metrics(self, metrics: Dict[str, Any], phase: Optional[str] = None) -> None:
        """Log metrics for the research workflow.
        
        Args:
            metrics: Metrics to log
            phase: Optional phase to associate metrics with
        """
        if phase:
            for name, value in metrics.items():
                self.logger.log_phase_metric(phase, name, value)
        else:
            for name, value in metrics.items():
                self.logger.log_metric(name, value) 