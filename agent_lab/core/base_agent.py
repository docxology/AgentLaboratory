"""Base agent for the Agent Laboratory.

This module defines the BaseAgent class that all agents inherit from,
providing common functionality and interfaces.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import os
import json

from agent_lab.core.llm_interface import LLMInterface

class BaseAgent(ABC):
    """Base class for all agents in the Agent Laboratory."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        notes: Optional[Dict[str, Any]] = None,
        max_steps: int = 10,
        api_key: Optional[str] = None,
        logger: Optional[Any] = None
    ):
        """Initialize the base agent.
        
        Args:
            model: The model to use
            notes: Notes for the research task
            max_steps: Maximum number of steps to take
            api_key: API key for the LLM provider
            logger: Logger instance
        """
        self.model = model
        self.notes = notes or {}
        self.max_steps = max_steps
        self.api_key = api_key
        self.logger = logger
        
        # Initialize LLM interface
        self.llm = LLMInterface(model=model, api_key=api_key)
        
        # Initialize state
        self.step_count = 0
        self.history = []
        self.current_phase = None
    
    def run(self, phase: str, **kwargs) -> Dict[str, Any]:
        """Run the agent for a specific phase.
        
        Args:
            phase: The phase to run
            **kwargs: Additional keyword arguments
            
        Returns:
            Dict[str, Any]: The result of the agent's execution
        """
        self.current_phase = phase
        self.logger.info(f"Running agent for phase: {phase}")
        
        # Reset step count for each phase
        self.step_count = 0
        
        # Get the phase-specific prompt
        prompt = self.phase_prompt(phase)
        
        # Run the agent-specific logic
        result = self.execute(prompt, phase, **kwargs)
        
        # Log the result
        self.logger.info(f"Agent completed phase: {phase}")
        
        return result
    
    def execute(self, prompt: str, phase: str, **kwargs) -> Dict[str, Any]:
        """Execute the agent with a specific prompt.
        
        Args:
            prompt: The prompt to use
            phase: The current phase
            **kwargs: Additional keyword arguments
            
        Returns:
            Dict[str, Any]: The result of the agent's execution
        """
        system_message = self.get_system_message(phase)
        
        # Build context with phase-specific information
        context = self.context(phase)
        if context:
            prompt = f"{prompt}\n\n{context}"
        
        # Check if we've reached the max step count
        if self.step_count >= self.max_steps:
            self.logger.warning(f"Reached maximum step count ({self.max_steps}) for phase: {phase}")
            return {"status": "max_steps_reached", "message": "Reached maximum step count"}
        
        self.step_count += 1
        
        # Run inference
        response = self.llm.completion(
            prompt=prompt,
            system_message=system_message
        )
        
        # Record interaction in history
        self.history.append({
            "phase": phase,
            "step": self.step_count,
            "prompt": prompt,
            "system_message": system_message,
            "response": response
        })
        
        return {
            "status": "success",
            "response": response,
            "step": self.step_count
        }
    
    def override_inference(self, query: str, temperature: float = 0.7) -> str:
        """Run a custom inference with a specific query.
        
        Args:
            query: The query to send to the model
            temperature: Temperature for sampling
            
        Returns:
            str: The model's response
        """
        return self.llm.completion(
            prompt=query,
            temperature=temperature
        )
    
    def get_system_message(self, phase: str) -> str:
        """Get the system message for a specific phase.
        
        Args:
            phase: The current phase
            
        Returns:
            str: The system message
        """
        role = self.role_description()
        examples = self.example_command(phase)
        
        system_message = f"""You are a {role}

Your task is to help with the current research phase: {phase}

{self.command_descriptions(phase)}

{examples}

Respond in a clear, concise, and helpful manner.
"""
        return system_message
    
    @abstractmethod
    def role_description(self) -> str:
        """Get the role description for the agent.
        
        Returns:
            str: The role description
        """
        pass
    
    @abstractmethod
    def command_descriptions(self, phase: str) -> str:
        """Get the command descriptions for a specific phase.
        
        Args:
            phase: The current phase
            
        Returns:
            str: The command descriptions
        """
        pass
    
    @abstractmethod
    def example_command(self, phase: str) -> str:
        """Get an example command for a specific phase.
        
        Args:
            phase: The current phase
            
        Returns:
            str: The example command
        """
        pass
    
    @abstractmethod
    def context(self, phase: str) -> str:
        """Get context for a specific phase.
        
        Args:
            phase: The current phase
            
        Returns:
            str: The context
        """
        pass
    
    @abstractmethod
    def phase_prompt(self, phase: str) -> str:
        """Get the prompt for a specific phase.
        
        Args:
            phase: The current phase
            
        Returns:
            str: The prompt
        """
        pass
    
    def save_state(self, output_dir: str) -> None:
        """Save the agent's state to a file.
        
        Args:
            output_dir: The output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        state = {
            "model": self.model,
            "step_count": self.step_count,
            "history": self.history,
            "current_phase": self.current_phase
        }
        
        state_file = os.path.join(output_dir, f"{self.__class__.__name__}_state.json")
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Agent state saved to {state_file}")
    
    def load_state(self, state_file: str) -> None:
        """Load the agent's state from a file.
        
        Args:
            state_file: The state file path
            
        Raises:
            FileNotFoundError: If the state file does not exist
        """
        if not os.path.isfile(state_file):
            raise FileNotFoundError(f"State file not found: {state_file}")
        
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self.model = state.get("model", self.model)
        self.step_count = state.get("step_count", 0)
        self.history = state.get("history", [])
        self.current_phase = state.get("current_phase")
        
        self.logger.info(f"Agent state loaded from {state_file}")
    
    def get_dialog(self) -> List[Dict[str, Any]]:
        """Get the agent's dialog history.
        
        Returns:
            List[Dict[str, Any]]: The dialog history
        """
        return self.history 