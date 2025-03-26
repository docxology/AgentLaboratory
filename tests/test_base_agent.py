"""Tests for the base agent module."""

import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile

from agent_lab.core.base_agent import BaseAgent


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""
    
    def role_description(self) -> str:
        """Get the role description for the agent."""
        return "Test Agent"
    
    def command_descriptions(self, phase: str) -> str:
        """Get the command descriptions for a specific phase."""
        return "Test Commands"
    
    def example_command(self, phase: str) -> str:
        """Get an example command for a specific phase."""
        return "Test Example"
    
    def context(self, phase: str) -> str:
        """Get context for a specific phase."""
        return "Test Context"
    
    def phase_prompt(self, phase: str) -> str:
        """Get the prompt for a specific phase."""
        return "Test Prompt"


class TestBaseAgent(unittest.TestCase):
    """Test case for the BaseAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Mock the LLMInterface
        self.llm_patcher = patch('agent_lab.core.llm_interface.LLMInterface')
        self.mock_llm = self.llm_patcher.start()
        self.mock_llm_instance = MagicMock()
        self.mock_llm.return_value = self.mock_llm_instance
        self.mock_llm_instance.completion.return_value = "Test response"
        
        # Mock logger
        self.mock_logger = MagicMock()
        
        # Create a configuration
        self.config = {
            "research_topic": "Test research topic",
            "output_dir": self.output_dir,
            "copilot_mode": False
        }
        
        # Create an agent with the mock LLM
        self.agent = ConcreteAgent(
            model="gpt-4o-mini",
            notes={"test": "note"},
            max_steps=5,
            api_key="test-api-key",
            logger=self.mock_logger
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up the temporary directory
        self.temp_dir.cleanup()
        
        # Stop the LLM patcher
        self.llm_patcher.stop()
    
    def test_init(self):
        """Test initialization of the agent."""
        # Test that the model is set correctly
        self.assertEqual(self.agent.model, "gpt-4o-mini")
        
        # Test that the notes are set correctly
        self.assertEqual(self.agent.notes, {"test": "note"})
        
        # Test that the max_steps is set correctly
        self.assertEqual(self.agent.max_steps, 5)
        
        # Test that the api_key is set correctly
        self.assertEqual(self.agent.api_key, "test-api-key")
        
        # Test that the logger is set correctly
        self.assertEqual(self.agent.logger, self.mock_logger)
        
        # Test that the step_count is initialized to 0
        self.assertEqual(self.agent.step_count, 0)
        
        # Test that the history is initialized as an empty list
        self.assertEqual(self.agent.history, [])
    
    def test_get_system_message(self):
        """Test getting the system message."""
        # Get the system message
        system_message = self.agent.get_system_message("test_phase")
        
        # Test that the system message contains the role description
        self.assertIn("Test Agent", system_message)
        
        # Test that the system message contains the phase
        self.assertIn("test_phase", system_message)
        
        # Test that the system message contains the command descriptions
        self.assertIn("Test Commands", system_message)
        
        # Test that the system message contains the example command
        self.assertIn("Test Example", system_message)
    
    def test_execute(self):
        """Test executing the agent."""
        # Mock the get_system_message method
        with patch.object(self.agent, 'get_system_message', return_value="Test system message"):
            # Execute the agent
            result = self.agent.execute("Test prompt", "test_phase")
            
            # Test that the result contains the success status
            self.assertEqual(result["status"], "success")
            
            # Test that the result contains the response
            self.assertEqual(result["response"], "Test response")
            
            # Test that the result contains the step
            self.assertEqual(result["step"], 1)
            
            # Test that the step_count is incremented
            self.assertEqual(self.agent.step_count, 1)
            
            # Test that the history is updated
            self.assertEqual(len(self.agent.history), 1)
            self.assertEqual(self.agent.history[0]["phase"], "test_phase")
            self.assertEqual(self.agent.history[0]["step"], 1)
            self.assertEqual(self.agent.history[0]["prompt"], "Test prompt\n\nTest Context")
            self.assertEqual(self.agent.history[0]["system_message"], "Test system message")
            self.assertEqual(self.agent.history[0]["response"], "Test response")
    
    def test_max_steps(self):
        """Test that the agent stops after reaching max_steps."""
        # Set step_count to max_steps
        self.agent.step_count = self.agent.max_steps
        
        # Execute the agent
        result = self.agent.execute("Test prompt", "test_phase")
        
        # Test that the result contains the max_steps_reached status
        self.assertEqual(result["status"], "max_steps_reached")
    
    def test_save_state(self):
        """Test saving the agent's state."""
        # Add some history
        self.agent.history = [{"test": "history"}]
        self.agent.step_count = 3
        self.agent.current_phase = "test_phase"
        
        # Save the state
        self.agent.save_state(self.output_dir)
        
        # Test that the state file is created
        state_file = os.path.join(self.output_dir, "ConcreteAgent_state.json")
        self.assertTrue(os.path.exists(state_file))
        
        # Test the content of the state file
        with open(state_file, "r") as f:
            state = f.read()
            self.assertIn("gpt-4o-mini", state)
            self.assertIn("3", state)
            self.assertIn("test_phase", state)
            self.assertIn("history", state)
    
    def test_override_inference(self):
        """Test overriding inference."""
        # Override inference
        response = self.agent.override_inference("Test query")
        
        # Test that the response is returned
        self.assertEqual(response, "Test response")
        
        # Test that the LLM completion is called with the right parameters
        self.mock_llm_instance.completion.assert_called_with(
            prompt="Test query",
            temperature=0.7
        )


if __name__ == '__main__':
    unittest.main() 