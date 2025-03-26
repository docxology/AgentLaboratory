"""Tests for the workflow module."""

import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import yaml

from agent_lab.core.workflow import LaboratoryWorkflow
from agent_lab.core.llm_interface import LLMInterface
from agent_lab.agents import ProfessorAgent


class TestLaboratoryWorkflow(unittest.TestCase):
    """Test case for the LaboratoryWorkflow class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create a test configuration
        self.config = {
            "research_topic": "Test research topic",
            "llm_backend": "gpt-4o-mini",
            "api_key": "test-api-key",
            "copilot_mode": False,
            "max_steps": 2,
            "output_dir": self.output_dir
        }
        
        # Create a YAML file with the configuration
        self.yaml_path = os.path.join(self.temp_dir.name, "test_config.yaml")
        with open(self.yaml_path, "w") as f:
            yaml.dump(self.config, f)
        
        # Mock the LLMInterface
        self.llm_patcher = patch('agent_lab.core.llm_interface.LLMInterface')
        self.mock_llm = self.llm_patcher.start()
        self.mock_llm_instance = MagicMock()
        self.mock_llm.return_value = self.mock_llm_instance
        self.mock_llm_instance.completion.return_value = "Test response"
        
        # Create a workflow with the test configuration
        self.workflow = LaboratoryWorkflow(config=self.config)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up the temporary directory
        self.temp_dir.cleanup()
        
        # Stop the LLMInterface patcher
        self.llm_patcher.stop()
    
    def test_init(self):
        """Test initialization of the workflow."""
        # Test that the configuration is set correctly
        self.assertEqual(self.workflow.config, self.config)
        
        # Test that the output directory is created
        self.assertTrue(os.path.exists(self.output_dir))
        
        # Test that the LLM interface is initialized
        self.assertEqual(self.workflow.llm.model, "gpt-4o-mini")
        self.assertEqual(self.workflow.llm.api_key, "test-api-key")
    
    def test_create_agent(self):
        """Test creation of an agent."""
        # Create a professor agent
        agent = self.workflow.create_agent("professor")
        
        # Test that the agent is created correctly
        self.assertIsInstance(agent, ProfessorAgent)
        self.assertEqual(agent.llm, self.workflow.llm)
    
    @patch('agent_lab.core.workflow.os.makedirs')
    def test_initialize_output_directory(self, mock_makedirs):
        """Test initialization of the output directory."""
        # Set a custom output directory
        self.workflow.config['output_dir'] = "/test/output"
        
        # Initialize the output directory
        output_dir = self.workflow.initialize_output_directory()
        
        # Test that the output directory is created
        mock_makedirs.assert_called_with("/test/output", exist_ok=True)
        self.assertEqual(output_dir, "/test/output")
    
    @patch('agent_lab.core.workflow.logging')
    def test_setup_logging(self, mock_logging):
        """Test setup of logging."""
        # Setup logging
        log_file = self.workflow.setup_logging()
        
        # Test that the log file is created in the output directory
        self.assertTrue(log_file.startswith(self.output_dir))
        self.assertTrue(log_file.endswith(".log"))
        
        # Test that the logging is configured
        mock_logging.basicConfig.assert_called_once()
    
    @patch('agent_lab.core.workflow.LaboratoryWorkflow.run_step')
    def test_run(self, mock_run_step):
        """Test running the workflow."""
        # Set the mock run_step to return False (completed)
        mock_run_step.return_value = False
        
        # Run the workflow
        self.workflow.run()
        
        # Test that run_step is called the correct number of times
        self.assertEqual(mock_run_step.call_count, 1)
    
    @patch('agent_lab.core.workflow.LaboratoryWorkflow.save_state')
    def test_run_step(self, mock_save_state):
        """Test running a step of the workflow."""
        # Mock the agent
        mock_agent = MagicMock()
        self.workflow.agents["professor"] = mock_agent
        mock_agent.step.return_value = "Test result"
        
        # Run a step
        result = self.workflow.run_step()
        
        # Test that the agent step is called
        mock_agent.step.assert_called_once()
        
        # Test that the state is saved
        mock_save_state.assert_called_once()
        
        # Test that the result is returned
        self.assertTrue(result)
    
    def test_from_yaml(self):
        """Test creating a workflow from a YAML file."""
        # Create a workflow from the YAML file
        workflow = LaboratoryWorkflow.from_yaml(self.yaml_path)
        
        # Test that the configuration is loaded correctly
        self.assertEqual(workflow.config["research_topic"], "Test research topic")
        self.assertEqual(workflow.config["llm_backend"], "gpt-4o-mini")
        self.assertEqual(workflow.config["api_key"], "test-api-key")
        self.assertEqual(workflow.config["copilot_mode"], False)
        self.assertEqual(workflow.config["max_steps"], 2)
        self.assertEqual(workflow.config["output_dir"], self.output_dir)


if __name__ == '__main__':
    unittest.main() 