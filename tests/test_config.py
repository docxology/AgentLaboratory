"""Tests for the configuration loader module."""

import os
import tempfile
import unittest
import yaml
from unittest.mock import patch, MagicMock

from agent_lab.config.loader import load_config, load_from_yaml, load_from_args, load_from_env


class TestConfigLoader(unittest.TestCase):
    """Test case for the configuration loader."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary YAML file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.yaml_path = os.path.join(self.temp_dir.name, "test_config.yaml")
        
        self.test_config = {
            "research_topic": "Test Research",
            "llm_backend": "test-model",
            "api_key": "test-api-key",
            "copilot_mode": True,
            "max_steps": 5
        }
        
        with open(self.yaml_path, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_load_from_yaml(self):
        """Test loading configuration from a YAML file."""
        config = load_from_yaml(self.yaml_path)
        
        self.assertEqual(config["research_topic"], "Test Research")
        self.assertEqual(config["llm_backend"], "test-model")
        self.assertEqual(config["api_key"], "test-api-key")
        self.assertEqual(config["copilot_mode"], True)
        self.assertEqual(config["max_steps"], 5)
    
    def test_load_from_args(self):
        """Test loading configuration from command-line arguments."""
        args = MagicMock()
        args.yaml_location = self.yaml_path
        args.llm_backend = "override-model"
        args.debug = True
        
        config = load_from_args(args)
        
        self.assertEqual(config["research_topic"], "Test Research")
        self.assertEqual(config["llm_backend"], "override-model")  # Should be overridden
        self.assertEqual(config["api_key"], "test-api-key")
        self.assertEqual(config["copilot_mode"], True)
        self.assertEqual(config["max_steps"], 5)
        self.assertEqual(config["debug"], True)
    
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        with patch.dict('os.environ', {
            'AGENT_LAB_LLM_BACKEND': 'env-model',
            'OPENAI_API_KEY': 'env-api-key',
            'AGENT_LAB_DEBUG': 'true'
        }):
            config = load_from_env()
            
            self.assertEqual(config["llm_backend"], "env-model")
            self.assertEqual(config["api_key"], "env-api-key")
            self.assertEqual(config["debug"], True)
    
    def test_load_config_function(self):
        """Test the load_config convenience function."""
        # Test with string path
        config = load_config(self.yaml_path)
        self.assertEqual(config["research_topic"], "Test Research")
        
        # Test with args object
        args = MagicMock()
        args.yaml_location = self.yaml_path
        args.llm_backend = "override-model"
        config = load_config(args)
        self.assertEqual(config["research_topic"], "Test Research")
        self.assertEqual(config["llm_backend"], "override-model")
        
        # Test with None (should use environment)
        with patch('agent_lab.config.loader.load_from_env', return_value={"research_topic": "Env Research"}):
            config = load_config(None)
            self.assertEqual(config["research_topic"], "Env Research")


if __name__ == '__main__':
    unittest.main() 