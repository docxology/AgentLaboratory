"""Tests for the LLM interface module."""

import unittest
from unittest.mock import patch, MagicMock
import json

from agent_lab.core.llm_interface import LLMInterface


class TestLLMInterface(unittest.TestCase):
    """Test case for the LLMInterface class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use a mock API key for testing
        self.api_key = "test-api-key"
        
        # Create an LLM interface with a test model
        with patch('agent_lab.core.llm_interface.requests'):
            self.llm = LLMInterface(
                model="gpt-4o-mini",
                api_key=self.api_key
            )
    
    def test_get_model_provider(self):
        """Test getting the model provider."""
        # Test OpenAI model
        self.llm.model = "gpt-4o-mini"
        self.assertEqual(self.llm.get_model_provider(), "openai")
        
        # Test DeepSeek model
        self.llm.model = "deepseek-chat"
        self.assertEqual(self.llm.get_model_provider(), "deepseek")
        
        # Test unknown model
        self.llm.model = "unknown-model"
        self.assertEqual(self.llm.get_model_provider(), "unknown")
    
    @patch('agent_lab.core.llm_interface.requests.post')
    def test_openai_completion(self, mock_post):
        """Test OpenAI completion."""
        # Setup the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Test response"
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        # Set to OpenAI model
        self.llm.model = "gpt-4o-mini"
        
        # Test the completion
        response = self.llm.completion(
            prompt="Test prompt",
            system_message="Test system message"
        )
        
        # Assertions
        self.assertEqual(response, "Test response")
        mock_post.assert_called_once()
        
        # Check that the correct data was sent
        call_args = mock_post.call_args[1]
        payload = json.loads(call_args['data']) if 'data' in call_args else call_args['json']
        
        self.assertEqual(payload['model'], "gpt-4o-mini")
        self.assertEqual(len(payload['messages']), 2)
        self.assertEqual(payload['messages'][0]['role'], "system")
        self.assertEqual(payload['messages'][0]['content'], "Test system message")
        self.assertEqual(payload['messages'][1]['role'], "user")
        self.assertEqual(payload['messages'][1]['content'], "Test prompt")
    
    @patch('agent_lab.core.llm_interface.requests.post')
    def test_deepseek_completion(self, mock_post):
        """Test DeepSeek completion."""
        # Setup the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Test response"
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        # Set to DeepSeek model and API key
        self.llm.model = "deepseek-chat"
        self.llm.deepseek_api_key = "test-deepseek-key"
        
        # Test the completion
        response = self.llm.completion(
            prompt="Test prompt",
            system_message="Test system message"
        )
        
        # Assertions
        self.assertEqual(response, "Test response")
        mock_post.assert_called_once()
        
        # Check that the correct data was sent
        call_args = mock_post.call_args[1]
        payload = json.loads(call_args['data']) if 'data' in call_args else call_args['json']
        
        self.assertEqual(payload['model'], "deepseek-chat")
        self.assertEqual(len(payload['messages']), 2)
        self.assertEqual(payload['messages'][0]['role'], "system")
        self.assertEqual(payload['messages'][0]['content'], "Test system message")
        self.assertEqual(payload['messages'][1]['role'], "user")
        self.assertEqual(payload['messages'][1]['content'], "Test prompt")
    
    @patch('agent_lab.core.llm_interface.requests.post')
    def test_completion_error(self, mock_post):
        """Test error handling in completion."""
        # Setup the mock response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Error message"
        mock_post.return_value = mock_response
        
        # Set to OpenAI model
        self.llm.model = "gpt-4o-mini"
        
        # Test the completion with error
        with self.assertRaises(Exception) as context:
            self.llm.completion(prompt="Test prompt")
        
        # Check the error message
        self.assertIn("OpenAI API error", str(context.exception))
        self.assertIn("400", str(context.exception))
        self.assertIn("Error message", str(context.exception))
    
    def test_invalid_model_error(self):
        """Test error when invalid model is specified."""
        # Test with invalid provider
        self.llm.model = "unknown-model"
        
        with self.assertRaises(ValueError) as context:
            self.llm.completion(prompt="Test prompt")
        
        self.assertIn("Unsupported model provider", str(context.exception))


if __name__ == '__main__':
    unittest.main() 