"""
LLM Interface for the Agent Laboratory.

This module provides a unified interface for interacting with various language models,
abstracting away provider-specific details.
"""

import os
import json
import time
import logging
import requests
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from abc import ABC, abstractmethod
import openai
import tiktoken
import google.generativeai as genai
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

class LLMInterface:
    """Interface for interacting with language models."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the LLM interface.
        
        Args:
            model: Name of the model to use
            api_key: API key for accessing the model
            deepseek_api_key: API key for DeepSeek models
            config: Additional configuration
        """
        self.model = model
        self.config = config or {}
        
        # Initialize the API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.deepseek_api_key = deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY")
        
        if not self.api_key and not self.deepseek_api_key:
            logger.error("No API key provided for LLM interface")
        else:
            api_type = "OpenAI API key" if self.api_key else "DeepSeek API key"
            api_key_length = len(self.api_key) if self.api_key else len(self.deepseek_api_key)
            logger.info(f"LLM interface initialized with {api_type} (length: {api_key_length})")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text based on the prompt.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for the model
            
        Returns:
            The generated text
        """
        if "deepseek" in self.model.lower() and self.deepseek_api_key:
            return self._generate_deepseek(prompt, **kwargs)
        else:
            return self._generate_openai(prompt, **kwargs)
    
    def _generate_openai(self, prompt: str, **kwargs) -> str:
        """Generate text using the OpenAI API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for the model
            
        Returns:
            The generated text
        """
        try:
            import openai
            
            # Configure the client
            if self.api_key:
                client = openai.OpenAI(api_key=self.api_key)
                
                # Check for API key validity
                if not self.api_key.startswith("sk-"):
                    logger.warning(f"OpenAI API key may be invalid (doesn't start with 'sk-')")
                
                # Set default parameters
                model = kwargs.get("model", self.model)
                temperature = kwargs.get("temperature", 0.4)
                max_tokens = kwargs.get("max_tokens", 16000)
                
                messages = [
                    {"role": "system", "content": kwargs.get("system_message", "You are a helpful assistant.")},
                    {"role": "user", "content": prompt}
                ]
                
                # Estimate token count for input
                try:
                    import tiktoken
                    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
                    system_tokens = len(encoding.encode(messages[0]["content"]))
                    user_tokens = len(encoding.encode(prompt))
                    total_input_tokens = system_tokens + user_tokens
                    logger.info(f"Estimated token usage - System: {system_tokens}, User: {user_tokens}, Total: {total_input_tokens}")
                except Exception as e:
                    logger.warning(f"Failed to estimate token count: {str(e)}")
                
                logger.info(f"Calling OpenAI API with model={model}, temp={temperature}, max_tokens={max_tokens}")
                logger.debug(f"Input prompt size: {len(prompt)} characters")
                
                # Log calling location to help with debugging
                import traceback
                caller_info = "Unknown"
                try:
                    stack = traceback.extract_stack()
                    if len(stack) >= 2:
                        caller = stack[-2]
                        caller_info = f"{caller.filename}:{caller.lineno} in {caller.name}"
                except Exception:
                    pass
                logger.info(f"OpenAI API call initiated from: {caller_info}")
                
                # Make the API call
                start_time = time.time()
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                call_duration = time.time() - start_time
                
                # Extract response and log
                content = response.choices[0].message.content
                
                # Log token usage if available
                usage = getattr(response, "usage", None)
                if usage:
                    prompt_tokens = getattr(usage, "prompt_tokens", 0)
                    completion_tokens = getattr(usage, "completion_tokens", 0)
                    total_tokens = getattr(usage, "total_tokens", 0)
                    logger.info(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
                
                logger.info(f"OpenAI API call completed in {call_duration:.2f}s, response size: {len(content)} characters")
                
                if kwargs.get("save_response", False):
                    # Save response to file for debugging if requested
                    output_dir = kwargs.get("output_dir", "outputs")
                    import os
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"{output_dir}/openai_response_{timestamp}.json"
                    
                    with open(filename, "w") as f:
                        json.dump({
                            "model": model,
                            "request": [m["content"] for m in messages],
                            "response": content,
                            "usage": {
                                "prompt_tokens": prompt_tokens if usage else 0,
                                "completion_tokens": completion_tokens if usage else 0,
                                "total_tokens": total_tokens if usage else 0
                            },
                            "duration": call_duration
                        }, f, indent=2)
                    
                    logger.info(f"Saved API response to {filename}")
                
                return content
            else:
                logger.error("No OpenAI API key provided")
                return "ERROR: OpenAI API key is missing"
        
        except ImportError:
            logger.error("Failed to import openai module. Make sure it's installed: pip install openai")
            return "ERROR: OpenAI module not installed"
        
        except Exception as e:
            logger.error(f"Error generating text with OpenAI API: {str(e)}")
            return f"ERROR: {str(e)}"
    
    def _generate_deepseek(self, prompt: str, **kwargs) -> str:
        """Generate text using the DeepSeek API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for the model
            
        Returns:
            The generated text
        """
        try:
            import deepseek
            
            # Configure the client
            if self.deepseek_api_key:
                client = deepseek.DeepSeek(api_key=self.deepseek_api_key)
                
                # Set default parameters
                model = kwargs.get("model", self.model)
                temperature = kwargs.get("temperature", 0.7)
                max_tokens = kwargs.get("max_tokens", 16000)
                
                messages = [
                    {"role": "system", "content": kwargs.get("system_message", "You are a helpful assistant.")},
                    {"role": "user", "content": prompt}
                ]
                
                # Log input data
                logger.info(f"Calling DeepSeek API with model={model}, temp={temperature}, max_tokens={max_tokens}")
                logger.debug(f"Input prompt size: {len(prompt)} characters")
                
                # Estimate token count
                system_chars = len(messages[0]["content"])
                user_chars = len(prompt)
                estimated_tokens = (system_chars + user_chars) // 4  # Rough estimate
                logger.info(f"Estimated input token usage (approximate): {estimated_tokens}")
                
                # Log calling location to help with debugging
                import traceback
                caller_info = "Unknown"
                try:
                    stack = traceback.extract_stack()
                    if len(stack) >= 2:
                        caller = stack[-2]
                        caller_info = f"{caller.filename}:{caller.lineno} in {caller.name}"
                except Exception:
                    pass
                logger.info(f"DeepSeek API call initiated from: {caller_info}")
                
                # Make the API call
                start_time = time.time()
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                call_duration = time.time() - start_time
                
                # Extract response and log
                content = response.choices[0].message.content
                logger.info(f"DeepSeek API call completed in {call_duration:.2f}s, response size: {len(content)} characters")
                
                if kwargs.get("save_response", False):
                    # Save response to file for debugging if requested
                    output_dir = kwargs.get("output_dir", "outputs")
                    import os
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"{output_dir}/deepseek_response_{timestamp}.json"
                    
                    with open(filename, "w") as f:
                        json.dump({
                            "model": model,
                            "request": [m["content"] for m in messages],
                            "response": content,
                            "estimated_tokens": estimated_tokens,
                            "duration": call_duration
                        }, f, indent=2)
                    
                    logger.info(f"Saved API response to {filename}")
                
                return content
            else:
                logger.error("No DeepSeek API key provided")
                return "ERROR: DeepSeek API key is missing"
        
        except ImportError:
            logger.error("Failed to import deepseek module. Make sure it's installed: pip install deepseek")
            return "ERROR: DeepSeek module not installed"
        
        except Exception as e:
            logger.error(f"Error generating text with DeepSeek API: {str(e)}")
            return f"ERROR: {str(e)}"
    
    def completion(self, prompt: str, system_message: str = "You are a helpful assistant.", temperature: float = 0.7) -> str:
        """Generate a completion for the prompt.
        
        Args:
            prompt: The input prompt
            system_message: System message for the model
            temperature: Temperature for sampling
            
        Returns:
            The generated completion
        """
        return self.generate(
            prompt=prompt,
            system_message=system_message,
            temperature=temperature
        )


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def run_inference(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run inference with the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that stop generation
            
        Returns:
            Response dictionary with 'content' and usage statistics
        """
        pass


class OpenAIBackend(LLMBackend):
    """Backend for OpenAI models."""
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float = 0.7,
        timeout: int = 120
    ):
        """
        Initialize the OpenAI backend.
        
        Args:
            model_name: Name of the OpenAI model
            api_key: OpenAI API key
            temperature: Temperature for generation
            timeout: Timeout for API requests in seconds
        """
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=api_key, timeout=timeout)
    
    def run_inference(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run inference with OpenAI.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that stop generation
            
        Returns:
            Response dictionary with 'content' and usage statistics
        """
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        
        if stop_sequences:
            kwargs["stop"] = stop_sequences
        
        response = self.client.chat.completions.create(**kwargs)
        
        # Convert to standard format
        result = {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "model": response.model,
            "id": response.id
        }
        
        return result


class DeepSeekBackend(LLMBackend):
    """Backend for DeepSeek models."""
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float = 0.7,
        timeout: int = 120
    ):
        """
        Initialize the DeepSeek backend.
        
        Args:
            model_name: Name of the DeepSeek model
            api_key: DeepSeek API key
            temperature: Temperature for generation
            timeout: Timeout for API requests in seconds
        """
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        
        # DeepSeek API endpoint
        self.api_endpoint = "https://api.deepseek.com/v1/chat/completions"
    
    def run_inference(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run inference with DeepSeek.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that stop generation
            
        Returns:
            Response dictionary with 'content' and usage statistics
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        response = requests.post(
            self.api_endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"DeepSeek API error: {response.status_code} - {response.text}")
        
        response_json = response.json()
        
        # Convert to standard format
        result = {
            "content": response_json["choices"][0]["message"]["content"],
            "usage": response_json["usage"],
            "model": response_json["model"],
            "id": response_json["id"]
        }
        
        return result


class LLMBackendFactory:
    """Factory for creating LLM backends."""
    
    def __init__(self):
        """Initialize the backend factory."""
        self.backends = {}
        
        # Register built-in backends
        self.register_backend("openai", self._create_openai_backend)
        self.register_backend("deepseek", self._create_deepseek_backend)
    
    def register_backend(self, provider: str, factory_fn: Callable) -> None:
        """
        Register a new backend provider.
        
        Args:
            provider: Name of the provider
            factory_fn: Factory function to create the backend
        """
        self.backends[provider] = factory_fn
    
    def get_backend(
        self,
        model_name: str,
        api_key: str,
        temperature: float = 0.7,
        timeout: int = 120
    ) -> LLMBackend:
        """
        Get a backend for the specified model.
        
        Args:
            model_name: Name of the model
            api_key: API key for the provider
            temperature: Temperature for generation
            timeout: Timeout for API requests in seconds
            
        Returns:
            LLM backend instance
            
        Raises:
            ValueError: If no backend is available for the model
        """
        model_lower = model_name.lower()
        
        if "gpt" in model_lower or "claude" in model_lower or "llama" in model_lower or "mistral" in model_lower:
            return self.backends["openai"](model_name, api_key, temperature, timeout)
        elif "deepseek" in model_lower:
            return self.backends["deepseek"](model_name, api_key, temperature, timeout)
        else:
            # Default to OpenAI
            return self.backends["openai"](model_name, api_key, temperature, timeout)
    
    def _create_openai_backend(
        self,
        model_name: str,
        api_key: str,
        temperature: float = 0.7,
        timeout: int = 120
    ) -> OpenAIBackend:
        """
        Create an OpenAI backend.
        
        Args:
            model_name: Name of the OpenAI model
            api_key: OpenAI API key
            temperature: Temperature for generation
            timeout: Timeout for API requests in seconds
            
        Returns:
            OpenAI backend instance
        """
        return OpenAIBackend(model_name, api_key, temperature, timeout)
    
    def _create_deepseek_backend(
        self,
        model_name: str,
        api_key: str,
        temperature: float = 0.7,
        timeout: int = 120
    ) -> DeepSeekBackend:
        """
        Create a DeepSeek backend.
        
        Args:
            model_name: Name of the DeepSeek model
            api_key: DeepSeek API key
            temperature: Temperature for generation
            timeout: Timeout for API requests in seconds
            
        Returns:
            DeepSeek backend instance
        """
        return DeepSeekBackend(model_name, api_key, temperature, timeout)

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens
        """
        # Use tiktoken for OpenAI models
        if self.model_name in ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3-mini"]:
            encoding = tiktoken.encoding_for_model("gpt-4o-mini")  # Use gpt-4o-mini encoding
            return len(encoding.encode(text))
        
        # For other models, use an approximate token count
        # This is a rough approximation - about 4 characters per token
        return len(text) // 4
    
    def clip_tokens(self, text: str, max_tokens: int = 16000) -> str:
        """Clip text to a maximum number of tokens.
        
        Args:
            text: Text to clip
            max_tokens: Maximum number of tokens
            
        Returns:
            Clipped text
        """
        current_tokens = self.count_tokens(text)
        
        if current_tokens <= max_tokens:
            return text
        
        # For OpenAI models use tiktoken for accurate clipping
        if self.model_name in ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3-mini"]:
            encoding = tiktoken.encoding_for_model("gpt-4o-mini")
            tokens = encoding.encode(text)
            # Get just the last max_tokens
            clipped_tokens = tokens[-max_tokens:]
            return encoding.decode(clipped_tokens)
        
        # For other models, use approximate clipping based on characters
        # About 4 characters per token
        char_limit = max_tokens * 4
        return text[-char_limit:] 