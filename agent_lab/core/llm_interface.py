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

class LLMInterface:
    """Interface for interacting with language models."""
    
    def __init__(
        self, 
        model: str = "gpt-4o-mini", 
        api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 120
    ):
        """Initialize the LLM interface.
        
        Args:
            model: The model to use (default: "gpt-4o-mini")
            api_key: OpenAI API key
            deepseek_api_key: DeepSeek API key
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum number of tokens to generate (default: 2048)
            timeout: Timeout for API requests in seconds (default: 120)
            
        Raises:
            ValueError: If no API key is provided for the selected model
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Use provided API keys or try to get from environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.deepseek_api_key = deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY")
        
        # Validate API keys based on model
        if "gpt" in self.model and not self.api_key:
            raise ValueError("OpenAI API key required for GPT models")
        elif "deepseek" in self.model and not self.deepseek_api_key:
            raise ValueError("DeepSeek API key required for DeepSeek models")
        
        # Initialize backend factory
        self.backend_factory = LLMBackendFactory()
        
        # Get appropriate backend
        self.backend = self.backend_factory.get_backend(
            model_name=model,
            api_key=self.api_key if "deepseek" not in model.lower() else self.deepseek_api_key,
            temperature=temperature,
            timeout=timeout
        )
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        
        # Track usage for metrics
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0
    
    def get_model_provider(self) -> str:
        """Get the provider for the current model.
        
        Returns:
            str: The model provider ("openai", "deepseek", or "unknown")
        """
        if "gpt" in self.model:
            return "openai"
        elif "deepseek" in self.model:
            return "deepseek"
        return "unknown"
    
    def completion(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Get a completion from the language model.
        
        Args:
            prompt: The prompt to send to the model
            system_message: Optional system message for chat models
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            stop_sequences: Optional list of stop sequences
            
        Returns:
            str: The model's response
            
        Raises:
            Exception: If the API request fails
        """
        # Use provided values or instance defaults
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        provider = self.get_model_provider()
        
        if provider == "openai":
            return self._openai_completion(
                prompt=prompt,
                system_message=system_message,
                temperature=temp,
                max_tokens=tokens,
                stop_sequences=stop_sequences
            )
        elif provider == "deepseek":
            return self._deepseek_completion(
                prompt=prompt,
                system_message=system_message,
                temperature=temp,
                max_tokens=tokens,
                stop_sequences=stop_sequences
            )
        else:
            raise ValueError(f"Unsupported model provider for {self.model}")
    
    def _openai_completion(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Get a completion from OpenAI.
        
        Args:
            prompt: The prompt to send to the model
            system_message: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Optional list of stop sequences
            
        Returns:
            str: The model's response
            
        Raises:
            Exception: If the API request fails
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
        
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]
    
    def _deepseek_completion(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Get a completion from DeepSeek.
        
        Args:
            prompt: The prompt to send to the model
            system_message: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Optional list of stop sequences
            
        Returns:
            str: The model's response
            
        Raises:
            Exception: If the API request fails
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.deepseek_api_key}"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model if "deepseek" in self.model else "deepseek-chat",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
        
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]
    
    def stream_completion(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Stream a completion from the language model.
        
        Args:
            prompt: The prompt to send to the model
            system_message: Optional system message for chat models
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            stop_sequences: Optional list of stop sequences
            callback: Optional callback function for each chunk
            
        Returns:
            str: The model's complete response
            
        Raises:
            Exception: If the API request fails
        """
        # Use provided values or instance defaults
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        provider = self.get_model_provider()
        
        if provider == "openai":
            return self._openai_stream_completion(
                prompt=prompt,
                system_message=system_message,
                temperature=temp,
                max_tokens=tokens,
                stop_sequences=stop_sequences,
                callback=callback
            )
        elif provider == "deepseek":
            return self._deepseek_stream_completion(
                prompt=prompt,
                system_message=system_message,
                temperature=temp,
                max_tokens=tokens,
                stop_sequences=stop_sequences,
                callback=callback
            )
        else:
            raise ValueError(f"Unsupported model provider for {self.model}")
    
    def _openai_stream_completion(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop_sequences: Optional[List[str]] = None,
        callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Stream a completion from OpenAI.
        
        Args:
            prompt: The prompt to send to the model
            system_message: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Optional list of stop sequences
            callback: Optional callback function for each chunk
            
        Returns:
            str: The model's complete response
            
        Raises:
            Exception: If the API request fails
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
        
        full_response = ""
        
        for line in response.iter_lines():
            if line:
                line_text = line.decode("utf-8")
                if line_text.startswith("data: ") and line_text != "data: [DONE]":
                    data = json.loads(line_text[6:])
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            content = delta["content"]
                            full_response += content
                            if callback:
                                callback(content)
        
        return full_response
    
    def _deepseek_stream_completion(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop_sequences: Optional[List[str]] = None,
        callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Stream a completion from DeepSeek.
        
        Args:
            prompt: The prompt to send to the model
            system_message: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Optional list of stop sequences
            callback: Optional callback function for each chunk
            
        Returns:
            str: The model's complete response
            
        Raises:
            Exception: If the API request fails
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.deepseek_api_key}"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model if "deepseek" in self.model else "deepseek-chat",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
        
        full_response = ""
        
        for line in response.iter_lines():
            if line:
                line_text = line.decode("utf-8")
                if line_text.startswith("data: ") and line_text != "data: [DONE]":
                    data = json.loads(line_text[6:])
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            content = delta["content"]
                            full_response += content
                            if callback:
                                callback(content)
        
        return full_response
    
    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get usage statistics.
        
        Returns:
            Dictionary with token usage statistics
        """
        return {
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_requests": self.total_requests
        }


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
        if self.model_name in ["gpt-4o", "gpt-4o-mini", "o1", "o1-preview", "o1-mini", "o3-mini"]:
            encoding = tiktoken.encoding_for_model("gpt-4")  # Default to gpt-4 encoding
            return len(encoding.encode(text))
        
        # For other models, use an approximate token count
        # This is a rough approximation - about 4 characters per token
        return len(text) // 4
    
    def clip_tokens(self, text: str, max_tokens: int = 8000) -> str:
        """
        Clip text to a maximum number of tokens
        
        Args:
            text: The text to clip
            max_tokens: Maximum number of tokens
            
        Returns:
            Clipped text
        """
        current_tokens = self.count_tokens(text)
        
        if current_tokens <= max_tokens:
            return text
        
        # For OpenAI models use tiktoken for accurate clipping
        if self.model_name in ["gpt-4o", "gpt-4o-mini", "o1", "o1-preview", "o1-mini", "o3-mini"]:
            encoding = tiktoken.encoding_for_model("gpt-4")
            tokens = encoding.encode(text)
            # Get just the last max_tokens
            clipped_tokens = tokens[-max_tokens:]
            return encoding.decode(clipped_tokens)
        
        # For other models, use approximate clipping based on characters
        # About 4 characters per token
        char_limit = max_tokens * 4
        return text[-char_limit:] 