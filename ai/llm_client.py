"""
LLM Client Abstraction Layer
Supports multiple LLM providers: OpenAI, Anthropic, Ollama, LlamaCPP
"""

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timeout = config.get('timeout', 30)

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is available"""
        pass


class OllamaClient(BaseLLMClient):
    """Client for local Ollama LLM"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.model = config.get('model', 'llama3.2:3b')

        try:
            import ollama
            self.client = ollama.Client(host=self.base_url)
            logger.info(f"Ollama client initialized with model {self.model}")
        except ImportError:
            logger.error("ollama package not installed. Install with: pip install ollama")
            raise

    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            # Try to list models to verify connection
            self.client.list()
            return True
        except Exception as e:
            logger.error(f"Ollama not available: {e}")
            return False

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama"""
        try:
            temperature = kwargs.get('temperature', 0.85)
            max_tokens = kwargs.get('max_tokens', 600)

            # Ollama uses async, run in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        'temperature': temperature,
                        'num_predict': max_tokens,
                    }
                )
            )

            return response['response']

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Get API key from config or environment
        api_key = config.get('api_key', os.getenv('OPENAI_API_KEY'))
        if not api_key or api_key.startswith('${'):
            raise ValueError("OpenAI API key not configured")

        self.model = config.get('model', 'gpt-4')
        self.base_url = config.get('base_url', 'https://api.openai.com/v1')

        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
            logger.info(f"OpenAI client initialized with model {self.model}")
        except ImportError:
            logger.error("openai package not installed. Install with: pip install openai")
            raise

    def is_available(self) -> bool:
        """Check if API key is configured"""
        return True  # Assume available if initialized

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API"""
        try:
            temperature = kwargs.get('temperature', 0.85)
            max_tokens = kwargs.get('max_tokens', 600)

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude API"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Get API key from config or environment
        api_key = config.get('api_key', os.getenv('ANTHROPIC_API_KEY'))
        if not api_key or api_key.startswith('${'):
            raise ValueError("Anthropic API key not configured")

        self.model = config.get('model', 'claude-3-5-sonnet-20241022')

        try:
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(
                api_key=api_key,
                timeout=self.timeout
            )
            logger.info(f"Anthropic client initialized with model {self.model}")
        except ImportError:
            logger.error("anthropic package not installed. Install with: pip install anthropic")
            raise

    def is_available(self) -> bool:
        """Check if API key is configured"""
        return True

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic API"""
        try:
            temperature = kwargs.get('temperature', 0.85)
            max_tokens = kwargs.get('max_tokens', 600)

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise


class LlamaCppClient(BaseLLMClient):
    """Client for local LlamaCPP models"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.model_path = config.get('model_path')
        if not self.model_path or not os.path.exists(self.model_path):
            raise ValueError(f"LlamaCPP model not found at {self.model_path}")

        self.n_ctx = config.get('n_ctx', 2048)
        self.n_threads = config.get('n_threads', 4)

        try:
            from llama_cpp import Llama

            # Initialize in thread pool since it's CPU intensive
            loop = asyncio.get_event_loop()
            self.llm = loop.run_until_complete(
                loop.run_in_executor(
                    None,
                    lambda: Llama(
                        model_path=self.model_path,
                        n_ctx=self.n_ctx,
                        n_threads=self.n_threads,
                        verbose=False
                    )
                )
            )
            logger.info(f"LlamaCPP model loaded from {self.model_path}")
        except ImportError:
            logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            raise

    def is_available(self) -> bool:
        """Check if model is loaded"""
        return self.llm is not None

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using LlamaCPP"""
        try:
            temperature = kwargs.get('temperature', 0.85)
            max_tokens = kwargs.get('max_tokens', 600)

            # Run in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["</s>", "\n\n\n"],
                    echo=False
                )
            )

            return response['choices'][0]['text']

        except Exception as e:
            logger.error(f"LlamaCPP generation failed: {e}")
            raise


class LLMClientFactory:
    """Factory to create appropriate LLM client based on configuration"""

    _clients = {
        'ollama': OllamaClient,
        'openai': OpenAIClient,
        'anthropic': AnthropicClient,
        'llamacpp': LlamaCppClient,
    }

    @staticmethod
    def create_client(llm_config: Dict[str, Any]) -> BaseLLMClient:
        """
        Create an LLM client based on configuration

        Args:
            llm_config: LLM configuration from ai_config.yaml

        Returns:
            Initialized LLM client
        """
        provider = llm_config.get('provider', 'ollama').lower()

        if provider not in LLMClientFactory._clients:
            raise ValueError(f"Unknown LLM provider: {provider}")

        # Get provider-specific config
        provider_config = llm_config.get('providers', {}).get(provider, {})

        # Merge with general timeout
        provider_config['timeout'] = llm_config.get('timeout', provider_config.get('timeout', 30))

        client_class = LLMClientFactory._clients[provider]
        client = client_class(provider_config)

        # Verify availability
        if not client.is_available():
            logger.warning(f"{provider} client created but service may not be available")

        return client


# Convenience function for creating client
def create_llm_client(config: Dict[str, Any]) -> BaseLLMClient:
    """Create LLM client from configuration"""
    return LLMClientFactory.create_client(config.get('llm', {}))
