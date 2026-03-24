"""
Local LLM Client Module for Ollama and llama.cpp backends.
Provides unified interface for local model inference with automatic backend detection.
"""

import requests
import time
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from LLM."""
    content: str
    model: str
    usage: Dict[str, int]
    latency_ms: float
    success: bool
    error: Optional[str] = None


class LocalLLMClient:
    """Client for interacting with local LLM backends (Ollama, llama.cpp)."""
    
    SUPPORTED_BACKENDS = ['ollama', 'llama_cpp', 'auto']
    
    def __init__(self, config):
        """Initialize with configuration.
        
        Args:
            config: Configuration object with backend, model, host, and other parameters
        """
        self.config = config
        self.backend = config.backend
        self.model = config.model
        self.host = getattr(config, 'host', 'http://localhost')
        self.port = getattr(config, 'port', None)
        self.temperature = getattr(config, 'temperature', 0.7)
        self.max_tokens = getattr(config, 'max_tokens', 4096)
        self.timeout = getattr(config, 'timeout', 120)
        
        # Build API endpoint
        if self.port:
            self.api_base = f"{self.host}:{self.port}"
        else:
            self.api_base = self.host
        
        # Remove trailing slash if present
        self.api_base = self.api_base.rstrip('/')
        
        self.request_count = 0
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        self.last_request_time = 0
        self.min_request_interval = getattr(config, 'min_request_interval', 0.1)
        
        # Auto-detect backend if not specified
        if self.backend == 'auto':
            self.backend = self._detect_backend()
            logger.info(f"Auto-detected backend: {self.backend}")
        
        logger.info(f"Local LLM Client initialized: backend={self.backend}, model={self.model}, api_base={self.api_base}")
    
    def _detect_backend(self) -> str:
        """Auto-detect available backend by probing endpoints."""
        # Try Ollama first
        try:
            response = requests.get(
                f"{self.api_base}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                logger.debug("Ollama backend detected")
                return 'ollama'
        except requests.exceptions.RequestException:
            pass
        
        # Try llama.cpp
        try:
            response = requests.get(
                f"{self.api_base}/v1/models",
                timeout=5
            )
            if response.status_code == 200:
                logger.debug("llama.cpp backend detected")
                return 'llama_cpp'
        except requests.exceptions.RequestException:
            pass
        
        raise ValueError(
            f"Could not detect backend at {self.api_base}. "
            "Please specify backend explicitly (ollama or llama_cpp) or ensure the server is running."
        )
    
    def _rate_limit(self):
        """Apply rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_ollama_request(self, messages: List[Dict[str, str]], 
                             max_retries: int = 3) -> LLMResponse:
        """Make request to Ollama API."""
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                # Convert messages to Ollama format
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                }
                
                response = self.session.post(
                    f"{self.api_base}/api/chat",
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 404:
                    error_msg = f"Model '{self.model}' not found on Ollama server"
                    logger.error(error_msg)
                    return LLMResponse(
                        content="",
                        model=self.model,
                        usage={},
                        latency_ms=(time.time() - start_time) * 1000,
                        success=False,
                        error=error_msg
                    )
                
                response.raise_for_status()
                data = response.json()
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Parse Ollama response
                content = data.get('message', {}).get('content', '')
                
                # Calculate token usage
                usage = {
                    'prompt_tokens': data.get('prompt_eval_count', 0),
                    'completion_tokens': data.get('eval_count', 0),
                    'total_tokens': data.get('prompt_eval_count', 0) + data.get('eval_count', 0)
                }
                
                if not content or not content.strip():
                    logger.warning(f"Empty response from Ollama (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        return LLMResponse(
                            content="",
                            model=self.model,
                            usage=usage,
                            latency_ms=latency_ms,
                            success=False,
                            error="Empty response from Ollama"
                        )
                
                return LLMResponse(
                    content=content,
                    model=data.get('model', self.model),
                    usage=usage,
                    latency_ms=latency_ms,
                    success=True
                )
                
            except requests.exceptions.Timeout:
                logger.error(f"Request timed out (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return LLMResponse(
                        content="",
                        model=self.model,
                        usage={},
                        latency_ms=(time.time() - start_time) * 1000,
                        success=False,
                        error="Request timed out"
                    )
            except requests.exceptions.ConnectionError:
                logger.error(f"Connection error to Ollama at {self.api_base}")
                return LLMResponse(
                    content="",
                    model=self.model,
                    usage={},
                    latency_ms=(time.time() - start_time) * 1000,
                    success=False,
                    error=f"Could not connect to Ollama server at {self.api_base}. Is it running?"
                )
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return LLMResponse(
                        content="",
                        model=self.model,
                        usage={},
                        latency_ms=(time.time() - start_time) * 1000,
                        success=False,
                        error=str(e)
                    )
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse response: {e}")
                return LLMResponse(
                    content="",
                    model=self.model,
                    usage={},
                    latency_ms=(time.time() - start_time) * 1000,
                    success=False,
                    error=f"Parse error: {e}"
                )
        
        return LLMResponse(
            content="",
            model=self.model,
            usage={},
            latency_ms=(time.time() - start_time) * 1000,
            success=False,
            error="Max retries exceeded"
        )
    
    def _make_llama_cpp_request(self, messages: List[Dict[str, str]], 
                                 max_retries: int = 3) -> LLMResponse:
        """Make request to llama.cpp API (OpenAI-compatible)."""
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                # llama.cpp uses OpenAI-compatible API format
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": False
                }
                
                response = self.session.post(
                    f"{self.api_base}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 404:
                    error_msg = f"Model '{self.model}' not found on llama.cpp server"
                    logger.error(error_msg)
                    return LLMResponse(
                        content="",
                        model=self.model,
                        usage={},
                        latency_ms=(time.time() - start_time) * 1000,
                        success=False,
                        error=error_msg
                    )
                
                response.raise_for_status()
                data = response.json()
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Parse response (OpenAI-compatible format)
                content = None
                error_detail = None
                
                try:
                    if 'choices' not in data:
                        error_detail = "Missing 'choices' in response"
                    elif not data['choices']:
                        error_detail = "Empty 'choices' array in response"
                    else:
                        choice = data['choices'][0]
                        if 'message' not in choice:
                            error_detail = "Missing 'message' in choice"
                        else:
                            msg = choice['message']
                            if 'content' in msg:
                                content = msg['content']
                            
                            # Try reasoning field for some models
                            if (content is None or not str(content).strip()) and 'reasoning' in msg:
                                content = msg['reasoning']
                            
                            if content is None:
                                error_detail = "Content is None"
                                content = ""
                            elif not isinstance(content, str):
                                content = str(content)
                except Exception as e:
                    error_detail = f"Exception parsing response: {str(e)}"
                
                # Get usage info
                usage = data.get('usage', {})
                if not usage:
                    usage = {
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0
                    }
                
                if error_detail or not content or not content.strip():
                    log_msg = error_detail if error_detail else "Empty response content"
                    logger.warning(f"Invalid response from llama.cpp (attempt {attempt + 1}/{max_retries}): {log_msg}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        final_error = error_detail if error_detail else "Empty response content after all retries"
                        return LLMResponse(
                            content=content if content else "",
                            model=self.model,
                            usage=usage,
                            latency_ms=latency_ms,
                            success=False,
                            error=final_error
                        )
                
                return LLMResponse(
                    content=content,
                    model=data.get('model', self.model),
                    usage=usage,
                    latency_ms=latency_ms,
                    success=True
                )
                
            except requests.exceptions.Timeout:
                logger.error(f"Request timed out (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return LLMResponse(
                        content="",
                        model=self.model,
                        usage={},
                        latency_ms=(time.time() - start_time) * 1000,
                        success=False,
                        error="Request timed out"
                    )
            except requests.exceptions.ConnectionError:
                logger.error(f"Connection error to llama.cpp at {self.api_base}")
                return LLMResponse(
                    content="",
                    model=self.model,
                    usage={},
                    latency_ms=(time.time() - start_time) * 1000,
                    success=False,
                    error=f"Could not connect to llama.cpp server at {self.api_base}. Is it running?"
                )
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return LLMResponse(
                        content="",
                        model=self.model,
                        usage={},
                        latency_ms=(time.time() - start_time) * 1000,
                        success=False,
                        error=str(e)
                    )
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse response: {e}")
                return LLMResponse(
                    content="",
                    model=self.model,
                    usage={},
                    latency_ms=(time.time() - start_time) * 1000,
                    success=False,
                    error=f"Parse error: {e}"
                )
        
        return LLMResponse(
            content="",
            model=self.model,
            usage={},
            latency_ms=(time.time() - start_time) * 1000,
            success=False,
            error="Max retries exceeded"
        )
    
    def _make_request(self, messages: List[Dict[str, str]], 
                      max_retries: int = 3) -> LLMResponse:
        """Make API request with retry logic."""
        if self.backend == 'ollama':
            return self._make_ollama_request(messages, max_retries)
        elif self.backend == 'llama_cpp':
            return self._make_llama_cpp_request(messages, max_retries)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def query(self, prompt: str, system_message: Optional[str] = None) -> LLMResponse:
        """Send a single prompt to the LLM."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        return self._make_request(messages)
    
    def query_with_history(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Send a conversation with history."""
        return self._make_request(messages)
    
    def batch_query(self, prompts: List[str], 
                    system_message: Optional[str] = None) -> List[LLMResponse]:
        """Process multiple prompts sequentially."""
        responses = []
        for prompt in prompts:
            response = self.query(prompt, system_message)
            responses.append(response)
        return responses
    
    def list_models(self) -> List[str]:
        """List available models on the backend."""
        if self.backend == 'ollama':
            try:
                response = requests.get(
                    f"{self.api_base}/api/tags",
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            except Exception as e:
                logger.error(f"Failed to list Ollama models: {e}")
                return []
        elif self.backend == 'llama_cpp':
            try:
                response = requests.get(
                    f"{self.api_base}/v1/models",
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                return [model['id'] for model in data.get('data', [])]
            except Exception as e:
                logger.error(f"Failed to list llama.cpp models: {e}")
                return []
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def check_connection(self) -> bool:
        """Check if the backend server is reachable."""
        try:
            if self.backend == 'ollama':
                response = requests.get(
                    f"{self.api_base}/api/tags",
                    timeout=5
                )
                return response.status_code == 200
            elif self.backend == 'llama_cpp':
                response = requests.get(
                    f"{self.api_base}/v1/models",
                    timeout=5
                )
                return response.status_code == 200
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
        except requests.exceptions.RequestException:
            return False


def get_available_backends() -> List[str]:
    """Get list of supported backend types."""
    return LocalLLMClient.SUPPORTED_BACKENDS


if __name__ == "__main__":
    # Test the local LLM client
    from dataclasses import dataclass
    
    @dataclass
    class TestConfig:
        backend: str
        model: str
        host: str = "http://localhost"
        port: int = None
        temperature: float = 0.7
        max_tokens: int = 4096
        timeout: int = 120
        min_request_interval: float = 0.1
    
    print("Testing Local LLM Client...")
    print(f"Supported backends: {get_available_backends()}")
    
    # Test with Ollama (default port 11434)
    try:
        config = TestConfig(
            backend="auto",
            model="llama3.2",
            host="http://localhost",
            port=11434
        )
        client = LocalLLMClient(config)
        
        # Check connection
        if client.check_connection():
            print(f"\nConnected to {client.backend} backend")
            
            # List models
            models = client.list_models()
            print(f"Available models: {models}")
            
            # Test query
            print("\nTesting query...")
            response = client.query("Say 'Hello from local LLM!' and nothing else.")
            
            print(f"Success: {response.success}")
            print(f"Content: {response.content}")
            print(f"Latency: {response.latency_ms:.2f}ms")
            if response.error:
                print(f"Error: {response.error}")
        else:
            print(f"\nCould not connect to {client.backend} server at {client.api_base}")
            print("Make sure Ollama or llama.cpp is running:")
            print("  - Ollama: ollama serve")
            print("  - llama.cpp: ./llama-server -m model.gguf -c 4096")
    except ValueError as e:
        print(f"\nBackend detection failed: {e}")
    except Exception as e:
        print(f"\nError: {e}")
