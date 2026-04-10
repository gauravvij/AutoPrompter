"""
LLM Client Module for interacting with OpenRouter API.
Handles queries to both Optimizer and Target LLMs with rate limiting and error recovery.
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


class LLMClient:
    """Client for interacting with OpenRouter API."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self.api_key = config.api_key
        self.api_base = config.api_base
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.request_count = 0  # Track requests for temperature jitter
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://prompt-autoresearch.local",
            "X-Title": "Prompt Autoresearch System"
        })
        self.last_request_time = 0
        self.min_request_interval = 2.0  # seconds between requests (free models need spacing)
    
    def _rate_limit(self):
        """Apply rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, messages: List[Dict[str, str]], 
                      max_retries: int = 3) -> LLMResponse:
        """Make API request with retry logic."""
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                # Apply temperature jitter every 3 requests to encourage exploration
                self.request_count += 1
                effective_temperature = self.temperature
                if self.request_count % 3 == 0:
                    # Add jitter: vary temperature by ±0.1, clamped to [0.0, 1.0]
                    import random
                    jitter = random.uniform(-0.1, 0.1)
                    effective_temperature = max(0.0, min(1.0, self.temperature + jitter))
                    logger.debug(f"Applied temperature jitter: {self.temperature} -> {effective_temperature}")
                
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": effective_temperature,
                    "max_tokens": self.max_tokens
                }
                
                response = self.session.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    timeout=60
                )

                # Handle error status codes before parsing body
                if response.status_code in (429, 503):
                    wait_time = 10 * (2 ** attempt)  # 10s, 20s, 40s
                    logger.warning(f"Rate limited (HTTP {response.status_code}). Waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue

                if response.status_code >= 400:
                    # Try to read the JSON body for a more informative error message
                    error_msg = f"HTTP {response.status_code}"
                    is_rate_limit = False
                    try:
                        err_body = response.json()
                        err_detail = err_body.get('error', {})
                        err_code = err_detail.get('code') if isinstance(err_detail, dict) else None
                        err_text = (err_detail.get('message', '') if isinstance(err_detail, dict) else str(err_detail)) or ''
                        raw_text = (err_detail.get('metadata', {}) or {}).get('raw', '') if isinstance(err_detail, dict) else ''
                        error_msg = f"HTTP {response.status_code}: {err_text}"
                        # Detect provider-side rate limit returned as 4xx
                        rl_indicators = ('rate', 'quota', 'limit', 'throttl', 'too many', 'capacity', 'upstream')
                        if err_code in (429, 503) or any(w in (err_text + raw_text).lower() for w in rl_indicators):
                            is_rate_limit = True
                    except Exception:
                        error_msg = f"HTTP {response.status_code}: {response.text[:200]}"

                    if is_rate_limit:
                        wait_time = 10 * (2 ** attempt)
                        logger.warning(f"Provider rate limit ({error_msg}). Waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue

                    # Genuine bad request — log and fail immediately (retrying won't help)
                    logger.error(f"Request rejected by API ({error_msg}) for model {self.model}")
                    return LLMResponse(
                        content="",
                        model=self.model,
                        usage={},
                        latency_ms=(time.time() - start_time) * 1000,
                        success=False,
                        error=error_msg
                    )

                data = response.json()
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Validate response structure and content
                # Handle OpenRouter's nested JSON: choices[0].message.content
                # Some models (e.g., Qwen) may return content=None but have reasoning
                content = None
                error_detail = None
                
                try:
                    if 'choices' not in data:
                        error_detail = "Missing 'choices' in response"
                    elif not data['choices']:
                        error_detail = "Empty 'choices' array in response"
                    elif not isinstance(data['choices'], list):
                        error_detail = "'choices' is not a list"
                    elif len(data['choices']) == 0:
                        error_detail = "'choices' array has no elements"
                    else:
                        choice = data['choices'][0]
                        if not isinstance(choice, dict):
                            error_detail = "Choice is not a dictionary"
                        elif 'message' not in choice:
                            error_detail = "Missing 'message' in choice"
                        elif not isinstance(choice['message'], dict):
                            error_detail = "'message' is not a dictionary"
                        else:
                            msg = choice['message']
                            # Try to get content from 'content' field first
                            if 'content' in msg:
                                content = msg['content']
                            
                            # If content is None or empty, try 'reasoning' field (Qwen models)
                            if (content is None or not str(content).strip()) and 'reasoning' in msg:
                                content = msg['reasoning']
                                logger.debug(f"Using 'reasoning' field instead of content for {self.model}")
                            
                            # Convert to string if needed
                            if content is None:
                                error_detail = "Content is None and no reasoning field available"
                                content = ""
                            elif not isinstance(content, str):
                                content = str(content)
                except Exception as e:
                    error_detail = f"Exception parsing response: {str(e)}"
                
                # Check if content is valid
                if error_detail or not content or not content.strip():
                    # Empty or invalid response - treat as failure and retry
                    log_msg = error_detail if error_detail else "Empty response content"
                    logger.warning(f"Invalid response from LLM (attempt {attempt + 1}/{max_retries}): {log_msg}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        final_error = error_detail if error_detail else "Empty response content after all retries"
                        return LLMResponse(
                            content=content if content else "",
                            model=self.model,
                            usage=data.get('usage', {}),
                            latency_ms=latency_ms,
                            success=False,
                            error=final_error
                        )
                
                return LLMResponse(
                    content=content,
                    model=data.get('model', self.model),
                    usage=data.get('usage', {}),
                    latency_ms=latency_ms,
                    success=True
                )
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed with model: {self.model} (attempt {attempt + 1}/{max_retries}): {e}")
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
            except (KeyError, json.JSONDecodeError) as e:
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


if __name__ == "__main__":
    # Test the LLM client
    from config_manager import LLMConfig
    
    # Create test config
    config = LLMConfig(
        model="qwen/qwen3-8b:free",
        temperature=0.1,
        max_tokens=100
    )
    
    client = LLMClient(config)
    
    # Test query
    print("Testing LLM client...")
    response = client.query("Say 'Hello, World!' and nothing else.")
    
    print(f"Success: {response.success}")
    print(f"Content: {response.content}")
    print(f"Latency: {response.latency_ms:.2f}ms")
    if response.error:
        print(f"Error: {response.error}")
