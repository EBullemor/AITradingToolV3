"""
LLM Client Module

Wrapper for Claude API with retry logic, rate limiting, and error handling.
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from loguru import logger

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    logger.warning("anthropic package not installed. Install with: pip install anthropic")


@dataclass
class LLMResponse:
    """Structured response from LLM."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    latency_ms: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMClient:
    """
    Claude API client with retry logic and rate limiting.
    """
    
    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    MAX_RETRIES = 3
    BASE_DELAY = 1.0  # seconds
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = MAX_RETRIES,
        timeout: int = 60
    ):
        """
        Initialize LLM client.
        
        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            model: Model to use (defaults to claude-sonnet-4-20250514)
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        if not HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )
        
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided. Set ANTHROPIC_API_KEY environment variable.")
        
        self.model = model or self.DEFAULT_MODEL
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Initialize client
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = None
        
        # Track usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        
        logger.info(f"LLM client initialized with model: {self.model}")
    
    def _retry_with_backoff(
        self,
        func,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with exponential backoff retry."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except anthropic.RateLimitError as e:
                last_exception = e
                delay = self.BASE_DELAY * (2 ** attempt)
                logger.warning(f"Rate limited. Retrying in {delay}s... (attempt {attempt + 1})")
                time.sleep(delay)
            except anthropic.APIConnectionError as e:
                last_exception = e
                delay = self.BASE_DELAY * (2 ** attempt)
                logger.warning(f"Connection error. Retrying in {delay}s... (attempt {attempt + 1})")
                time.sleep(delay)
            except anthropic.APIStatusError as e:
                # Don't retry client errors (4xx)
                if 400 <= e.status_code < 500:
                    raise
                last_exception = e
                delay = self.BASE_DELAY * (2 ** attempt)
                logger.warning(f"API error {e.status_code}. Retrying in {delay}s...")
                time.sleep(delay)
        
        raise last_exception
    
    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> LLMResponse:
        """
        Generate completion from Claude.
        
        Args:
            prompt: User message/prompt
            system: System message (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            stop_sequences: Optional stop sequences
        
        Returns:
            LLMResponse with content and metadata
        """
        if not self.client:
            raise ValueError("No API key configured. Cannot make API calls.")
        
        start_time = time.time()
        
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        
        if system:
            kwargs["system"] = system
        
        if stop_sequences:
            kwargs["stop_sequences"] = stop_sequences
        
        # Make API call with retry
        response = self._retry_with_backoff(
            self.client.messages.create,
            **kwargs
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract content
        content = ""
        if response.content:
            content = response.content[0].text
        
        # Track usage
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens
        self.total_requests += 1
        
        result = LLMResponse(
            content=content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
            latency_ms=latency_ms,
        )
        
        logger.debug(
            f"LLM response: {result.total_tokens} tokens, "
            f"{latency_ms:.0f}ms, stop={response.stop_reason}"
        )
        
        return result
    
    def complete_with_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3  # Lower temp for structured output
    ) -> Dict:
        """
        Generate completion and parse as JSON.
        
        Args:
            prompt: Prompt that asks for JSON output
            system: System message
            max_tokens: Maximum tokens
            temperature: Sampling temperature
        
        Returns:
            Parsed JSON as dictionary
        """
        import json
        
        # Add JSON instruction to prompt if not present
        if "json" not in prompt.lower():
            prompt = prompt + "\n\nRespond with valid JSON only."
        
        response = self.complete(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Try to extract JSON from response
        content = response.content.strip()
        
        # Handle markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Content was: {content[:500]}")
            raise ValueError(f"LLM did not return valid JSON: {e}")
    
    def get_usage_stats(self) -> Dict:
        """Get usage statistics."""
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        }
    
    def reset_stats(self):
        """Reset usage statistics."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0


class MockLLMClient:
    """
    Mock LLM client for testing without API calls.
    """
    
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        """
        Initialize mock client.
        
        Args:
            responses: Dict mapping prompt keywords to responses
        """
        self.responses = responses or {}
        self.call_log = []
    
    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> LLMResponse:
        """Return mock response."""
        self.call_log.append({
            "prompt": prompt,
            "system": system,
            "max_tokens": max_tokens,
        })
        
        # Find matching response
        content = "Mock response"
        for keyword, response in self.responses.items():
            if keyword.lower() in prompt.lower():
                content = response
                break
        
        return LLMResponse(
            content=content,
            model="mock-model",
            input_tokens=len(prompt.split()),
            output_tokens=len(content.split()),
            stop_reason="end_turn",
            latency_ms=10,
        )
    
    def complete_with_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3
    ) -> Dict:
        """Return mock JSON response."""
        import json
        
        response = self.complete(prompt, system, max_tokens, temperature)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Return default structure
            return {
                "summary": "Mock summary",
                "key_events": ["Event 1"],
                "confidence": "HIGH",
            }


def create_llm_client(
    api_key: Optional[str] = None,
    mock: bool = False
) -> Union[LLMClient, MockLLMClient]:
    """
    Factory function to create LLM client.
    
    Args:
        api_key: API key (uses environment if not provided)
        mock: Whether to create mock client
    
    Returns:
        LLM client instance
    """
    if mock:
        return MockLLMClient()
    
    return LLMClient(api_key=api_key)
