"""
Tool Result Pre-filtering System
Summarizes large tool outputs before sending to Claude.
Reduces token usage by 80% for tool results.
"""

import asyncio
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
import aiohttp
import requests


# Configuration constants
DEFAULT_OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://192.168.85.158:11434")
DEFAULT_SUMMARY_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
SUMMARY_MAX_TOKENS = 200
OLLAMA_TIMEOUT = 30
MAX_INPUT_CHARS = 8000  # ~2000 tokens, prevents overly large prompts
MAX_PRESERVED_ITEMS = 20
RATE_LIMIT_CALLS_PER_MINUTE = 60  # Ollama rate limit protection


@dataclass
class SummaryResult:
    """Result of tool output summarization"""
    original_output: str
    summary: str
    was_summarized: bool
    original_tokens: int
    summary_tokens: int
    tool_name: str
    processing_time_ms: float
    metadata: Dict[str, Any]


@dataclass
class ToolConfig:
    """Configuration for a specific tool type"""
    name: str
    token_threshold: int
    summary_focus: str
    preserve_patterns: List[str]  # Regex patterns to always preserve
    max_summary_tokens: int = SUMMARY_MAX_TOKENS


class ToolRegistry:
    """
    Registry of tool configurations.
    Each tool type has specific summarization rules.
    """
    
    DEFAULT_CONFIGS = {
        "file_read": ToolConfig(
            name="file_read",
            token_threshold=1000,
            summary_focus="key sections, imports, function signatures, and structure",
            preserve_patterns=[
                r"^(import|from)\s+",  # Import statements
                r"^(def|class)\s+\w+",  # Function/class definitions
                r"^\s*#\s*TODO|FIXME|NOTE",  # Comments
            ]
        ),
        "directory_list": ToolConfig(
            name="directory_list",
            token_threshold=500,
            summary_focus="directory structure, file counts by type, and key files",
            preserve_patterns=[
                r"\.(py|js|ts|jsx|tsx|go|rs|java|cpp|c|h)$",  # Source files
                r"(README|CHANGELOG|LICENSE|package\.json|Cargo\.toml)"
            ]
        ),
        "search_results": ToolConfig(
            name="search_results",
            token_threshold=1000,
            summary_focus="top matches, file locations, and context snippets",
            preserve_patterns=[
                r"^\d+:\s+",  # Line numbers
                r"(error|fail|exception|bug)",  # Error indicators
            ]
        ),
        "log_output": ToolConfig(
            name="log_output",
            token_threshold=2000,
            summary_focus="error patterns, recent entries, and frequency statistics",
            preserve_patterns=[
                r"(ERROR|FATAL|CRITICAL|Exception|Traceback)",  # Errors
                r"\d{4}-\d{2}-\d{2}.*\d{2}:\d{2}:\d{2}",  # Timestamps
            ]
        ),
        "http_response": ToolConfig(
            name="http_response",
            token_threshold=1000,
            summary_focus="status code, key headers, and body structure",
            preserve_patterns=[
                r"^HTTP/\d\.\d\s+\d+",  # Status line
                r"^(Content-Type|Authorization|Location):",  # Key headers
                r'"(error|message|status|code)":',  # JSON error fields
            ]
        ),
        "database_query": ToolConfig(
            name="database_query",
            token_threshold=1000,
            summary_focus="row count, column names, and sample data",
            preserve_patterns=[
                r"^\|?\s*[-]+\s*\|?",  # Table separators
                r"(COUNT|SUM|AVG|MAX|MIN)\s*\(",  # Aggregates
            ]
        ),
        "exec_output": ToolConfig(
            name="exec_output",
            token_threshold=1500,
            summary_focus="command status, key output lines, and errors",
            preserve_patterns=[
                r"^(Success|Error|Failed|Completed)",
                r"(exit code|returned|status):\s*\d+",
            ]
        ),
        "git_status": ToolConfig(
            name="git_status",
            token_threshold=500,
            summary_focus="modified files, untracked files, and branch status",
            preserve_patterns=[
                r"^(M|A|D|R|C|U)\s+",  # Git status codes
                r"^(modified|new file|deleted):",
            ]
        ),
    }
    
    def __init__(self):
        self.configs = self.DEFAULT_CONFIGS.copy()
    
    def get_config(self, tool_name: str) -> Optional[ToolConfig]:
        """Get configuration for a tool"""
        return self.configs.get(tool_name)
    
    def register_tool(self, config: ToolConfig):
        """Register a new tool configuration"""
        self.configs[config.name] = config
    
    def should_summarize(self, tool_name: str, output: str) -> bool:
        """Check if this tool output should be summarized"""
        config = self.get_config(tool_name)
        if not config:
            return False
        
        tokens = self._estimate_tokens(output)
        return tokens >= config.token_threshold
    
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Estimate tokens using improved estimator.
        Uses different ratios for code vs text.
        """
        return TokenEstimator.estimate(text)


class SummaryFormatter:
    """
    Formats tool summaries in a consistent structure
    that Claude can easily parse.
    """
    
    @staticmethod
    def format_summary(tool_name: str, summary: str, metadata: Dict) -> str:
        """Format summary with metadata header"""
        lines = [
            f"[Tool: {tool_name} - Summarized Output]",
            f"Original size: {metadata.get('original_tokens', '?')} tokens",
            f"Summary: {metadata.get('summary_tokens', '?')} tokens",
            "",
            "=== Summary ===",
            summary,
            "",
            "=== Key Details ===",
        ]
        
        # Add preserved items if any
        preserved = metadata.get('preserved_items', [])
        if preserved:
            lines.append("Important lines preserved from original:")
            for item in preserved[:10]:  # Max 10 preserved items
                lines.append(f"  • {item[:100]}")  # Truncate long lines
        
        return "\n".join(lines)
    
    @staticmethod
    def format_error(tool_name: str, error: str, original: str) -> str:
        """Format error case - return truncated original"""
        return f"""[Tool: {tool_name} - Summary Failed]
Error: {error}

Returning truncated original ({len(original)} chars):
{original[:2000]}...
"""


class MetricsTracker:
    """
    Tracks token savings and performance metrics.
    """
    
    def __init__(self):
        self.stats = {
            "total_calls": 0,
            "summarized_calls": 0,
            "total_tokens_saved": 0,
            "tool_breakdown": {}
        }
    
    def record(self, tool_name: str, 
               original_tokens: int, 
               summary_tokens: int,
               was_summarized: bool,
               processing_time_ms: float):
        """Record a summarization event"""
        self.stats["total_calls"] += 1
        
        if was_summarized:
            self.stats["summarized_calls"] += 1
            saved = original_tokens - summary_tokens
            self.stats["total_tokens_saved"] += saved
        
        # Per-tool stats
        if tool_name not in self.stats["tool_breakdown"]:
            self.stats["tool_breakdown"][tool_name] = {
                "calls": 0,
                "summarized": 0,
                "tokens_saved": 0
            }
        
        tool_stats = self.stats["tool_breakdown"][tool_name]
        tool_stats["calls"] += 1
        if was_summarized:
            tool_stats["summarized"] += 1
            tool_stats["tokens_saved"] += original_tokens - summary_tokens
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        total = self.stats["total_calls"]
        summarized = self.stats["summarized_calls"]
        
        return {
            **self.stats,
            "summarization_rate": summarized / total if total > 0 else 0,
            "avg_tokens_saved": (
                self.stats["total_tokens_saved"] / summarized 
                if summarized > 0 else 0
            )
        }
    
    def print_report(self):
        """Print formatted statistics report"""
        stats = self.get_stats()
        print("\n=== Tool Pre-filtering Metrics ===")
        print(f"Total calls: {stats['total_calls']}")
        print(f"Summarized: {stats['summarized_calls']} ({stats['summarization_rate']:.1%})")
        print(f"Total tokens saved: {stats['total_tokens_saved']:,}")
        print(f"Avg tokens saved per summary: {stats['avg_tokens_saved']:.0f}")
        print("\nBy tool:")
        for tool, tool_stats in stats['tool_breakdown'].items():
            rate = tool_stats['summarized'] / tool_stats['calls'] if tool_stats['calls'] > 0 else 0
            print(f"  {tool}: {tool_stats['calls']} calls, {rate:.1%} summarized, {tool_stats['tokens_saved']:,} tokens saved")


def validate_input(func):
    """Decorator to validate inputs to summarize methods"""
    @wraps(func)
    def wrapper(self, tool_name: str, output: str, context: Optional[Dict] = None, *args, **kwargs):
        # Validate tool_name
        if not isinstance(tool_name, str):
            raise TypeError(f"tool_name must be a string, got {type(tool_name)}")
        if not tool_name.strip():
            raise ValueError("tool_name cannot be empty")
        
        # Validate output
        if output is None:
            raise ValueError("output cannot be None")
        if not isinstance(output, str):
            output = str(output)
        
        # Validate context
        if context is not None and not isinstance(context, dict):
            raise TypeError(f"context must be a dict or None, got {type(context)}")
        
        return func(self, tool_name, output, context, *args, **kwargs)
    return wrapper


class ToolSummarizer:
    """
    Main interface for tool result pre-filtering.
    Summarizes large tool outputs using local 7B model.
    """
    
    def __init__(self,
                 ollama_endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
                 model: str = DEFAULT_SUMMARY_MODEL,
                 rate_limit_calls_per_minute: int = RATE_LIMIT_CALLS_PER_MINUTE):
        self.ollama_endpoint = ollama_endpoint
        self.model = model
        self.registry = ToolRegistry()
        self.formatter = SummaryFormatter()
        self.metrics = MetricsTracker()
        self.rate_limiter = RateLimiter(rate_limit_calls_per_minute)
    
    @validate_input
    def summarize(self, 
                  tool_name: str, 
                  output: str,
                  context: Optional[Dict] = None) -> SummaryResult:
        """
        Summarize tool output if it exceeds threshold.
        
        Args:
            tool_name: Name of the tool that produced output
            output: Raw tool output
            context: Additional context (file path, query, etc.)
        
        Returns:
            SummaryResult with original and summary
        """
        import time
        start_time = time.time()
        
        original_tokens = self.registry._estimate_tokens(output)
        context = context or {}
        
        # Check if we should summarize
        config = self.registry.get_config(tool_name)
        if not config or not self.registry.should_summarize(tool_name, output):
            # No summarization needed
            processing_time = (time.time() - start_time) * 1000
            self.metrics.record(tool_name, original_tokens, original_tokens, False, processing_time)
            
            return SummaryResult(
                original_output=output,
                summary=output,
                was_summarized=False,
                original_tokens=original_tokens,
                summary_tokens=original_tokens,
                tool_name=tool_name,
                processing_time_ms=processing_time,
                metadata={"reason": "Below threshold"}
            )
        
        # Extract preserved items
        preserved_items = self._extract_preserved_items(output, config.preserve_patterns)
        
        # Generate summary
        try:
            summary = self._generate_summary(output, config, context)
            summary_tokens = self.registry._estimate_tokens(summary)
            
            # Format final output
            formatted_summary = self.formatter.format_summary(
                tool_name, summary, {
                    "original_tokens": original_tokens,
                    "summary_tokens": summary_tokens,
                    "preserved_items": preserved_items
                }
            )
            
            processing_time = (time.time() - start_time) * 1000
            self.metrics.record(tool_name, original_tokens, summary_tokens, True, processing_time)
            
            return SummaryResult(
                original_output=output,
                summary=formatted_summary,
                was_summarized=True,
                original_tokens=original_tokens,
                summary_tokens=summary_tokens,
                tool_name=tool_name,
                processing_time_ms=processing_time,
                metadata={
                    "preserved_items": preserved_items,
                    "focus": config.summary_focus
                }
            )
            
        except Exception as e:
            # Fallback: return truncated original
            processing_time = (time.time() - start_time) * 1000
            error_summary = self.formatter.format_error(tool_name, str(e), output)
            
            self.metrics.record(tool_name, original_tokens, original_tokens, False, processing_time)
            
            return SummaryResult(
                original_output=output,
                summary=error_summary,
                was_summarized=False,
                original_tokens=original_tokens,
                summary_tokens=original_tokens,
                tool_name=tool_name,
                processing_time_ms=processing_time,
                metadata={"error": str(e)}
            )
    
    def _extract_preserved_items(self, output: str, patterns: List[str]) -> List[str]:
        """Extract lines matching preserve patterns"""
        preserved = []
        lines = output.split('\n')
        
        for line in lines:
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    preserved.append(line.strip())
                    break
        
        return preserved[:MAX_PRESERVED_ITEMS]
    
    def _generate_summary(self, 
                         output: str, 
                         config: ToolConfig,
                         context: Dict) -> str:
        """Generate summary using local 7B model"""
        
        # Apply rate limiting
        self.rate_limiter.acquire()
        
        # Truncate very long outputs for the prompt
        if len(output) > MAX_INPUT_CHARS:
            output = output[:MAX_INPUT_CHARS] + "\n... [truncated for summary]"
        
        # Build prompt
        prompt = f"""You are a tool output summarizer. Create a concise summary of the following tool output.

Tool: {config.name}
Context: {json.dumps(context)}
Focus on: {config.summary_focus}

Output to summarize:
```
{output}
```

Provide a brief summary (max 150 words) highlighting:
1. Key findings or results
2. Important files/locations mentioned
3. Any errors or warnings
4. Overall structure or pattern

Summary:"""
        
        # Call Ollama
        response = requests.post(
            f"{self.ollama_endpoint}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": config.max_summary_tokens
                }
            },
            timeout=OLLAMA_TIMEOUT
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama error: {response.status_code}")
        
        result = response.json()
        summary = result.get("response", "").strip()
        
        return summary
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        return self.metrics.get_stats()
    
    def print_metrics(self):
        """Print metrics report"""
        self.metrics.print_report()


class AsyncToolSummarizer(ToolSummarizer):
    """
    Async version of ToolSummarizer for non-blocking operation.
    Uses aiohttp for async HTTP calls to Ollama.
    """
    
    def __init__(self,
                 ollama_endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
                 model: str = DEFAULT_SUMMARY_MODEL,
                 rate_limit_calls_per_minute: int = RATE_LIMIT_CALLS_PER_MINUTE):
        super().__init__(ollama_endpoint, model, rate_limit_calls_per_minute)
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close the aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def summarize(self, 
                       tool_name: str, 
                       output: str,
                       context: Optional[Dict] = None) -> SummaryResult:
        """
        Async version of summarize.
        Non-blocking tool output summarization.
        """
        import time
        start_time = time.time()
        
        # Validate inputs (reuse sync validation)
        if not isinstance(tool_name, str):
            raise TypeError(f"tool_name must be a string, got {type(tool_name)}")
        if not tool_name.strip():
            raise ValueError("tool_name cannot be empty")
        if output is None:
            raise ValueError("output cannot be None")
        if not isinstance(output, str):
            output = str(output)
        if context is not None and not isinstance(context, dict):
            raise TypeError(f"context must be a dict or None, got {type(context)}")
        
        original_tokens = TokenEstimator.estimate(output)
        context = context or {}
        
        # Check if we should summarize
        config = self.registry.get_config(tool_name)
        if not config or not self.registry.should_summarize(tool_name, output):
            processing_time = (time.time() - start_time) * 1000
            self.metrics.record(tool_name, original_tokens, original_tokens, False, processing_time)
            
            return SummaryResult(
                original_output=output,
                summary=output,
                was_summarized=False,
                original_tokens=original_tokens,
                summary_tokens=original_tokens,
                tool_name=tool_name,
                processing_time_ms=processing_time,
                metadata={"reason": "Below threshold"}
            )
        
        # Extract preserved items
        preserved_items = self._extract_preserved_items(output, config.preserve_patterns)
        
        # Generate summary
        try:
            summary = await self._generate_summary_async(output, config, context)
            summary_tokens = TokenEstimator.estimate(summary)
            
            formatted_summary = self.formatter.format_summary(
                tool_name, summary, {
                    "original_tokens": original_tokens,
                    "summary_tokens": summary_tokens,
                    "preserved_items": preserved_items
                }
            )
            
            processing_time = (time.time() - start_time) * 1000
            self.metrics.record(tool_name, original_tokens, summary_tokens, True, processing_time)
            
            return SummaryResult(
                original_output=output,
                summary=formatted_summary,
                was_summarized=True,
                original_tokens=original_tokens,
                summary_tokens=summary_tokens,
                tool_name=tool_name,
                processing_time_ms=processing_time,
                metadata={
                    "preserved_items": preserved_items,
                    "focus": config.summary_focus,
                    "async": True
                }
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            error_summary = self.formatter.format_error(tool_name, str(e), output)
            
            self.metrics.record(tool_name, original_tokens, original_tokens, False, processing_time)
            
            return SummaryResult(
                original_output=output,
                summary=error_summary,
                was_summarized=False,
                original_tokens=original_tokens,
                summary_tokens=original_tokens,
                tool_name=tool_name,
                processing_time_ms=processing_time,
                metadata={"error": str(e), "async": True}
            )
    
    async def _generate_summary_async(self, 
                                     output: str, 
                                     config: ToolConfig,
                                     context: Dict) -> str:
        """Generate summary asynchronously using aiohttp"""
        
        # Apply async rate limiting
        await self.rate_limiter.acquire_async()
        
        # Truncate very long outputs
        if len(output) > MAX_INPUT_CHARS:
            output = output[:MAX_INPUT_CHARS] + "\n... [truncated for summary]"
        
        prompt = f"""You are a tool output summarizer. Create a concise summary of the following tool output.

Tool: {config.name}
Context: {json.dumps(context)}
Focus on: {config.summary_focus}

Output to summarize:
```
{output}
```

Provide a brief summary (max 150 words) highlighting:
1. Key findings or results
2. Important files/locations mentioned
3. Any errors or warnings
4. Overall structure or pattern

Summary:"""
        
        session = await self._get_session()
        
        async with session.post(
            f"{self.ollama_endpoint}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": config.max_summary_tokens
                }
            },
            timeout=aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT)
        ) as response:
            if response.status != 200:
                raise Exception(f"Ollama error: {response.status}")
            
            result = await response.json()
            return result.get("response", "").strip()


class TokenEstimator:
    """
    Better token estimation for code and text.
    Uses a hybrid approach that's more accurate than simple char count.
    """
    
    # Average tokens per character by content type
    CODE_RATIO = 0.25  # Code: ~4 chars per token
    TEXT_RATIO = 0.30  # Prose: ~3.3 chars per token
    
    @classmethod
    def estimate(cls, text: str) -> int:
        """
        Estimate tokens more accurately.
        Uses different ratios for code-heavy vs text-heavy content.
        """
        if not text:
            return 0
        
        # Detect code-heavy content
        code_indicators = [
            r'[{};]',  # Braces and semicolons
            r'^(import|from|def|class|function|const|let|var|#include)',
            r'[=+\-*/<>!]+',  # Operators
        ]
        
        code_score = 0
        for indicator in code_indicators:
            code_score += len(re.findall(indicator, text, re.MULTILINE))
        
        # Calculate ratio based on code density
        code_density = code_score / max(len(text) / 100, 1)
        if code_density > 0.5:
            ratio = cls.CODE_RATIO
        else:
            ratio = cls.TEXT_RATIO
        
        return int(len(text) * ratio)


class RateLimiter:
    """
    Simple rate limiter for Ollama API calls.
    Prevents overwhelming the local model.
    """
    
    def __init__(self, calls_per_minute: int = RATE_LIMIT_CALLS_PER_MINUTE):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call_time = 0
        self.call_count = 0
        self.window_start = time.time()
    
    def acquire(self):
        """
        Acquire permission to make a call.
        Blocks if rate limit would be exceeded.
        """
        now = time.time()
        
        # Reset window if minute has passed
        if now - self.window_start >= 60:
            self.call_count = 0
            self.window_start = now
        
        # Check if we're at the limit
        if self.call_count >= self.calls_per_minute:
            sleep_time = 60 - (now - self.window_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.call_count = 0
            self.window_start = time.time()
        
        # Ensure minimum interval between calls
        time_since_last = now - self.last_call_time
        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)
        
        self.last_call_time = time.time()
        self.call_count += 1
    
    async def acquire_async(self):
        """Async version of acquire"""
        now = time.time()
        
        if now - self.window_start >= 60:
            self.call_count = 0
            self.window_start = now
        
        if self.call_count >= self.calls_per_minute:
            sleep_time = 60 - (now - self.window_start)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            self.call_count = 0
            self.window_start = time.time()
        
        time_since_last = now - self.last_call_time
        if time_since_last < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last)
        
        self.last_call_time = time.time()
        self.call_count += 1


# Convenience function for quick use
def summarize_tool_output(
    tool_name: str,
    output: str,
    ollama_endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
    context: Optional[Dict] = None
) -> str:
    """
    Quick function to summarize tool output.
    Returns summarized output or original if summarization not needed.
    """
    summarizer = ToolSummarizer(ollama_endpoint=ollama_endpoint)
    result = summarizer.summarize(tool_name, output, context)
    return result.summary


async def summarize_tool_output_async(
    tool_name: str,
    output: str,
    ollama_endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
    context: Optional[Dict] = None
) -> str:
    """
    Async version of summarize_tool_output.
    """
    summarizer = AsyncToolSummarizer(ollama_endpoint=ollama_endpoint)
    result = await summarizer.summarize(tool_name, output, context)
    return result.summary
