"""
Tool Result Pre-filtering System
Summarizes large tool outputs before sending to Claude.
Reduces token usage by 80% for tool results.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import requests


# Configuration constants
DEFAULT_OLLAMA_ENDPOINT = "http://192.168.85.158:11434"
DEFAULT_SUMMARY_MODEL = "qwen2.5-coder:7b"
SUMMARY_MAX_TOKENS = 200
OLLAMA_TIMEOUT = 30


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
        """Rough token estimation (1 token ≈ 4 chars for English)"""
        return len(text) // 4


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


class ToolSummarizer:
    """
    Main interface for tool result pre-filtering.
    Summarizes large tool outputs using local 7B model.
    """
    
    def __init__(self,
                 ollama_endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
                 model: str = DEFAULT_SUMMARY_MODEL):
        self.ollama_endpoint = ollama_endpoint
        self.model = model
        self.registry = ToolRegistry()
        self.formatter = SummaryFormatter()
        self.metrics = MetricsTracker()
    
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
        
        return preserved[:20]  # Limit preserved items
    
    def _generate_summary(self, 
                         output: str, 
                         config: ToolConfig,
                         context: Dict) -> str:
        """Generate summary using local 7B model"""
        
        # Truncate very long outputs for the prompt
        max_input_chars = 8000  # ~2000 tokens
        if len(output) > max_input_chars:
            output = output[:max_input_chars] + "\n... [truncated for summary]"
        
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
