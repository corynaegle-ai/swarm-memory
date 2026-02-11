# Tool Result Pre-filtering System

Summarizes large tool outputs before sending to Claude, reducing token usage by 80%.

## Problem

Tool outputs (file listings, logs, search results) often exceed 2k tokens but contain mostly noise. Sending raw output to Claude wastes tokens.

## Solution

Use local 7B model (qwen2.5-coder:7b) to pre-summarize tool outputs:
- **Input:** Raw tool output (2k-10k tokens)
- **Output:** Structured summary (~200 tokens)
- **Savings:** 80-90% token reduction

## Components

- **ToolSummarizer**: Main interface for summarizing tool outputs
- **ToolRegistry**: Configuration per tool type
- **SummaryFormatter**: Consistent output format
- **MetricsTracker**: Token savings measurement

## Supported Tools

| Tool | Threshold | Summary Focus |
|------|-----------|---------------|
| file_read | 1000 tokens | Key sections, imports, structure |
| directory_list | 500 tokens | Directory structure, file counts |
| search_results | 1000 tokens | Top matches, context snippets |
| log_output | 2000 tokens | Error patterns, recent entries |
| http_response | 1000 tokens | Status, key headers, body summary |
| database_query | 1000 tokens | Row count, sample data, schema |

## Usage

```python
from tool_prefilter import ToolSummarizer

summarizer = ToolSummarizer(
    ollama_endpoint="http://192.168.85.158:11434"
)

# Summarize tool output
result = summarizer.summarize(
    tool_name="directory_list",
    output=large_directory_listing,
    context={"path": "/src"}
)

if result.was_summarized:
    print(f"Reduced {result.original_tokens} → {result.summary_tokens} tokens")
    claude_input = result.summary
else:
    claude_input = result.original_output
```

## Architecture

```
Tool Output
    │
    ▼
[Size Check] ──< threshold?──→ Send raw
    │ >= threshold
    ▼
[Tool Registry] → Get summarization config
    │
    ▼
[Local 7B Model] → Generate summary
    │
    ▼
[Structured Output] → Send to Claude
```
