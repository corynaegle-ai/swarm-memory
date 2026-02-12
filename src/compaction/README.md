# Context Compaction Offload System

Hierarchical context compaction using local models (7B/32B) with Claude fallback.

## Architecture

```
Raw Context (100k+ tokens)
    │
    ▼
[Level 1: 7B Model] ──Each turn → 200 tokens──┐
    │                                           │
    ▼                                           │
Level 1 Summary (~2k tokens)                    │
    │                                           │
    ▼                                           │
[Level 2: 32B Model] ──10 turns → 500 tokens──┤
    │                                           │
    ▼                                           │
Level 2 Summary (~1k tokens)                    │
    │                                           │
    ▼                                           │
[Still too large?] ──Yes──► [Level 3: Claude] │
    │ No                                        │
    ▼                                           │
Use Level 2 Summary ◄──────────────────────────┘
```

## Components

- **ContextCompactor**: Main orchestrator for 3-level hierarchy
- **Level1Summarizer**: 7B model, per-turn summarization (~200 tokens)
- **Level2Summarizer**: 32B model, multi-turn summarization (~500 tokens)
- **Level3Summarizer**: Claude fallback for extreme cases (~4k tokens)
- **RAGPreRanker**: Local model ranking for RAG queries
- **CompactionMetrics**: Track token savings across levels

## Usage

```python
from context_compaction import ContextCompactor

compactor = ContextCompactor(
    ollama_endpoint="http://192.168.85.158:11434"
)

# Compact large context
result = compactor.compact(
    conversation_history=large_history,
    max_output_tokens=4000,
    agent_id="damon"
)

print(f"Reduced {result.original_tokens} → {result.compacted_tokens}")
print(f"Levels used: {result.levels_used}")
```

## Token Savings by Level

| Level | Model | Input | Output | Savings |
|-------|-------|-------|--------|---------|
| 0 (raw) | - | 100k tokens | 100k | 0% |
| 1 (7B) | qwen2.5:7b | 100k | 2k | 98% |
| 2 (32B) | qwen2.5:32b | 2k | 500 | 75% |
| 3 (Claude) | claude-3-5 | 500 | 4k | -700% (expansion) |

## RAG Pre-Ranking

Before sending RAG queries to the embedding model:
1. Use local 7B to pre-rank documents
2. Filter to top-K most relevant
3. Only embed top-K, reducing embedding API calls by 70%
