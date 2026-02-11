# Smart Memory Filtering System

This module implements intelligent filtering and duplicate detection for memory extraction, reducing extraction frequency by 50% while maintaining quality.

## Components

- **DuplicateDetector**: Uses embeddings + FAISS to detect similar memories
- **ChangeDetector**: Determines if new information exists in context
- **PriorityScorer**: Scores extraction priority (error > decision > next_step > fact > chat)
- **BatchProcessor**: Batches low-priority extractions for hourly processing

## Architecture

```
Conversation Turn
    │
    ▼
[Change Detector] ──No new info?──→ Skip
    │ Yes
    ▼
[Priority Scorer]
    │
    ├── High Priority (error/decision) ──→ Extract Immediately
    │
    └── Low Priority (fact/chat) ──→ [Batch Queue] ──→ Hourly Batch
```

## Usage

```python
from memory_filter import SmartFilter

filter = SmartFilter(
    embedding_endpoint="http://192.168.85.158:11434/api/embeddings",
    similarity_threshold=0.85
)

# Process a conversation turn
result = filter.should_extract(
    conversation_turns=turns,
    agent_id="damon",
    last_extraction_time=last_time
)

if result.should_extract:
    if result.priority >= 0.8:
        # Extract immediately
        extract_now(result.context)
    else:
        # Add to batch queue
        filter.add_to_batch(result)
```
