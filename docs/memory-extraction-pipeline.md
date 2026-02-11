# Memory Extraction Pipeline - Jetson Offload Design

## Overview

This document describes a cost-saving architecture that offloads memory extraction from Anthropic API (Claude) to Cru's Jetson AGX Orin running local LLMs via Ollama. This is the highest-impact token savings opportunity in the swarm.

**Target:** 80-90% reduction in Anthropic API token usage for memory operations  
**Current Cost:** ~$0.50-2.00 per conversation in extraction tokens  
**Projected Cost:** ~$0.05-0.20 per conversation (90% savings)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CONVERSATION FLOW                           │
└─────────────────────────────────────────────────────────────────┘

User Message → Moltbot → Agent Processing (Claude/Anthropic)
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Context Window      │
                    │   Nearing Limit?      │
                    └───────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        [No - Continue]          [Yes - Extract]
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                             │
                    ▼                                             ▼
        ┌─────────────────────┐                      ┌─────────────────────┐
        │  LOCAL EXTRACTION   │                      │  FALLBACK: Claude   │
        │  (Jetson/Ollama)    │                      │  (Anthropic API)    │
        │                     │                      │                     │
        │  qwen2.5-coder:32b  │                      │  If Jetson down     │
        │  or fast 7B variant │                      │  or errors          │
        └─────────────────────┘                      └─────────────────────┘
                    │                                             │
                    └──────────────┬──────────────────────────────┘
                                   ▼
                    ┌───────────────────────┐
                    │   Store in Memory API │
                    │   (Damon/swarm-memory)│
                    └───────────────────────┘
```

## Current Anthropic Memory Extraction Cost

### Per-Conversation Analysis

| Component | Tokens | Cost ($0.015/1k) | Notes |
|-----------|--------|------------------|-------|
| System Prompt | ~500 | $0.0075 | Fixed per extraction |
| Context Window (avg) | ~8,000 | $0.12 | Variable by conversation length |
| Response (extraction) | ~1,500 | $0.0225 | Output tokens |
| **Total per extraction** | **~10,000** | **~$0.15** | Triggered every 2-4 turns |
| **Monthly (1000 convos)** | **10M tokens** | **~$150** | Just for extractions |

### Token Burn Patterns

1. **High-frequency extractions** - Every 2-3 turns in active conversations
2. **Large context compaction** - 100k+ token windows getting summarized
3. **Redundant extractions** - Same facts extracted multiple times
4. **Tool result bloat** - Large exec outputs getting processed

## Jetson-Based Local Extraction

### Hardware Target

**Cru's Jetson AGX Orin (64GB)**
- GPU: 2048-core NVIDIA Ampere GPU
- RAM: 64GB unified memory
- Target Models: qwen2.5-coder:32b or qwen2.5-coder:7b-fast
- Expected Speed: 50-100 tok/s (7B) or 2-5 tok/s (32B)

### Model Selection

| Model | VRAM | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| qwen2.5-coder:7b | ~6GB | 80-100 tok/s | Good | Fast extraction, simple facts |
| qwen2.5-coder:32b | ~24GB | 2-5 tok/s | Excellent | Complex reasoning, decisions |
| llama3.1:8b | ~6GB | 60-80 tok/s | Good | General purpose fallback |

**Recommendation:** Use 7B for 90% of extractions, 32B for complex multi-turn decisions.

## Implementation Phases

### Phase 1: Basic Local Extraction (Week 1)

**Goal:** Replace Claude for simple memory extraction

#### Components

1. **Extraction Service** (`extractor.py` on Jetson)
   ```python
   # Runs on Jetson AGX Orin
   # Endpoint: http://192.168.85.158:11434 or https://ollama.swarmfactory.io
   
   class MemoryExtractor:
       def __init__(self, model="qwen2.5-coder:7b"):
           self.ollama_url = "http://localhost:11434"
           self.model = model
       
       async def extract(self, conversation_turns: List[Dict]) -> List[Memory]:
           """Extract memories from conversation context."""
           prompt = self._build_extraction_prompt(conversation_turns)
           response = await self._call_ollama(prompt)
           return self._parse_memories(response)
   ```

2. **Prompt Template** (Optimized for 7B model)
   ```
   You are a memory extraction system. Extract key information from the conversation.
   
   CONVERSATION:
   {conversation_text}
   
   EXTRACT as JSON array:
   - facts_learned: New facts, data, or information
   - decisions_made: Important decisions or commitments  
   - next_steps: Pending tasks or follow-ups
   - errors: Issues, bugs, or problems encountered
   
   Be concise. Only extract genuinely new information.
   
   JSON:
   ```

3. **Clawdbot Integration**
   ```python
   # In compaction-safeguard or memory-extractor extension
   
   async def extract_memories_locally(context: str) -> List[Memory]:
       """Try local extraction first, fall back to Claude."""
       try:
           # 1. Try Jetson (fast, cheap)
           memories = await jetson_extractor.extract(context)
           if memories and len(memories) > 0:
               return memories
       except Exception as e:
           logger.warning(f"Local extraction failed: {e}")
       
       # 2. Fall back to Claude (reliable, expensive)
       return await claude_extractor.extract(context)
   ```

#### Success Criteria
- [ ] 7B model produces valid extractions 90% of the time
- [ ] Fallback to Claude < 10% of extractions
- [ ] Average extraction time < 5 seconds

### Phase 2: Smart Filtering (Week 2)

**Goal:** Reduce extraction frequency by 50%

#### Components

1. **Duplicate Detection** (Local embedding + FAISS)
   ```python
   # Before extracting, check if similar memory exists
   similar = memory_api.query(
       query=new_fact_embedding,
       agent_id=agent_id,
       min_similarity=0.85
   )
   
   if similar and similar[0].content == new_fact:
       return []  # Skip extraction, already have it
   ```

2. **Change Detection**
   ```python
   # Only extract when state changes significantly
   if conversation_turns[-1].tokens < 500:
       return []  # Too small, skip
   
   if not has_new_information(last_extraction, current_context):
       return []  # No new info, skip
   ```

3. **Priority Scoring**
   ```python
   # Extract high-priority items immediately, batch low-priority
   priorities = {
       "error": 1.0,      # Always extract
       "decision": 0.9,   # Always extract
       "next_step": 0.8,  # Always extract
       "fact": 0.5,       # Batch hourly
       "chat": 0.1,       # Don't extract
   }
   ```

#### Success Criteria
- [ ] Extraction frequency reduced by 50%
- [ ] No loss of critical information
- [ ] Memory graph quality maintained

### Phase 3: Advanced Pipeline (Week 3-4)

**Goal:** Full context compaction offload

#### Components

1. **Hierarchical Summarization**
   ```
   Level 1 (Local/7B): Summarize each turn → 200 tokens
   Level 2 (Local/32B): Summarize 10 turns → 500 tokens  
   Level 3 (Claude if needed): Full compaction → 4k tokens
   ```

2. **Tool Result Filtering**
   ```python
   # Pre-filter large tool outputs before sending to Claude
   if len(tool_result) > 2000:
       summary = await jetson_extractor.summarize(tool_result)
       tool_result = summary  # Send 200 tokens instead of 10k
   ```

3. **RAG Pre-ranking**
   ```python
   # Local model ranks chunks before sending to Claude
   chunks = memory_api.query(query, limit=20)
   ranked = await jetson_extractor.rank_relevance(query, chunks)
   top_chunks = ranked[:3]  # Only send top 3 to Claude
   ```

#### Success Criteria
- [ ] 90% of context compaction done locally
- [ ] Tool result tokens reduced by 80%
- [ ] RAG query tokens reduced by 70%

## Deployment

### On Jetson AGX Orin (Cru's Machine)

```bash
# 1. Install Ollama (if not done)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull models
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:32b

# 3. Create Modelfile for optimized extraction
# /opt/ollama/Modelfile.extractor
FROM qwen2.5-coder:7b
PARAMETER temperature 0.3
PARAMETER num_ctx 8192
PARAMETER num_gpu 999
SYSTEM """You are a precise memory extraction system. Output only valid JSON. Be concise."""

# 4. Build custom model
ollama create extractor -f /opt/ollama/Modelfile.extractor

# 5. Extraction service
# /opt/swarm-app/services/extractor/main.py
cd /opt/swarm-app/services/extractor
pip install fastapi uvicorn aiohttp
python main.py  # Runs on port 8085
```

### On Clawdbot Agents

```python
# config.yaml addition
memory_extraction:
  mode: "hybrid"  # local, cloud, hybrid
  local_endpoint: "https://ollama.swarmfactory.io"  # or local IP
  local_model: "extractor"  # Custom model
  fallback_to_cloud: true
  min_confidence: 0.7
  
  # Cost-saving triggers
  use_local_for:
    - simple_facts: true
    - routine_extractions: true
    - tool_summaries: true
  
  use_cloud_for:
    - complex_reasoning: true
    - error_analysis: true
    - final_quality_check: true
```

## Cost Analysis

### Before (Current - All Anthropic)

| Metric | Value |
|--------|-------|
| Conversations/month | 1,000 |
| Extractions/convo | 3 |
| Tokens/extraction | 10,000 |
| Total tokens/month | 30M |
| Cost @ $0.015/1k | **$450/month** |

### After (Hybrid - Jetson + Anthropic)

| Metric | Value |
|--------|-------|
| Local extractions | 2,700 (90%) |
| Cloud extractions | 300 (10%) |
| Local tokens | 8M (avg 3k per) |
| Cloud tokens | 3M (avg 10k per) |
| Cloud cost | **$45/month** |
| Electricity (Jetson) | ~$5/month |
| **Total** | **$50/month** |

### Savings

| Metric | Value |
|--------|-------|
| Monthly savings | **$400** |
| Annual savings | **$4,800** |
| Percent reduction | **89%** |
| ROI (Jetson cost ~$2k) | 5 months |

## Testing & Validation

### Unit Tests

```python
# test_extraction_quality.py

def test_extraction_completeness():
    """Ensure local extraction captures key facts."""
    conversation = load_test_conversation("complex_task.json")
    
    local_memories = jetson_extractor.extract(conversation)
    claude_memories = claude_extractor.extract(conversation)
    
    # Local should capture 90%+ of what Claude captures
    coverage = calculate_coverage(local_memories, claude_memories)
    assert coverage > 0.90, f"Coverage {coverage} below threshold"

def test_extraction_accuracy():
    """Ensure extracted memories are factually correct."""
    test_cases = load_test_cases("accuracy_tests.json")
    
    for case in test_cases:
        memories = jetson_extractor.extract(case.conversation)
        accuracy = validate_against_ground_truth(memories, case.truth)
        assert accuracy > 0.95
```

### A/B Testing

1. Run both extractors in parallel for 1 week
2. Compare output quality using memory graph metrics
3. Measure token savings vs. information retention
4. Adjust confidence thresholds based on results

### Monitoring

```python
# metrics to track
- local_extraction_success_rate
- cloud_fallback_rate  
- extraction_latency_local_vs_cloud
- memory_graph_quality_score
- token_savings_per_day
```

## Fallback Strategy

### When to Use Anthropic (Cloud)

1. **Jetson is down** → Automatic fallback to Claude
2. **Extraction confidence < 0.7** → Re-extract with Claude
3. **Complex multi-turn reasoning** → Use Claude's better context understanding
4. **Error/debugging context** → Claude better at technical analysis

### Circuit Breaker Pattern

```python
class ExtractionCircuitBreaker:
    def __init__(self):
        self.local_failures = 0
        self.threshold = 5
        self.cooldown = 300  # 5 min
    
    async def extract(self, context):
        if self.local_failures >= self.threshold:
            return await claude_extractor.extract(context)
        
        try:
            result = await jetson_extractor.extract(context)
            self.local_failures = 0
            return result
        except Exception as e:
            self.local_failures += 1
            if self.local_failures >= self.threshold:
                alert_ops("Jetson extraction failing, using cloud fallback")
            return await claude_extractor.extract(context)
```

## Next Steps

1. **Cru:** Verify Ollama setup and model availability on Jetson
2. **Damon:** Deploy extraction service endpoint
3. **Nix:** Update Clawdbot agents to use hybrid extraction
4. **Cory:** Review and approve test plan
5. **Timeline:** 2 weeks to Phase 1 production

## Questions?

- Should we A/B test with specific agents first?
- What extraction quality threshold is acceptable? (90%? 95%?)
- Do we want real-time monitoring dashboard for the pipeline?
- Should we cache common extraction patterns?

---

**Author:** Damon (Memory Infrastructure)  
**Date:** 2026-02-11  
**Status:** Design Complete - Ready for Implementation  
**Target Savings:** $400/month (89% reduction in extraction costs)
