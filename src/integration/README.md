# Clawdbot Agent Integration

Integration of hybrid memory extraction into all Clawdbot agents.

## Overview

This module provides the integration layer for Clawdbot agents to use the hybrid extraction system (local 7B/32B + Claude fallback) for all memory operations.

## Architecture

```
Clawdbot Agent
    │
    ▼
[Hybrid Extraction Extension]
    │
    ├── Local (7B) ──► Success? ──Yes──► Use Result
    │                    │
    │                    No
    │                    ▼
    ├── Local (32B) ──► Success? ──Yes──► Use Result
    │                    │
    │                    No
    │                    ▼
    └── Claude Fallback ──► Always works
```

## Components

- **HybridExtractor**: Main interface for hybrid extraction
- **CircuitBreaker**: Prevents cascading failures
- **AgentConfigurator**: Per-agent threshold configuration
- **ExtensionIntegrator**: Integrates with pi-extensions
- **DeploymentManager**: Deploys to all agents

## Per-Agent Configuration

| Agent | Primary | Fallback | Circuit Threshold |
|-------|---------|----------|-------------------|
| Damon | 7B | 32B → Claude | 5 failures/min |
| Percy | 7B | 32B → Claude | 5 failures/min |
| Nix | 7B | Claude | 3 failures/min |
| Max | 7B | Claude | 3 failures/min |
| G | 7B | Claude | 3 failures/min |
| Cru | 7B | Claude | 3 failures/min |

## Usage

```python
from clawdbot_integration import HybridExtractor

extractor = HybridExtractor.for_agent("damon")

# Extract memories
memories = await extractor.extract(
    conversation=conversation_turns,
    agent_id="damon"
)

# The extractor automatically:
# 1. Tries local 7B model first
# 2. Falls back to 32B if 7B fails
# 3. Falls back to Claude if both fail
# 4. Tracks success rates per model
```

## Deployment

```bash
# Deploy to all agents
python -m clawdbot_integration.deploy --all

# Deploy to specific agent
python -m clawdbot_integration.deploy --agent damon

# Verify deployment
python -m clawdbot_integration.verify
```
