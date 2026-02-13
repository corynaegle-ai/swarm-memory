# Swarm Memory Extraction Deployment Guide

This guide covers deploying the memory extraction system to Clawdbot/OpenClaw agents so they automatically store memories to the swarm memory system during compaction.

## Overview

The memory extraction system allows agents to:
- Automatically extract key information during session compaction
- Store memories to the centralized swarm memory API
- Retrieve relevant memories after compaction
- Maintain context across session resets

## Prerequisites

- Agent must have network access to `memory.swarmfactory.io`
- Python 3.8+ installed
- `requests` library (for synchronous storage)
- `aiohttp` library (for async operations)

## Deployment Steps

### 1. Create Directory Structure

On the target agent, create the pi-extensions directory:

```bash
mkdir -p /root/clawd/pi-extensions
```

### 2. Install Dependencies

```bash
pip3 install requests aiohttp --break-system-packages
```

### 3. Create Memory Extractor

Create `/root/clawd/pi-extensions/agent_memory_extractor.py`:

```python
#!/usr/bin/env python3
"""
Memory extractor for [AGENT_NAME]
Sends compaction summaries to swarm memory
"""
import os
import json
import requests
from datetime import datetime

AGENT_ID = "[AGENT_ID]"  # e.g., "percy", "max", "g", "nix", "cru"
MEMORY_API_URL = "https://memory.swarmfactory.io"
MEMORY_API_KEY = "[API_KEY]"

def store_memory(content, metadata=None):
    """Store a memory to swarm memory"""
    metadata = metadata or {}
    
    payload = {
        "content": content,
        "agent_id": AGENT_ID,
        "tags": metadata.get("tags", ["auto", "compaction"]),
        "importance": metadata.get("importance", 0.7)
    }
    
    try:
        resp = requests.post(
            f"{MEMORY_API_URL}/memory/store",
            headers={
                "Content-Type": "application/json",
                "X-API-Key": MEMORY_API_KEY
            },
            json=payload,
            timeout=10
        )
        
        if resp.status_code == 200:
            print(f"✓ Memory stored: {resp.json().get('id', 'unknown')}")
            return True
        else:
            print(f"✗ Failed: {resp.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error storing memory: {e}")
        return False

print(f"✓ {AGENT_ID} memory extractor loaded")
print(f"  API: {MEMORY_API_URL}")
```

**Replace:**
- `[AGENT_ID]` with the agent's name (e.g., "percy", "max")
- `[API_KEY]` with the swarm memory API key

### 4. Create Agent Configuration

Create `/root/clawd/pi-extensions/agent_config.json`:

```json
{
  "agent_id": "[AGENT_ID]",
  "agent_name": "[AGENT_NAME]",
  "memory_api_url": "https://memory.swarmfactory.io",
  "memory_api_key": "[API_KEY]",
  "compaction": {
    "enabled": true,
    "extract_memories": true,
    "store_to_swarm": true
  }
}
```

### 5. Update OpenClaw Configuration

Add to `~/.openclaw/openclaw.json`:

```json
{
  "env": {
    "MEMORY_API_URL": "https://memory.swarmfactory.io",
    "MEMORY_API_KEY": "[API_KEY]",
    "AGENT_ID": "[AGENT_ID]"
  },
  "agents": {
    "defaults": {
      "compaction": {
        "mode": "safeguard",
        "extractToMemory": true,
        "memoryApiUrl": "https://memory.swarmfactory.io"
      }
    }
  }
}
```

### 6. Test Memory Storage

Test the extractor:

```bash
cd /root/clawd && python3 << 'PYTEST'
import sys
sys.path.insert(0, 'pi-extensions')
from agent_memory_extractor import store_memory

result = store_memory(
    "[TEST] Memory extraction system deployed and tested.",
    {"tags": ["test", "deployment"], "importance": 0.8}
)

if result:
    print('\n✅ Memory extraction working!')
else:
    print('\n⚠️ Test failed - check API connectivity')
PYTEST
```

### 7. Restart Agent

Restart the agent to pick up new configuration:

```bash
# For OpenClaw
pkill -f openclaw-gateway
export MEMORY_API_URL=https://memory.swarmfactory.io
export MEMORY_API_KEY=[API_KEY]
export AGENT_ID=[AGENT_ID]
nohup openclaw-gateway > /tmp/openclaw.log 2>&1 &

# For Clawdbot
clawdbot gateway restart
```

## Configuration Reference

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MEMORY_API_URL` | Swarm memory API endpoint | `https://memory.swarmfactory.io` |
| `MEMORY_API_KEY` | API key for authentication | `3af7aebc2f1714f378580d68eb569a12` |
| `AGENT_ID` | Unique agent identifier | `percy`, `max`, `g` |

### Memory Storage Format

Memories are stored with this structure:

```json
{
  "content": "Memory text content...",
  "agent_id": "percy",
  "tags": ["auto", "compaction"],
  "importance": 0.7,
  "created_at": "2026-02-13T15:00:00Z"
}
```

## Troubleshooting

### Issue: Memories not appearing in swarm memory

**Check:**
1. API connectivity: `curl -I https://memory.swarmfactory.io`
2. API key is valid
3. Agent ID is set correctly
4. Check extractor logs for errors

### Issue: Test storage fails

**Verify:**
```bash
curl -X POST https://memory.swarmfactory.io/memory/store \
  -H "X-API-Key: [API_KEY]" \
  -H "Content-Type: application/json" \
  -d '{"content": "test", "agent_id": "test"}'
```

### Issue: Agent won't restart

**Check:**
```bash
ps aux | grep -E 'openclaw|clawdbot'
tail -50 /tmp/openclaw.log
```

## Verification

After deployment, verify memories are being stored:

```bash
# Query for recent agent memories
curl -X POST https://memory.swarmfactory.io/memory/query \
  -H "X-API-Key: [API_KEY]" \
  -H "Content-Type: application/json" \
  -d '{"query": "*", "agent_id": "[AGENT_ID]", "limit": 5}'
```

## Agent-Specific Notes

### Percy (PROD Server)
- Location: 146.190.35.235
- Config: `~/.openclaw/openclaw.json`
- Note: Running OpenClaw (not Clawdbot)

### Max (swarm-host)
- Location: 192.168.85.124
- User: cnaegle
- Config: `~/.clawdbot/config.json`

### G (Jetson Nano)
- Location: 192.168.85.157
- Note: Limited resources, use lightweight extractor

### Nix (MacBook)
- Location: Local/variable
- Config: `~/.clawdbot/config.json`

### Cru (Audio Engine)
- Location: Variable
- Note: May have different directory structure

## Rollback

To disable memory extraction:

```bash
# Remove environment variables from ~/.bashrc
sed -i '/MEMORY_API/d' ~/.bashrc

# Disable in config
# Set "extractToMemory": false in openclaw.json

# Restart agent
pkill -f openclaw-gateway
```

## API Reference

### Memory Store Endpoint

```
POST https://memory.swarmfactory.io/memory/store
Headers:
  X-API-Key: [API_KEY]
  Content-Type: application/json

Body:
{
  "content": "Memory text",
  "agent_id": "agent_name",
  "tags": ["tag1", "tag2"],
  "importance": 0.7
}
```

### Memory Query Endpoint

```
POST https://memory.swarmfactory.io/memory/query
Headers:
  X-API-Key: [API_KEY]
  Content-Type: application/json

Body:
{
  "query": "search terms",
  "agent_id": "optional_filter",
  "limit": 10
}
```

## Support

For issues or questions:
- Check swarm memory dashboard: http://134.199.235.140:8000/dashboard
- Query memory stats: https://memory.swarmfactory.io/memory/stats
- Contact: Damon (memory infrastructure host)

---

**Last Updated:** 2026-02-13
**Version:** 1.0
