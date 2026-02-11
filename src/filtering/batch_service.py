"""
Batch Processing Service for Low-Priority Memory Extractions
Processes batched extractions on an hourly schedule.
"""

import json
import asyncio
from datetime import datetime
from typing import List, Dict
import aiohttp

from memory_filter import SmartFilter, FilterResult


class BatchExtractionService:
    """
    Service that processes batched low-priority memory extractions.
    Runs hourly or when batch size threshold reached.
    """
    
    def __init__(self,
                 memory_api_url: str = "https://memory.swarmfactory.io",
                 api_key: str = "3af7aebc2f1714f378580d68eb569a12",
                 batch_interval_minutes: int = 60,
                 max_batch_size: int = 10):
        self.memory_api_url = memory_api_url
        self.api_key = api_key
        self.batch_interval = batch_interval_minutes
        self.max_batch_size = max_batch_size
        self.filter = SmartFilter()
        self.running = False
    
    async def start(self):
        """Start the batch processing service"""
        self.running = True
        print(f"[{datetime.now()}] Batch extraction service started")
        print(f"  - Interval: {self.batch_interval} minutes")
        print(f"  - Max batch size: {self.max_batch_size}")
        
        while self.running:
            try:
                # Check if batch should be processed
                if self.filter.batch_processor.should_process():
                    await self._process_batch()
                
                # Sleep for 1 minute between checks
                await asyncio.sleep(60)
                
            except Exception as e:
                print(f"[{datetime.now()}] Error in batch service: {e}")
                await asyncio.sleep(60)
    
    async def _process_batch(self):
        """Process the current batch of extractions"""
        batch = self.filter.batch_processor.get_batch()
        
        if not batch:
            return
        
        print(f"[{datetime.now()}] Processing batch of {len(batch)} items")
        
        # Group by agent for efficiency
        by_agent = {}
        for item in batch:
            agent_id = item.get("context", {}).get("agent_id", "unknown")
            if agent_id not in by_agent:
                by_agent[agent_id] = []
            by_agent[agent_id].append(item)
        
        # Process each agent's batch
        for agent_id, items in by_agent.items():
            await self._process_agent_batch(agent_id, items)
    
    async def _process_agent_batch(self, agent_id: str, items: List[Dict]):
        """Process batched items for a single agent"""
        print(f"  Processing {len(items)} items for {agent_id}")
        
        # Combine related extractions
        combined_content = self._combine_extractions(items)
        
        # Store to memory API
        await self._store_extraction(agent_id, combined_content, {
            "source": "batch_processor",
            "item_count": len(items),
            "batched": True
        })
    
    def _combine_extractions(self, items: List[Dict]) -> str:
        """Combine multiple batched extractions into one"""
        parts = []
        for item in items:
            context = item.get("context", {})
            result = item.get("result", FilterResult(
                should_extract=True, priority=0.5, reason="",
                context_hash="", is_duplicate=False, similar_memories=[]
            ))
            
            # Extract key information
            content = context.get("content", "")[:500]  # Truncate long content
            priority = result.priority
            
            parts.append(f"[Priority {priority:.1f}] {content}")
        
        return "\n\n".join(parts)
    
    async def _store_extraction(self, agent_id: str, content: str, metadata: Dict):
        """Store extraction to memory API"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "agent_id": agent_id,
                    "content": content,
                    "tags": ["memory", "batched", "filtered"],
                    "importance": 0.6,  # Batched items are lower importance
                    "metadata": metadata
                }
                
                async with session.post(
                    f"{self.memory_api_url}/memory/store",
                    headers={"X-API-Key": self.api_key},
                    json=payload
                ) as response:
                    if response.status == 200:
                        print(f"    ✓ Stored batch for {agent_id}")
                    else:
                        print(f"    ✗ Failed to store: {response.status}")
                        
        except Exception as e:
            print(f"    ✗ Error storing extraction: {e}")
    
    def stop(self):
        """Stop the batch service"""
        self.running = False


# Standalone batch processor for command-line use
def process_batch_now():
    """Process any pending batches immediately"""
    import requests
    
    filter = SmartFilter()
    batch = filter.process_batch()
    
    if not batch:
        print("No batched items to process")
        return
    
    print(f"Processing {len(batch)} batched items...")
    
    # Simple console output for now
    # In production, this would call the memory API
    for item in batch:
        context = item.get("context", {})
        print(f"  - {context.get('agent_id', 'unknown')}: {context.get('content', '')[:100]}...")
    
    print(f"\nProcessed {len(batch)} items")


if __name__ == "__main__":
    # Can run as standalone service
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "process":
        # Process pending batch immediately
        process_batch_now()
    else:
        # Run as service
        print("Starting batch extraction service...")
        print("Use Ctrl+C to stop")
        
        service = BatchExtractionService()
        try:
            asyncio.run(service.start())
        except KeyboardInterrupt:
            print("\nStopping service...")
            service.stop()
