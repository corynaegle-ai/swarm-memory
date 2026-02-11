"""
Batch Processing Service for Low-Priority Memory Extractions
Processes batched extractions on an hourly schedule.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import List, Dict
import aiohttp

from memory_filter import SmartFilter, FilterResult, MAX_CONTENT_LENGTH, BATCH_INTERVAL_MINUTES, MAX_BATCH_SIZE

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchExtractionService:
    """
    Service that processes batched low-priority memory extractions.
    Runs hourly or when batch size threshold reached.
    """
    
    def __init__(self,
                 memory_api_url: str = "https://memory.swarmfactory.io",
                 api_key: str = None,
                 batch_interval_minutes: int = BATCH_INTERVAL_MINUTES,
                 max_batch_size: int = MAX_BATCH_SIZE):
        self.memory_api_url = memory_api_url
        # Get API key from environment if not provided
        self.api_key = api_key or os.getenv("SWARM_MEMORY_API_KEY", "")
        if not self.api_key:
            logger.warning("SWARM_MEMORY_API_KEY not set - API calls will fail")
        
        self.batch_interval = batch_interval_minutes
        self.max_batch_size = max_batch_size
        self.filter = SmartFilter()
        self.running = False
    
    async def start(self):
        """Start the batch processing service"""
        self.running = True
        logger.info("Batch extraction service started")
        logger.info(f"  - Interval: {self.batch_interval} minutes")
        logger.info(f"  - Max batch size: {self.max_batch_size}")
        
        while self.running:
            try:
                # Check if batch should be processed
                if self.filter.batch_processor.should_process():
                    await self._process_batch()
                
                # Sleep for 1 minute between checks
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in batch service: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _process_batch(self):
        """Process the current batch of extractions"""
        batch = self.filter.batch_processor.get_batch()
        
        if not batch:
            return
        
        logger.info(f"Processing batch of {len(batch)} items")
        
        # Group by agent for efficiency
        by_agent: Dict[str, List[Dict]] = {}
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
        logger.info(f"Processing {len(items)} items for {agent_id}")
        
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
            content = context.get("content", "")[:MAX_CONTENT_LENGTH]  # Truncate long content
            priority = result.priority
            
            parts.append(f"[Priority {priority:.1f}] {content}")
        
        return "\n\n".join(parts)
    
    async def _store_extraction(self, agent_id: str, content: str, metadata: Dict):
        """Store extraction to memory API"""
        if not self.api_key:
            logger.error("Cannot store extraction: API key not configured")
            return
        
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
                        logger.info(f"Stored batch for {agent_id}")
                    else:
                        logger.error(f"Failed to store: HTTP {response.status}")
                        
        except Exception as e:
            logger.error(f"Error storing extraction: {e}", exc_info=True)
    
    def stop(self):
        """Stop the batch service"""
        self.running = False
        logger.info("Batch service stopping...")


# Standalone batch processor for command-line use
def process_batch_now():
    """Process any pending batches immediately"""
    filter = SmartFilter()
    batch = filter.process_batch()
    
    if not batch:
        logger.info("No batched items to process")
        return
    
    logger.info(f"Processing {len(batch)} batched items...")
    
    # Simple console output for now
    # In production, this would call the memory API
    for item in batch:
        context = item.get("context", {})
        logger.info(f"  - {context.get('agent_id', 'unknown')}: {context.get('content', '')[:100]}...")
    
    logger.info(f"Processed {len(batch)} items")


if __name__ == "__main__":
    # Can run as standalone service
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "process":
        # Process pending batch immediately
        process_batch_now()
    else:
        # Run as service
        logger.info("Starting batch extraction service...")
        logger.info("Use Ctrl+C to stop")
        
        service = BatchExtractionService()
        try:
            asyncio.run(service.start())
        except KeyboardInterrupt:
            logger.info("Stopping service...")
            service.stop()
