"""
Clawdbot Agent Integration
Integrates hybrid extraction into all Clawdbot agents.
"""

import asyncio
import json
import os
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

# Import our extraction modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filtering.memory_filter import SmartFilter, FilterResult
from prefilter.tool_prefilter import ToolSummarizer
from compaction.context_compaction_v2 import ContextCompactor, AsyncOllamaClient


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class ExtractionResult:
    """Result from hybrid extraction"""
    memories: List[Dict[str, Any]]
    source: str  # "7b", "32b", or "claude"
    tokens_saved: int
    cost_saved: float
    processing_time_ms: float
    cache_hit: bool
    success: bool
    error: Optional[str] = None


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_max_calls: int = 3
    
    def __post_init__(self):
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        self.half_open_calls = 0
    
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info("Circuit breaker entering HALF_OPEN state")
                return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls
        
        return True
    
    def record_success(self):
        """Record successful execution"""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.failures = 0
                logger.info("Circuit breaker CLOSED (recovered)")
        else:
            self.failures = max(0, self.failures - 1)
    
    def record_failure(self):
        """Record failed execution"""
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker OPEN (half-open failed)")
        elif self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker OPEN ({self.failures} failures)")


@dataclass
class AgentConfig:
    """Configuration for a Clawdbot agent"""
    name: str
    ollama_endpoint: str
    anthropic_api_key: Optional[str]
    use_32b_fallback: bool = True
    use_claude_fallback: bool = True
    circuit_threshold: int = 5
    cache_enabled: bool = True
    rate_limit_per_minute: int = 60
    
    # Quality thresholds
    min_quality_score: float = 0.85
    max_retries: int = 3


AGENT_CONFIGS = {
    "damon": AgentConfig(
        name="damon",
        ollama_endpoint="http://192.168.85.158:11434",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        use_32b_fallback=True,
        use_claude_fallback=True,
        circuit_threshold=5
    ),
    "percy": AgentConfig(
        name="percy",
        ollama_endpoint="http://192.168.85.158:11434",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        use_32b_fallback=True,
        use_claude_fallback=True,
        circuit_threshold=5
    ),
    "nix": AgentConfig(
        name="nix",
        ollama_endpoint="http://192.168.85.158:11434",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        use_32b_fallback=False,
        use_claude_fallback=True,
        circuit_threshold=3
    ),
    "max": AgentConfig(
        name="max",
        ollama_endpoint="http://192.168.85.158:11434",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        use_32b_fallback=False,
        use_claude_fallback=True,
        circuit_threshold=3
    ),
    "g": AgentConfig(
        name="g",
        ollama_endpoint="http://192.168.85.158:11434",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        use_32b_fallback=False,
        use_claude_fallback=True,
        circuit_threshold=3
    ),
    "cru": AgentConfig(
        name="cru",
        ollama_endpoint="http://192.168.85.158:11434",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        use_32b_fallback=False,
        use_claude_fallback=True,
        circuit_threshold=3
    )
}


class HybridExtractor:
    """
    Main interface for hybrid memory extraction.
    Coordinates local 7B, 32B, and Claude fallbacks.
    """
    
    def __init__(self, agent_name: str, config: Optional[AgentConfig] = None):
        self.agent_name = agent_name
        self.config = config or AGENT_CONFIGS.get(agent_name, AGENT_CONFIGS["damon"])
        
        # Initialize components
        self.ollama_client = AsyncOllamaClient(self.config.ollama_endpoint)
        self.filter = SmartFilter(embedding_endpoint=self.config.ollama_endpoint)
        self.compactor = ContextCompactor(
            ollama_endpoint=self.config.ollama_endpoint,
            anthropic_api_key=self.config.anthropic_api_key
        )
        
        # Circuit breakers for each model
        self.circuit_7b = CircuitBreaker(failure_threshold=self.config.circuit_threshold)
        self.circuit_32b = CircuitBreaker(failure_threshold=self.config.circuit_threshold)
        
        # Metrics
        self.metrics = {
            "total_calls": 0,
            "local_7b_success": 0,
            "local_32b_success": 0,
            "claude_fallback": 0,
            "total_tokens_saved": 0,
            "total_cost_saved": 0.0
        }
    
    async def close(self):
        """Cleanup resources"""
        await self.ollama_client.close()
        await self.compactor.close()
    
    async def extract(self, 
                     conversation: List[Dict[str, str]],
                     context: Optional[Dict] = None) -> ExtractionResult:
        """
        Extract memories using hybrid approach.
        
        Tries in order:
        1. Local 7B model (fastest, cheapest)
        2. Local 32B model (better quality, still cheap)
        3. Claude fallback (guaranteed quality, expensive)
        """
        start_time = time.time()
        self.metrics["total_calls"] += 1
        
        # Step 1: Try filtering first (skip if no new info)
        filter_result = self.filter.should_extract(conversation, self.agent_name)
        
        if not filter_result.should_extract:
            return ExtractionResult(
                memories=[],
                source="filtered",
                tokens_saved=0,
                cost_saved=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                cache_hit=False,
                success=True
            )
        
        # Step 2: Try local 7B model
        if self.circuit_7b.can_execute():
            try:
                result = await self._extract_with_7b(conversation)
                if result:
                    self.circuit_7b.record_success()
                    self.metrics["local_7b_success"] += 1
                    self.metrics["total_tokens_saved"] += result.tokens_saved
                    self.metrics["total_cost_saved"] += result.cost_saved
                    return result
            except Exception as e:
                logger.warning(f"7B extraction failed: {e}")
                self.circuit_7b.record_failure()
        
        # Step 3: Try local 32B model (if enabled)
        if self.config.use_32b_fallback and self.circuit_32b.can_execute():
            try:
                result = await self._extract_with_32b(conversation)
                if result:
                    self.circuit_32b.record_success()
                    self.metrics["local_32b_success"] += 1
                    self.metrics["total_tokens_saved"] += result.tokens_saved
                    self.metrics["total_cost_saved"] += result.cost_saved
                    return result
            except Exception as e:
                logger.warning(f"32B extraction failed: {e}")
                self.circuit_32b.record_failure()
        
        # Step 4: Fallback to Claude
        if self.config.use_claude_fallback:
            try:
                result = await self._extract_with_claude(conversation)
                self.metrics["claude_fallback"] += 1
                return result
            except Exception as e:
                logger.error(f"Claude fallback failed: {e}")
                return ExtractionResult(
                    memories=[],
                    source="failed",
                    tokens_saved=0,
                    cost_saved=0.0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    cache_hit=False,
                    success=False,
                    error=str(e)
                )
        
        # All options exhausted
        return ExtractionResult(
            memories=[],
            source="failed",
            tokens_saved=0,
            cost_saved=0.0,
            processing_time_ms=(time.time() - start_time) * 1000,
            cache_hit=False,
            success=False,
            error="All extraction methods failed"
        )
    
    async def _extract_with_7b(self, conversation: List[Dict[str, str]]) -> Optional[ExtractionResult]:
        """Extract using local 7B model"""
        start_time = time.time()
        
        # Use compaction to summarize conversation
        compaction_result = await self.compactor.compact(
            conversation, 
            max_output_tokens=2000
        )
        
        # Extract structured memories from summary
        memories = self._parse_memories(compaction_result.compacted_text)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Calculate savings (vs Claude)
        tokens_saved = compaction_result.token_savings
        cost_saved = tokens_saved * 0.000003  # Approximate Claude cost
        
        return ExtractionResult(
            memories=memories,
            source="7b",
            tokens_saved=tokens_saved,
            cost_saved=cost_saved,
            processing_time_ms=processing_time,
            cache_hit=compaction_result.metadata.get("cache_hit", False),
            success=True
        )
    
    async def _extract_with_32b(self, conversation: List[Dict[str, str]]) -> Optional[ExtractionResult]:
        """Extract using local 32B model"""
        # Similar to 7B but with 32B model
        # For now, use compaction which already uses 32B for level 2
        return await self._extract_with_7b(conversation)
    
    async def _extract_with_claude(self, conversation: List[Dict[str, str]]) -> ExtractionResult:
        """Extract using Claude (fallback)"""
        start_time = time.time()
        
        # Direct Claude extraction
        conversation_text = "\n\n".join([
            f"[{t.get('role', 'unknown')}]: {t.get('content', '')}"
            for t in conversation
        ])
        
        # Import here to avoid dependency if not used
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=self.config.anthropic_api_key)
        
        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": f"Extract key memories from this conversation:\n\n{conversation_text}"
            }]
        )
        
        memories = self._parse_memories(response.content[0].text)
        
        processing_time = (time.time() - start_time) * 1000
        
        return ExtractionResult(
            memories=memories,
            source="claude",
            tokens_saved=0,
            cost_saved=0.0,
            processing_time_ms=processing_time,
            cache_hit=False,
            success=True
        )
    
    def _parse_memories(self, text: str) -> List[Dict[str, Any]]:
        """Parse extracted memories from text"""
        memories = []
        
        # Simple parsing - look for bullet points or numbered lists
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('• '):
                memories.append({
                    "type": "fact",
                    "content": line[2:],
                    "timestamp": datetime.now().isoformat()
                })
            elif line.startswith('1.') or line.startswith('2.'):
                memories.append({
                    "type": "fact",
                    "content": line[3:],
                    "timestamp": datetime.now().isoformat()
                })
        
        return memories
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get extraction metrics"""
        total = self.metrics["total_calls"]
        if total == 0:
            return self.metrics
        
        return {
            **self.metrics,
            "local_7b_rate": self.metrics["local_7b_success"] / total,
            "local_32b_rate": self.metrics["local_32b_success"] / total,
            "claude_fallback_rate": self.metrics["claude_fallback"] / total,
            "avg_tokens_saved": self.metrics["total_tokens_saved"] / total,
            "avg_cost_saved": self.metrics["total_cost_saved"] / total,
            "circuit_7b_state": self.circuit_7b.state.value,
            "circuit_32b_state": self.circuit_32b.state.value
        }
    
    @classmethod
    def for_agent(cls, agent_name: str) -> "HybridExtractor":
        """Factory method for creating agent-specific extractor"""
        config = AGENT_CONFIGS.get(agent_name)
        if not config:
            raise ValueError(f"Unknown agent: {agent_name}")
        return cls(agent_name, config)


class AgentConfigurator:
    """
    Configures per-agent settings for hybrid extraction.
    """
    
    @staticmethod
    def get_config(agent_name: str) -> AgentConfig:
        """Get configuration for an agent"""
        return AGENT_CONFIGS.get(agent_name, AGENT_CONFIGS["damon"])
    
    @staticmethod
    def update_config(agent_name: str, **kwargs):
        """Update configuration for an agent"""
        if agent_name in AGENT_CONFIGS:
            config = AGENT_CONFIGS[agent_name]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    @staticmethod
    def list_agents() -> List[str]:
        """List all configured agents"""
        return list(AGENT_CONFIGS.keys())


class DeploymentManager:
    """
    Manages deployment of hybrid extraction to all agents.
    """
    
    AGENTS = ["damon", "percy", "nix", "max", "g", "cru"]
    
    @classmethod
    async def deploy_to_agent(cls, agent_name: str) -> bool:
        """Deploy hybrid extraction to a specific agent"""
        logger.info(f"Deploying to {agent_name}...")
        
        try:
            # Create extractor to verify it works
            extractor = HybridExtractor.for_agent(agent_name)
            
            # Test with simple conversation
            test_conv = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
            
            result = await extractor.extract(test_conv)
            await extractor.close()
            
            if result.success:
                logger.info(f"✓ {agent_name} deployed successfully")
                return True
            else:
                logger.error(f"✗ {agent_name} deployment failed: {result.error}")
                return False
                
        except Exception as e:
            logger.error(f"✗ {agent_name} deployment error: {e}")
            return False
    
    @classmethod
    async def deploy_all(cls) -> Dict[str, bool]:
        """Deploy to all agents"""
        results = {}
        
        for agent in cls.AGENTS:
            results[agent] = await cls.deploy_to_agent(agent)
        
        return results
    
    @classmethod
    def verify_deployment(cls) -> Dict[str, Dict]:
        """Verify deployment status of all agents"""
        status = {}
        
        for agent in cls.AGENTS:
            config = AGENT_CONFIGS.get(agent)
            status[agent] = {
                "configured": config is not None,
                "ollama_endpoint": config.ollama_endpoint if config else None,
                "has_claude_key": bool(config.anthropic_api_key) if config else False,
                "circuit_threshold": config.circuit_threshold if config else None
            }
        
        return status


# Convenience functions
async def extract_memories(agent_name: str, 
                          conversation: List[Dict[str, str]]) -> ExtractionResult:
    """Quick function to extract memories for an agent"""
    extractor = HybridExtractor.for_agent(agent_name)
    try:
        return await extractor.extract(conversation)
    finally:
        await extractor.close()


def get_agent_metrics(agent_name: str) -> Dict[str, Any]:
    """Get metrics for an agent"""
    # This would need persistent storage in production
    return {"agent": agent_name, "status": "not_implemented"}


async def run_health_check() -> Dict[str, bool]:
    """Run health check on all agents"""
    return await DeploymentManager.deploy_all()
