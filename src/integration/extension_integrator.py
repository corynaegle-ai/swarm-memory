"""
Pi-Extension Integration
Integrates hybrid extraction with Clawdbot pi-extensions.
"""

import asyncio
from typing import Dict, List, Any, Optional
from clawdbot_integration import HybridExtractor, ExtractionResult


class CompactionSafeguardExtension:
    """
    Extension that adds compaction safeguard to Clawdbot.
    Replaces the original compaction-safeguard with hybrid extraction.
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.extractor: Optional[HybridExtractor] = None
    
    async def initialize(self):
        """Initialize the extension"""
        self.extractor = HybridExtractor.for_agent(self.agent_name)
    
    async def shutdown(self):
        """Cleanup resources"""
        if self.extractor:
            await self.extractor.close()
    
    async def on_conversation_complete(self, 
                                      conversation: List[Dict[str, str]]) -> ExtractionResult:
        """
        Called when a conversation completes.
        Extracts memories using hybrid approach.
        """
        if not self.extractor:
            await self.initialize()
        
        result = await self.extractor.extract(conversation)
        
        # Log metrics
        print(f"[{self.agent_name}] Extracted {len(result.memories)} memories from {result.source}")
        print(f"  Tokens saved: {result.tokens_saved}, Cost saved: ${result.cost_saved:.4f}")
        
        return result
    
    async def get_status(self) -> Dict[str, Any]:
        """Get extension status"""
        if not self.extractor:
            return {"status": "not_initialized"}
        
        metrics = self.extractor.get_metrics()
        return {
            "status": "active",
            "metrics": metrics,
            "circuit_7b": self.extractor.circuit_7b.state.value,
            "circuit_32b": self.extractor.circuit_32b.state.value
        }


class MemoryExtractorExtension:
    """
    Extension that replaces memory-extractor with hybrid version.
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.extractor: Optional[HybridExtractor] = None
    
    async def initialize(self):
        """Initialize the extension"""
        self.extractor = HybridExtractor.for_agent(self.agent_name)
    
    async def shutdown(self):
        """Cleanup resources"""
        if self.extractor:
            await self.extractor.close()
    
    async def extract(self, 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract memories from context.
        
        Args:
            context: Dict with 'conversation' key containing turns
        
        Returns:
            Dict with extracted memories and metadata
        """
        if not self.extractor:
            await self.initialize()
        
        conversation = context.get("conversation", [])
        
        result = await self.extractor.extract(conversation)
        
        return {
            "memories": result.memories,
            "source": result.source,
            "success": result.success,
            "tokens_saved": result.tokens_saved,
            "cost_saved": result.cost_saved,
            "processing_time_ms": result.processing_time_ms
        }


class ExtensionIntegrator:
    """
    Integrates hybrid extraction extensions with Clawdbot.
    """
    
    @staticmethod
    def install_compaction_safeguard(agent_name: str):
        """
        Install compaction safeguard extension for an agent.
        
        In production, this would modify the agent's configuration files.
        """
        print(f"Installing compaction-safeguard extension for {agent_name}...")
        
        # Create extension instance
        extension = CompactionSafeguardExtension(agent_name)
        
        # In production, this would:
        # 1. Add to agent's extension list
        # 2. Configure hooks
        # 3. Set up logging
        
        print(f"✓ Installed for {agent_name}")
        return extension
    
    @staticmethod
    def install_memory_extractor(agent_name: str):
        """Install memory extractor extension for an agent"""
        print(f"Installing memory-extractor extension for {agent_name}...")
        
        extension = MemoryExtractorExtension(agent_name)
        
        print(f"✓ Installed for {agent_name}")
        return extension
    
    @staticmethod
    def install_all_extensions(agent_name: str) -> Dict[str, Any]:
        """Install all hybrid extraction extensions for an agent"""
        return {
            "compaction_safeguard": ExtensionIntegrator.install_compaction_safeguard(agent_name),
            "memory_extractor": ExtensionIntegrator.install_memory_extractor(agent_name)
        }
    
    @staticmethod
    def get_extension_config(agent_name: str) -> Dict[str, Any]:
        """Get extension configuration for an agent"""
        return {
            "agent": agent_name,
            "extensions": [
                {
                    "name": "compaction-safeguard",
                    "enabled": True,
                    "priority": 1,
                    "config": {
                        "use_hybrid": True,
                        "fallback_enabled": True
                    }
                },
                {
                    "name": "memory-extractor",
                    "enabled": True,
                    "priority": 2,
                    "config": {
                        "use_hybrid": True,
                        "cache_enabled": True
                    }
                }
            ]
        }


# Installation script
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extension_integrator.py <agent_name> [--all]")
        sys.exit(1)
    
    agent_name = sys.argv[1]
    
    if agent_name == "--all":
        agents = ["damon", "percy", "nix", "max", "g", "cru"]
        for agent in agents:
            ExtensionIntegrator.install_all_extensions(agent)
    else:
        ExtensionIntegrator.install_all_extensions(agent_name)
        config = ExtensionIntegrator.get_extension_config(agent_name)
        print(f"\nConfiguration for {agent_name}:")
        print(json.dumps(config, indent=2))
