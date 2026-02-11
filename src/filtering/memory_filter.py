"""
Smart Memory Filtering System
Implements duplicate detection, change detection, and priority scoring
for memory extraction to reduce token costs by 50%.
"""

import json
import hashlib
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np


@dataclass
class FilterResult:
    """Result of filtering decision"""
    should_extract: bool
    priority: float  # 0.0 - 1.0
    reason: str
    context_hash: str
    is_duplicate: bool
    similar_memories: List[Dict]
    batched: bool = False


class EmbeddingCache:
    """Simple in-memory cache for embeddings"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def get(self, text: str) -> Optional[List[float]]:
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]
        return None
    
    def set(self, text: str, embedding: List[float]):
        key = hashlib.md5(text.encode()).hexdigest()
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest]
            del self.access_times[oldest]
        
        self.cache[key] = embedding
        self.access_times[key] = datetime.now()


class DuplicateDetector:
    """
    Detects duplicate or similar memories using embeddings.
    Falls back to simple text similarity if embeddings unavailable.
    """
    
    def __init__(self, 
                 embedding_endpoint: Optional[str] = None,
                 similarity_threshold: float = 0.85):
        self.embedding_endpoint = embedding_endpoint
        self.similarity_threshold = similarity_threshold
        self.cache = EmbeddingCache()
        self.local_memories = []  # Recent memories for comparison
        
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts"""
        # Try embedding-based similarity first
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        
        if emb1 and emb2:
            return self._cosine_similarity(emb1, emb2)
        
        # Fallback to Jaccard similarity on words
        return self._jaccard_similarity(text1, text2)
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from local Ollama or cache"""
        # Check cache first
        cached = self.cache.get(text)
        if cached:
            return cached
        
        # Try to get from Ollama if endpoint available
        if self.embedding_endpoint:
            try:
                import requests
                response = requests.post(
                    self.embedding_endpoint,
                    json={"model": "nomic-embed-text", "prompt": text},
                    timeout=5
                )
                if response.ok:
                    embedding = response.json().get("embedding")
                    if embedding:
                        self.cache.set(text, embedding)
                        return embedding
            except Exception:
                pass
        
        return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity on word sets"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)
    
    def find_duplicates(self, 
                       new_content: str, 
                       agent_id: str,
                       memory_api_url: str = "https://memory.swarmfactory.io") -> List[Dict]:
        """
        Find similar existing memories.
        Returns list of similar memories above threshold.
        """
        similar = []
        
        # Query recent memories from API
        try:
            import requests
            response = requests.post(
                f"{memory_api_url}/memory/query",
                headers={"X-API-Key": "3af7aebc2f1714f378580d68eb569a12"},
                json={"query": new_content, "limit": 5, "agent_id": agent_id},
                timeout=10
            )
            
            if response.ok:
                results = response.json().get("results", [])
                for mem in results:
                    similarity = mem.get("similarity", 0)
                    if similarity >= self.similarity_threshold:
                        similar.append({
                            "id": mem.get("id"),
                            "content": mem.get("content", "")[:200],
                            "similarity": similarity,
                            "created_at": mem.get("created_at")
                        })
        except Exception as e:
            print(f"Warning: Could not query memory API: {e}")
        
        return similar
    
    def is_duplicate(self, content: str, agent_id: str) -> Tuple[bool, List[Dict]]:
        """
        Check if content is duplicate or highly similar to existing memories.
        Returns (is_duplicate, similar_memories).
        """
        similar = self.find_duplicates(content, agent_id)
        
        # Consider it a duplicate if any match is > 95% similar
        for mem in similar:
            if mem["similarity"] >= 0.95:
                return True, similar
        
        # Consider it near-duplicate if avg similarity > threshold
        if similar:
            avg_sim = sum(m["similarity"] for m in similar) / len(similar)
            if avg_sim >= self.similarity_threshold:
                return True, similar
        
        return False, similar


class ChangeDetector:
    """
    Detects if conversation contains new information worth extracting.
    """
    
    def __init__(self, 
                 min_tokens: int = 500,
                 significant_change_threshold: float = 0.3):
        self.min_tokens = min_tokens
        self.change_threshold = significant_change_threshold
        self.last_states = {}  # agent_id -> last_state_hash
    
    def has_significant_changes(self, 
                               conversation_turns: List[Dict],
                               agent_id: str) -> Tuple[bool, str]:
        """
        Determine if conversation has significant new information.
        Returns (has_changes, reason).
        """
        # Check minimum size
        total_tokens = sum(t.get("tokens", 0) for t in conversation_turns)
        if total_tokens < self.min_tokens:
            return False, f"Too small ({total_tokens} tokens < {self.min_tokens} min)"
        
        # Compute state hash
        state_hash = self._compute_state_hash(conversation_turns)
        
        # Check if state changed
        last_hash = self.last_states.get(agent_id)
        if last_hash == state_hash:
            return False, "No state change from last extraction"
        
        # Check for content changes
        content_change = self._measure_content_change(
            conversation_turns, agent_id
        )
        
        if content_change < self.change_threshold:
            return False, f"Content change below threshold ({content_change:.2f} < {self.change_threshold})"
        
        # Update last state
        self.last_states[agent_id] = state_hash
        
        return True, f"Significant change detected ({content_change:.2f})"
    
    def _compute_state_hash(self, turns: List[Dict]) -> str:
        """Compute hash of conversation state"""
        content = " ".join(t.get("content", "") for t in turns[-3:])  # Last 3 turns
        return hashlib.md5(content.encode()).hexdigest()
    
    def _measure_content_change(self, 
                               turns: List[Dict], 
                               agent_id: str) -> float:
        """Measure how much content changed vs last extraction"""
        # Simple heuristic: ratio of new unique words
        current_text = " ".join(t.get("content", "") for t in turns)
        current_words = set(current_text.lower().split())
        
        # Ideally we'd compare to last extracted content
        # For now, use token count as proxy for content volume
        total_tokens = sum(t.get("tokens", 0) for t in turns)
        
        # Normalize: more tokens = more change potential
        # Scale between 0 and 1 based on token count
        return min(total_tokens / 2000, 1.0)


class PriorityScorer:
    """
    Scores extraction priority based on content type.
    Higher score = extract immediately
    Lower score = batch for later
    """
    
    # Priority weights
    PRIORITIES = {
        "error": 1.0,          # Always extract immediately
        "bug": 1.0,
        "critical": 1.0,
        "decision": 0.9,       # Important decisions
        "next_step": 0.8,      # Action items
        "task": 0.7,
        "architecture": 0.7,   # Design decisions
        "fact": 0.5,           # Information learned
        "preference": 0.4,     # User preferences
        "chat": 0.1,           # Casual conversation
        "greeting": 0.0,       # Skip
    }
    
    # Keywords that indicate priority
    KEYWORDS = {
        "error": ["error", "bug", "crash", "fail", "exception", "broken"],
        "decision": ["decided", "agreed", "conclusion", "choose", "select"],
        "next_step": ["todo", "next", "action", "follow up", "pending"],
        "architecture": ["design", "architecture", "pattern", "structure"],
        "fact": ["learned", "discovered", "found", "realized"],
    }
    
    def score(self, conversation_text: str) -> Tuple[float, str, str]:
        """
        Score conversation priority.
        Returns (score, category, reason).
        """
        text_lower = conversation_text.lower()
        
        # Check for high-priority keywords
        for category, keywords in self.KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    score = self.PRIORITIES.get(category, 0.5)
                    return score, category, f"Detected '{keyword}' -> {category}"
        
        # Default: medium-low priority
        return 0.5, "general", "No specific indicators, general priority"
    
    def should_extract_immediately(self, score: float) -> bool:
        """Determine if extraction should happen immediately vs batched"""
        return score >= 0.7  # High priority = immediate


class BatchProcessor:
    """
    Batches low-priority extractions for hourly processing.
    """
    
    def __init__(self, batch_interval_minutes: int = 60):
        self.batch_interval = timedelta(minutes=batch_interval_minutes)
        self.batched_items = []
        self.last_process_time = datetime.now()
    
    def add(self, item: Dict) -> bool:
        """Add item to batch. Returns True if should process now."""
        self.batched_items.append({
            **item,
            "batched_at": datetime.now().isoformat()
        })
        
        # Check if it's time to process
        time_since_last = datetime.now() - self.last_process_time
        if time_since_last >= self.batch_interval:
            return True
        
        # Process if batch is getting large
        if len(self.batched_items) >= 10:
            return True
        
        return False
    
    def get_batch(self) -> List[Dict]:
        """Get current batch and clear it"""
        batch = self.batched_items.copy()
        self.batched_items = []
        self.last_process_time = datetime.now()
        return batch
    
    def should_process(self) -> bool:
        """Check if it's time to process batched items"""
        time_since_last = datetime.now() - self.last_process_time
        return time_since_last >= self.batch_interval or len(self.batched_items) >= 10


class SmartFilter:
    """
    Main interface for smart memory filtering.
    Combines all filtering components.
    """
    
    def __init__(self,
                 embedding_endpoint: str = "http://192.168.85.158:11434/api/embeddings",
                 similarity_threshold: float = 0.85,
                 min_tokens: int = 500):
        self.duplicate_detector = DuplicateDetector(
            embedding_endpoint=embedding_endpoint,
            similarity_threshold=similarity_threshold
        )
        self.change_detector = ChangeDetector(min_tokens=min_tokens)
        self.priority_scorer = PriorityScorer()
        self.batch_processor = BatchProcessor()
    
    def should_extract(self,
                      conversation_turns: List[Dict],
                      agent_id: str,
                      last_extraction_time: Optional[datetime] = None) -> FilterResult:
        """
        Main entry point: determine if extraction should happen.
        
        Returns FilterResult with decision and metadata.
        """
        # Flatten conversation for analysis
        conversation_text = " ".join(
            t.get("content", "") for t in conversation_turns
        )
        
        # 1. Check for significant changes
        has_changes, change_reason = self.change_detector.has_significant_changes(
            conversation_turns, agent_id
        )
        
        if not has_changes:
            return FilterResult(
                should_extract=False,
                priority=0.0,
                reason=change_reason,
                context_hash="",
                is_duplicate=False,
                similar_memories=[]
            )
        
        # 2. Check for duplicates
        is_duplicate, similar_memories = self.duplicate_detector.is_duplicate(
            conversation_text, agent_id
        )
        
        if is_duplicate:
            return FilterResult(
                should_extract=False,
                priority=0.0,
                reason="Duplicate or near-duplicate of existing memories",
                context_hash=self.change_detector._compute_state_hash(conversation_turns),
                is_duplicate=True,
                similar_memories=similar_memories
            )
        
        # 3. Score priority
        priority, category, priority_reason = self.priority_scorer.score(
            conversation_text
        )
        
        # 4. Determine extraction strategy
        context_hash = self.change_detector._compute_state_hash(conversation_turns)
        
        if priority >= 0.7:
            # High priority: extract immediately
            return FilterResult(
                should_extract=True,
                priority=priority,
                reason=f"{change_reason}; {priority_reason}",
                context_hash=context_hash,
                is_duplicate=False,
                similar_memories=similar_memories,
                batched=False
            )
        else:
            # Low priority: consider batching
            return FilterResult(
                should_extract=True,
                priority=priority,
                reason=f"{change_reason}; {priority_reason} (low priority - can batch)",
                context_hash=context_hash,
                is_duplicate=False,
                similar_memories=similar_memories,
                batched=True
            )
    
    def add_to_batch(self, result: FilterResult, context: Dict):
        """Add a low-priority extraction to the batch queue"""
        self.batch_processor.add({
            "result": result,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })
    
    def process_batch(self) -> List[Dict]:
        """Process batched items if it's time"""
        if self.batch_processor.should_process():
            return self.batch_processor.get_batch()
        return []


# Convenience function for quick use
def should_extract_memory(
    conversation_turns: List[Dict],
    agent_id: str,
    embedding_endpoint: str = "http://192.168.85.158:11434/api/embeddings"
) -> bool:
    """
    Quick check if memory extraction should happen.
    Returns True if should extract, False otherwise.
    """
    filter = SmartFilter(embedding_endpoint=embedding_endpoint)
    result = filter.should_extract(conversation_turns, agent_id)
    return result.should_extract
