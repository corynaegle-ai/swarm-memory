"""
Testing & Validation Framework
Comprehensive testing for memory extraction quality.
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import numpy as np
from difflib import SequenceMatcher


@dataclass
class GroundTruth:
    """Ground truth data for a conversation"""
    facts: List[str]
    decisions: List[str]
    next_steps: List[str]
    entities: Dict[str, List[str]]
    summary: Optional[str] = None


@dataclass
class TestConversation:
    """A single test case"""
    id: str
    domain: str
    difficulty: str
    turns: List[Dict[str, str]]
    ground_truth: GroundTruth
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result from extraction (local or Claude)"""
    facts: List[str]
    decisions: List[str]
    next_steps: List[str]
    entities: Dict[str, List[str]]
    summary: Optional[str] = None
    raw_output: str = ""
    tokens_used: int = 0
    cost: float = 0.0
    processing_time_ms: float = 0.0


@dataclass
class QualityMetrics:
    """Quality metrics for extraction"""
    coverage: float  # % of ground truth extracted
    accuracy: float  # % of extracted that are correct
    precision: float  # TP / (TP + FP)
    recall: float  # TP / (TP + FN)
    f1_score: float  # 2 * (P * R) / (P + R)
    semantic_similarity: float  # Embedding similarity
    fact_overlap: float  # Jaccard similarity of facts
    error_rate: float  # % of errors


@dataclass
class ABTestResult:
    """A/B test comparison result"""
    conversation_id: str
    local_result: ExtractionResult
    claude_result: ExtractionResult
    local_metrics: QualityMetrics
    claude_metrics: QualityMetrics
    winner: str  # "local", "claude", or "tie"
    quality_delta: float  # local - claude


@dataclass
class CorpusResults:
    """Results across entire test corpus"""
    local_quality: float
    claude_quality: float
    cost_reduction: float
    local_avg_metrics: QualityMetrics
    claude_avg_metrics: QualityMetrics
    comparisons: List[ABTestResult]
    meets_threshold: bool
    failures: List[str]


class TestCorpus:
    """
    Manages test corpus of conversations with ground truth.
    """
    
    def __init__(self, corpus_path: Optional[str] = None):
        self.conversations: List[TestConversation] = []
        self.corpus_path = corpus_path
        if corpus_path and os.path.exists(corpus_path):
            self.load(corpus_path)
    
    def load(self, path: str):
        """Load corpus from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        for conv_data in data.get("conversations", []):
            gt_data = conv_data.get("ground_truth", {})
            ground_truth = GroundTruth(
                facts=gt_data.get("facts", []),
                decisions=gt_data.get("decisions", []),
                next_steps=gt_data.get("next_steps", []),
                entities=gt_data.get("entities", {}),
                summary=gt_data.get("summary")
            )
            
            self.conversations.append(TestConversation(
                id=conv_data["id"],
                domain=conv_data.get("domain", "general"),
                difficulty=conv_data.get("difficulty", "medium"),
                turns=conv_data["turns"],
                ground_truth=ground_truth,
                metadata=conv_data.get("metadata", {})
            ))
    
    def save(self, path: str):
        """Save corpus to JSON file"""
        data = {
            "conversations": [
                {
                    "id": conv.id,
                    "domain": conv.domain,
                    "difficulty": conv.difficulty,
                    "turns": conv.turns,
                    "ground_truth": {
                        "facts": conv.ground_truth.facts,
                        "decisions": conv.ground_truth.decisions,
                        "next_steps": conv.ground_truth.next_steps,
                        "entities": conv.ground_truth.entities,
                        "summary": conv.ground_truth.summary
                    },
                    "metadata": conv.metadata
                }
                for conv in self.conversations
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_conversation(self, conversation: TestConversation):
        """Add a conversation to the corpus"""
        self.conversations.append(conversation)
    
    def get_by_domain(self, domain: str) -> List[TestConversation]:
        """Get conversations by domain"""
        return [c for c in self.conversations if c.domain == domain]
    
    def get_by_difficulty(self, difficulty: str) -> List[TestConversation]:
        """Get conversations by difficulty"""
        return [c for c in self.conversations if c.difficulty == difficulty]
    
    def sample(self, n: int, seed: Optional[int] = None) -> List[TestConversation]:
        """Sample n conversations randomly"""
        import random
        if seed:
            random.seed(seed)
        return random.sample(self.conversations, min(n, len(self.conversations)))
    
    def __len__(self):
        return len(self.conversations)


class QualityAnalyzer:
    """
    Analyzes extraction quality against ground truth.
    """
    
    def __init__(self, ollama_endpoint: str = "http://192.168.85.158:11434"):
        self.ollama_endpoint = ollama_endpoint
    
    def analyze(self, 
                extraction: ExtractionResult, 
                ground_truth: GroundTruth) -> QualityMetrics:
        """
        Calculate quality metrics for an extraction.
        """
        # Calculate precision and recall for facts
        extracted_facts = set(extraction.facts)
        true_facts = set(ground_truth.facts)
        
        tp = len(extracted_facts & true_facts)  # True positives
        fp = len(extracted_facts - true_facts)  # False positives
        fn = len(true_facts - extracted_facts)  # False negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Coverage: what % of ground truth was extracted
        total_ground_truth = (
            len(ground_truth.facts) + 
            len(ground_truth.decisions) + 
            len(ground_truth.next_steps)
        )
        total_extracted = (
            len(extraction.facts) + 
            len(extraction.decisions) + 
            len(extraction.next_steps)
        )
        coverage = total_extracted / total_ground_truth if total_ground_truth > 0 else 0.0
        
        # Accuracy: % of extracted that are correct
        accuracy = tp / len(extracted_facts) if len(extracted_facts) > 0 else 0.0
        
        # Semantic similarity (simplified - uses string similarity)
        semantic_sim = self._semantic_similarity(
            extraction.summary or "",
            ground_truth.summary or ""
        )
        
        # Fact overlap (Jaccard similarity)
        union = extracted_facts | true_facts
        fact_overlap = len(extracted_facts & true_facts) / len(union) if union else 0.0
        
        # Error rate
        error_rate = fp / len(extracted_facts) if len(extracted_facts) > 0 else 0.0
        
        return QualityMetrics(
            coverage=coverage,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            semantic_similarity=semantic_sim,
            fact_overlap=fact_overlap,
            error_rate=error_rate
        )
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        # Simple implementation using sequence matcher
        # In production, use embeddings
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    async def analyze_batch(self, 
                           extractions: List[ExtractionResult],
                           ground_truths: List[GroundTruth]) -> List[QualityMetrics]:
        """Analyze batch of extractions"""
        return [self analyze(e, g) for e, g in zip(extractions, ground_truths)]


class ABTestFramework:
    """
    A/B testing framework comparing local vs Claude extraction.
    """
    
    def __init__(self,
                 claude_api_key: Optional[str] = None,
                 ollama_endpoint: str = "http://192.168.85.158:11434"):
        self.claude_api_key = claude_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.ollama_endpoint = ollama_endpoint
        self.analyzer = QualityAnalyzer(ollama_endpoint)
    
    async def extract_with_claude(self, conversation: TestConversation) -> ExtractionResult:
        """Extract using Claude API"""
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=self.claude_api_key)
            
            # Format conversation
            conversation_text = "\n\n".join([
                f"[{t['role']}]: {t['content']}" 
                for t in conversation.turns
            ])
            
            prompt = f"""Extract all facts, decisions, and next steps from this conversation.

Conversation:
{conversation_text}

Return JSON format:
{{
  "facts": ["fact1", "fact2", ...],
  "decisions": ["decision1", ...],
  "next_steps": ["step1", ...],
  "entities": {{"person": [], "location": [], "organization": []}},
  "summary": "brief summary"
}}

Extraction:"""
            
            start_time = datetime.now()
            response = await client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Parse result
            raw_output = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            cost = tokens_used * 0.000003  # Approximate cost
            
            try:
                data = json.loads(raw_output)
                return ExtractionResult(
                    facts=data.get("facts", []),
                    decisions=data.get("decisions", []),
                    next_steps=data.get("next_steps", []),
                    entities=data.get("entities", {}),
                    summary=data.get("summary"),
                    raw_output=raw_output,
                    tokens_used=tokens_used,
                    cost=cost,
                    processing_time_ms=processing_time
                )
            except json.JSONDecodeError:
                return ExtractionResult(
                    facts=[],
                    decisions=[],
                    next_steps=[],
                    entities={},
                    raw_output=raw_output,
                    tokens_used=tokens_used,
                    cost=cost,
                    processing_time_ms=processing_time
                )
                
        except Exception as e:
            return ExtractionResult(
                facts=[],
                decisions=[],
                next_steps=[],
                entities={},
                raw_output=str(e),
                cost=0.0,
                processing_time_ms=0.0
            )
    
    async def extract_with_local(self, conversation: TestConversation) -> ExtractionResult:
        """Extract using local 7B model"""
        try:
            import aiohttp
            
            conversation_text = "\n\n".join([
                f"[{t['role']}]: {t['content']}" 
                for t in conversation.turns
            ])
            
            prompt = f"""Extract all facts, decisions, and next steps from this conversation.

Conversation:
{conversation_text}

Return JSON format:
{{
  "facts": ["fact1", "fact2", ...],
  "decisions": ["decision1", ...],
  "next_steps": ["step1", ...],
  "entities": {{"person": [], "location": [], "organization": []}},
  "summary": "brief summary"
}}

Extraction:"""
            
            start_time = datetime.now()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_endpoint}/api/generate",
                    json={
                        "model": "qwen2.5-coder:7b",
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3}
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    result = await response.json()
                    raw_output = result.get("response", "")
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Estimate tokens (approximate)
            tokens_used = len(prompt + raw_output) // 4
            cost = 0.0  # Local model = free
            
            try:
                data = json.loads(raw_output)
                return ExtractionResult(
                    facts=data.get("facts", []),
                    decisions=data.get("decisions", []),
                    next_steps=data.get("next_steps", []),
                    entities=data.get("entities", {}),
                    summary=data.get("summary"),
                    raw_output=raw_output,
                    tokens_used=tokens_used,
                    cost=cost,
                    processing_time_ms=processing_time
                )
            except json.JSONDecodeError:
                return ExtractionResult(
                    facts=[],
                    decisions=[],
                    next_steps=[],
                    entities={},
                    raw_output=raw_output,
                    tokens_used=tokens_used,
                    cost=cost,
                    processing_time_ms=processing_time
                )
                
        except Exception as e:
            return ExtractionResult(
                facts=[],
                decisions=[],
                next_steps=[],
                entities={},
                raw_output=str(e),
                cost=0.0,
                processing_time_ms=0.0
            )
    
    async def compare_single(self, conversation: TestConversation) -> ABTestResult:
        """Compare local vs Claude on a single conversation"""
        # Run both extractions in parallel
        local_result, claude_result = await asyncio.gather(
            self.extract_with_local(conversation),
            self.extract_with_claude(conversation)
        )
        
        # Analyze quality
        local_metrics = self.analyzer.analyze(local_result, conversation.ground_truth)
        claude_metrics = self.analyzer.analyze(claude_result, conversation.ground_truth)
        
        # Determine winner based on F1 score
        local_score = local_metrics.f1_score
        claude_score = claude_metrics.f1_score
        
        if local_score > claude_score + 0.05:
            winner = "local"
        elif claude_score > local_score + 0.05:
            winner = "claude"
        else:
            winner = "tie"
        
        quality_delta = local_score - claude_score
        
        return ABTestResult(
            conversation_id=conversation.id,
            local_result=local_result,
            claude_result=claude_result,
            local_metrics=local_metrics,
            claude_metrics=claude_metrics,
            winner=winner,
            quality_delta=quality_delta
        )
    
    async def compare_on_corpus(self, 
                                corpus: TestCorpus,
                                sample_size: Optional[int] = None,
                                progress_callback: Optional[callable] = None) -> CorpusResults:
        """
        Run A/B comparison on entire test corpus.
        """
        conversations = corpus.conversations
        if sample_size:
            conversations = corpus.sample(sample_size)
        
        comparisons = []
        for i, conv in enumerate(conversations):
            comparison = await self.compare_single(conv)
            comparisons.append(comparison)
            
            if progress_callback:
                progress_callback(i + 1, len(conversations))
        
        # Calculate aggregate metrics
        local_qualities = [c.local_metrics.f1_score for c in comparisons]
        claude_qualities = [c.claude_metrics.f1_score for c in comparisons]
        
        local_quality = np.mean(local_qualities)
        claude_quality = np.mean(claude_qualities)
        
        # Calculate cost reduction
        total_local_cost = sum(c.local_result.cost for c in comparisons)
        total_claude_cost = sum(c.claude_result.cost for c in comparisons)
        cost_reduction = (total_claude_cost - total_local_cost) / total_claude_cost if total_claude_cost > 0 else 0.0
        
        # Check thresholds
        quality_threshold = 0.90
        coverage_threshold = 0.90
        
        local_coverage = np.mean([c.local_metrics.coverage for c in comparisons])
        
        meets_threshold = (
            local_quality >= quality_threshold * claude_quality and
            local_coverage >= coverage_threshold
        )
        
        # Find failures
        failures = [
            c.conversation_id for c in comparisons 
            if c.local_metrics.f1_score < 0.7
        ]
        
        # Average metrics
        local_avg = QualityMetrics(
            coverage=np.mean([c.local_metrics.coverage for c in comparisons]),
            accuracy=np.mean([c.local_metrics.accuracy for c in comparisons]),
            precision=np.mean([c.local_metrics.precision for c in comparisons]),
            recall=np.mean([c.local_metrics.recall for c in comparisons]),
            f1_score=local_quality,
            semantic_similarity=np.mean([c.local_metrics.semantic_similarity for c in comparisons]),
            fact_overlap=np.mean([c.local_metrics.fact_overlap for c in comparisons]),
            error_rate=np.mean([c.local_metrics.error_rate for c in comparisons])
        )
        
        claude_avg = QualityMetrics(
            coverage=np.mean([c.claude_metrics.coverage for c in comparisons]),
            accuracy=np.mean([c.claude_metrics.accuracy for c in comparisons]),
            precision=np.mean([c.claude_metrics.precision for c in comparisons]),
            recall=np.mean([c.claude_metrics.recall for c in comparisons]),
            f1_score=claude_quality,
            semantic_similarity=np.mean([c.claude_metrics.semantic_similarity for c in comparisons]),
            fact_overlap=np.mean([c.claude_metrics.fact_overlap for c in comparisons]),
            error_rate=np.mean([c.claude_metrics.error_rate for c in comparisons])
        )
        
        return CorpusResults(
            local_quality=local_quality,
            claude_quality=claude_quality,
            cost_reduction=cost_reduction,
            local_avg_metrics=local_avg,
            claude_avg_metrics=claude_avg,
            comparisons=comparisons,
            meets_threshold=meets_threshold,
            failures=failures
        )


class RegressionDetector:
    """
    Detects quality regressions in extraction pipeline.
    """
    
    def __init__(self,
                 quality_threshold: float = 0.90,
                 coverage_threshold: float = 0.90,
                 comparison_baseline: Optional[CorpusResults] = None):
        self.quality_threshold = quality_threshold
        self.coverage_threshold = coverage_threshold
        self.baseline = comparison_baseline
        self.history: List[CorpusResults] = []
    
    def check_regression(self, results: CorpusResults) -> Tuple[bool, List[str]]:
        """
        Check if results show regression.
        Returns (is_regression, reasons).
        """
        regressions = []
        
        # Check quality threshold
        if results.local_quality < self.quality_threshold:
            regressions.append(
                f"Quality below threshold: {results.local_quality:.2%} < {self.quality_threshold:.2%}"
            )
        
        # Check coverage threshold
        if results.local_avg_metrics.coverage < self.coverage_threshold:
            regressions.append(
                f"Coverage below threshold: {results.local_avg_metrics.coverage:.2%} < {self.coverage_threshold:.2%}"
            )
        
        # Compare to baseline if available
        if self.baseline:
            quality_drop = self.baseline.local_quality - results.local_quality
            if quality_drop > 0.05:  # 5% drop threshold
                regressions.append(
                    f"Quality drop from baseline: -{quality_drop:.2%}"
                )
        
        # Compare to historical average
        if self.history:
            hist_quality = np.mean([h.local_quality for h in self.history])
            if results.local_quality < hist_quality - 0.03:
                regressions.append(
                    f"Quality below historical average: {results.local_quality:.2%} < {hist_quality:.2%}"
                )
        
        is_regression = len(regressions) > 0
        return is_regression, regressions
    
    def add_result(self, results: CorpusResults):
        """Add result to history"""
        self.history.append(results)
        # Keep last 10 results
        if len(self.history) > 10:
            self.history = self.history[-10:]


class MonitoringDashboard:
    """
    Simple monitoring dashboard for tracking extraction quality over time.
    """
    
    def __init__(self, metrics_file: str = "metrics_history.json"):
        self.metrics_file = metrics_file
        self.metrics_history: List[Dict] = []
        self.load_history()
    
    def load_history(self):
        """Load metrics history from file"""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                self.metrics_history = json.load(f)
    
    def save_history(self):
        """Save metrics history to file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def record_run(self, results: CorpusResults, metadata: Optional[Dict] = None):
        """Record a test run"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "local_quality": results.local_quality,
            "claude_quality": results.claude_quality,
            "cost_reduction": results.cost_reduction,
            "meets_threshold": results.meets_threshold,
            "num_tests": len(results.comparisons),
            "local_metrics": {
                "coverage": results.local_avg_metrics.coverage,
                "accuracy": results.local_avg_metrics.accuracy,
                "precision": results.local_avg_metrics.precision,
                "recall": results.local_avg_metrics.recall,
                "f1_score": results.local_avg_metrics.f1_score
            },
            "metadata": metadata or {}
        }
        
        self.metrics_history.append(entry)
        self.save_history()
    
    def get_trends(self) -> Dict:
        """Get quality trends over time"""
        if len(self.metrics_history) < 2:
            return {"status": "insufficient_data"}
        
        qualities = [m["local_quality"] for m in self.metrics_history]
        
        return {
            "quality_trend": "improving" if qualities[-1] > qualities[0] else "declining",
            "quality_change": qualities[-1] - qualities[0],
            "avg_quality": np.mean(qualities),
            "min_quality": min(qualities),
            "max_quality": max(qualities),
            "num_runs": len(self.metrics_history)
        }
    
    def generate_report(self) -> str:
        """Generate a text report of current status"""
        if not self.metrics_history:
            return "No metrics history available."
        
        latest = self.metrics_history[-1]
        trends = self.get_trends()
        
        report = f"""
=== Memory Extraction Quality Report ===
Generated: {datetime.now().isoformat()}

Latest Run ({latest['timestamp']}):
- Local Quality: {latest['local_quality']:.2%}
- Claude Quality: {latest['claude_quality']:.2%}
- Cost Reduction: {latest['cost_reduction']:.1%}
- Meets Threshold: {'✓' if latest['meets_threshold'] else '✗'}
- Tests Run: {latest['num_tests']}

Metrics:
- Coverage: {latest['local_metrics']['coverage']:.2%}
- Accuracy: {latest['local_metrics']['accuracy']:.2%}
- Precision: {latest['local_metrics']['precision']:.2%}
- Recall: {latest['local_metrics']['recall']:.2%}
- F1 Score: {latest['local_metrics']['f1_score']:.2%}

Trends ({trends['num_runs']} runs):
- Status: {trends['quality_trend']}
- Quality Change: {trends['quality_change']:+.2%}
- Avg Quality: {trends['avg_quality']:.2%}
"""
        return report


# Convenience functions
async def run_quality_check(corpus_path: str, 
                            sample_size: int = 100,
                            claude_api_key: Optional[str] = None) -> CorpusResults:
    """Run quality check on test corpus"""
    corpus = TestCorpus(corpus_path)
    ab_test = ABTestFramework(claude_api_key=claude_api_key)
    
    results = await ab_test.compare_on_corpus(corpus, sample_size)
    
    # Record to dashboard
    dashboard = MonitoringDashboard()
    dashboard.record_run(results)
    
    return results


def check_for_regression(results: CorpusResults, 
                        threshold: float = 0.90) -> bool:
    """Quick check for quality regression"""
    detector = RegressionDetector(quality_threshold=threshold)
    is_regression, reasons = detector.check_regression(results)
    
    if is_regression:
        print("⚠️ REGRESSION DETECTED:")
        for reason in reasons:
            print(f"  - {reason}")
    else:
        print("✅ Quality meets threshold")
    
    return is_regression
