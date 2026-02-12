# Testing & Validation Framework

Comprehensive testing and quality validation for memory extraction pipeline.

## Overview

This framework ensures local model extraction quality matches or exceeds Claude baseline while achieving 85-90% cost reduction.

## Components

- **TestCorpus**: 100+ conversations with ground truth extractions
- **QualityMetrics**: Coverage, accuracy, and semantic similarity scoring
- **ABTestFramework**: Parallel Claude vs Local extraction comparison
- **MonitoringDashboard**: Real-time success rates and quality tracking
- **RegressionDetector**: Automated quality threshold monitoring

## Quality Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Coverage** | % of facts extracted vs ground truth | ≥90% |
| **Accuracy** | % of extracted facts that are correct | ≥95% |
| **Semantic Similarity** | Embedding similarity to ground truth | ≥0.85 |
| **Fact Precision** | Precision of fact extraction | ≥0.90 |
| **Fact Recall** | Recall of fact extraction | ≥0.90 |
| **F1 Score** | Harmonic mean of precision/recall | ≥0.90 |

## A/B Testing

```python
from testing_framework import ABTestFramework

ab_test = ABTestFramework(
    claude_api_key="sk-...",
    ollama_endpoint="http://192.168.85.158:11434"
)

# Run comparison on test corpus
results = await ab_test.compare_on_corpus(
    corpus_path="test_corpus.json",
    sample_size=100
)

print(f"Local Quality: {results.local_quality:.2%}")
print(f"Claude Quality: {results.claude_quality:.2%}")
print(f"Cost Reduction: {results.cost_reduction:.2%}")
print(f"Acceptable: {results.meets_threshold}")
```

## Regression Detection

```python
from testing_framework import RegressionDetector

detector = RegressionDetector(
    quality_threshold=0.90,
    coverage_threshold=0.90
)

# On each code change
if detector.check_regression(new_results):
    raise Exception("Quality regression detected!")
```

## Test Corpus Format

```json
{
  "conversations": [
    {
      "id": "test-001",
      "domain": "software-engineering",
      "difficulty": "medium",
      "turns": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "ground_truth": {
        "facts": ["fact1", "fact2"],
        "decisions": ["decision1"],
        "next_steps": ["step1"],
        "entities": {"person": [...], "location": [...]}
      }
    }
  ]
}
```
