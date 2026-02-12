"""
Test Corpus Generator
Creates synthetic test conversations with ground truth for validation.
"""

import json
import random
from typing import List, Dict, Any
from datetime import datetime, timedelta

from testing_framework import TestConversation, GroundTruth


class TestCorpusGenerator:
    """Generate synthetic test conversations with ground truth"""
    
    DOMAINS = ["software-engineering", "devops", "data-science", "general", "customer-support"]
    DIFFICULTIES = ["easy", "medium", "hard"]
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
    
    def generate_corpus(self, n: int = 100) -> List[TestConversation]:
        """Generate n test conversations"""
        conversations = []
        
        for i in range(n):
            domain = random.choice(self.DOMAINS)
            difficulty = random.choice(self.DIFFICULTIES)
            
            conv = self._generate_conversation(i, domain, difficulty)
            conversations.append(conv)
        
        return conversations
    
    def _generate_conversation(self, idx: int, domain: str, difficulty: str) -> TestConversation:
        """Generate a single conversation with ground truth"""
        
        if domain == "software-engineering":
            return self._generate_software_conv(idx, difficulty)
        elif domain == "devops":
            return self._generate_devops_conv(idx, difficulty)
        elif domain == "data-science":
            return self._generate_data_science_conv(idx, difficulty)
        else:
            return self._generate_general_conv(idx, difficulty)
    
    def _generate_software_conv(self, idx: int, difficulty: str) -> TestConversation:
        """Generate software engineering conversation"""
        
        topics = [
            {
                "problem": "Database connection pooling issue",
                "solution": "Implemented connection pool with max 20 connections",
                "facts": [
                    "Database connection timeout was 30 seconds",
                    "Connection pool size set to 20 max connections",
                    "Using PostgreSQL 14 with pgbouncer"
                ],
                "decisions": [
                    "Use connection pooling instead of direct connections",
                    "Set max connections to 20 to prevent overload"
                ],
                "next_steps": [
                    "Monitor connection pool metrics",
                    "Add alerting for connection exhaustion"
                ]
            },
            {
                "problem": "API rate limiting needed",
                "solution": "Implemented token bucket algorithm",
                "facts": [
                    "API receiving 1000 requests per minute",
                    "Token bucket allows 100 requests per user per minute",
                    "Burst capacity of 150 requests"
                ],
                "decisions": [
                    "Use token bucket vs sliding window",
                    "Set rate limit to 100 req/min per user"
                ],
                "next_steps": [
                    "Deploy to staging",
                    "Update API documentation"
                ]
            }
        ]
        
        topic = random.choice(topics)
        
        turns = [
            {"role": "user", "content": f"We're having issues with {topic['problem']}. What's the best approach?"},
            {"role": "assistant", "content": f"I recommend {topic['solution']}. Here's why this approach works..."},
            {"role": "user", "content": "That makes sense. What are the next steps?"},
            {"role": "assistant", "content": f"Here are the next steps: {', '.join(topic['next_steps'][:2])}."}
        ]
        
        ground_truth = GroundTruth(
            facts=topic["facts"],
            decisions=topic["decisions"],
            next_steps=topic["next_steps"],
            entities={
                "technology": ["PostgreSQL", "pgbouncer", "Token bucket"],
                "metric": ["30 seconds", "20 connections", "1000 requests", "100 req/min"]
            },
            summary=f"Discussed {topic['problem']} and decided on {topic['solution']}"
        )
        
        return TestConversation(
            id=f"sw-test-{idx:04d}",
            domain="software-engineering",
            difficulty=difficulty,
            turns=turns,
            ground_truth=ground_truth,
            metadata={"topic": topic["problem"]}
        )
    
    def _generate_devops_conv(self, idx: int, difficulty: str) -> TestConversation:
        """Generate DevOps conversation"""
        
        scenarios = [
            {
                "issue": "Kubernetes pod scaling",
                "facts": [
                    "Current cluster has 3 nodes",
                    "HPA configured with 70% CPU threshold",
                    "Pods scale from 2 to 10 replicas"
                ],
                "decisions": [
                    "Enable cluster autoscaling",
                    "Set resource requests and limits"
                ],
                "next_steps": [
                    "Configure HPA metrics",
                    "Test load scenarios"
                ]
            }
        ]
        
        scenario = random.choice(scenarios)
        
        turns = [
            {"role": "user", "content": f"How do we handle {scenario['issue']}?"},
            {"role": "assistant", "content": "Let me analyze the current setup and propose a solution..."},
            {"role": "user", "content": "Sounds good. What should we configure?"},
            {"role": "assistant", "content": f"Here's what I recommend: {', '.join(scenario['decisions'])}"}
        ]
        
        ground_truth = GroundTruth(
            facts=scenario["facts"],
            decisions=scenario["decisions"],
            next_steps=scenario["next_steps"],
            entities={"tool": ["Kubernetes", "HPA"], "metric": ["70%", "2-10 replicas"]},
            summary=f"Configured {scenario['issue']}"
        )
        
        return TestConversation(
            id=f"devops-test-{idx:04d}",
            domain="devops",
            difficulty=difficulty,
            turns=turns,
            ground_truth=ground_truth
        )
    
    def _generate_data_science_conv(self, idx: int, difficulty: str) -> TestConversation:
        """Generate data science conversation"""
        
        turns = [
            {"role": "user", "content": "Our model accuracy dropped from 95% to 87%. What could be causing this?"},
            {"role": "assistant", "content": "Several factors could cause this: data drift, concept drift, or changes in feature distribution..."},
            {"role": "user", "content": "How do we identify which one it is?"},
            {"role": "assistant", "content": "I recommend checking feature statistics and running KS tests for drift detection."}
        ]
        
        ground_truth = GroundTruth(
            facts=[
                "Model accuracy dropped from 95% to 87%",
                "Possible causes: data drift, concept drift, feature distribution changes",
                "KS tests can detect distribution changes"
            ],
            decisions=["Run drift detection analysis", "Compare feature statistics"],
            next_steps=["Implement monitoring pipeline", "Set up alerting for drift"],
            entities={"metric": ["95%", "87%"], "technique": ["KS test", "drift detection"]},
            summary="Diagnosing model accuracy degradation and planning drift detection"
        )
        
        return TestConversation(
            id=f"ds-test-{idx:04d}",
            domain="data-science",
            difficulty=difficulty,
            turns=turns,
            ground_truth=ground_truth
        )
    
    def _generate_general_conv(self, idx: int, difficulty: str) -> TestConversation:
        """Generate general conversation"""
        
        turns = [
            {"role": "user", "content": "What's the status of the project?"},
            {"role": "assistant", "content": "The project is 80% complete. We finished the backend API and are working on the frontend..."},
            {"role": "user", "content": "Any blockers?"},
            {"role": "assistant", "content": "No major blockers. The team is waiting for design approval for the dashboard."}
        ]
        
        ground_truth = GroundTruth(
            facts=[
                "Project is 80% complete",
                "Backend API is finished",
                "Frontend in progress",
                "Waiting for design approval"
            ],
            decisions=["Proceed with frontend development"],
            next_steps=["Get design approval", "Complete frontend", "Integration testing"],
            entities={"status": ["80% complete"], "component": ["backend API", "frontend"]},
            summary="Project status update: 80% complete, waiting on design approval"
        )
        
        return TestConversation(
            id=f"gen-test-{idx:04d}",
            domain="general",
            difficulty=difficulty,
            turns=turns,
            ground_truth=ground_truth
        )
    
    def save_corpus(self, conversations: List[TestConversation], path: str):
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
                for conv in conversations
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(conversations)} test conversations to {path}")


# Generate and save test corpus
if __name__ == "__main__":
    generator = TestCorpusGenerator(seed=42)
    corpus = generator.generate_corpus(n=100)
    generator.save_corpus(corpus, "test_corpus.json")
    
    # Print sample
    print("\n=== Sample Conversation ===")
    sample = corpus[0]
    print(f"ID: {sample.id}")
    print(f"Domain: {sample.domain}")
    print(f"Difficulty: {sample.difficulty}")
    print(f"Facts: {len(sample.ground_truth.facts)}")
    print(f"Decisions: {len(sample.ground_truth.decisions)}")
    print(f"Next Steps: {len(sample.ground_truth.next_steps)}")
