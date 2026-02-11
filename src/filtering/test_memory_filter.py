import unittest
from datetime import datetime, timedelta
from memory_filter import (
    DuplicateDetector, 
    ChangeDetector, 
    PriorityScorer,
    BatchProcessor,
    SmartFilter,
    FilterResult
)


class TestDuplicateDetector(unittest.TestCase):
    
    def test_jaccard_similarity_identical(self):
        detector = DuplicateDetector()
        sim = detector._jaccard_similarity("hello world", "hello world")
        self.assertEqual(sim, 1.0)
    
    def test_jaccard_similarity_different(self):
        detector = DuplicateDetector()
        sim = detector._jaccard_similarity("hello world", "foo bar baz")
        self.assertEqual(sim, 0.0)
    
    def test_jaccard_similarity_partial(self):
        detector = DuplicateDetector()
        sim = detector._jaccard_similarity("hello world foo", "hello world bar")
        self.assertGreater(sim, 0.5)
        self.assertLess(sim, 1.0)
    
    def test_is_duplicate_exact_match(self):
        # This would need mocked API, so we test the logic
        detector = DuplicateDetector(similarity_threshold=0.9)
        # Test internal similarity logic
        sim = detector._jaccard_similarity(
            "The user wants to deploy on AWS",
            "The user wants to deploy on AWS"
        )
        self.assertEqual(sim, 1.0)


class TestChangeDetector(unittest.TestCase):
    
    def test_too_small_no_changes(self):
        detector = ChangeDetector(min_tokens=500)
        turns = [{"tokens": 100, "content": "small"}]
        has_changes, reason = detector.has_significant_changes(turns, "test")
        self.assertFalse(has_changes)
        self.assertIn("small", reason.lower())
    
    def test_large_enough_has_changes(self):
        detector = ChangeDetector(min_tokens=100)
        turns = [{"tokens": 500, "content": "This is a much larger conversation with many words that should trigger change detection because it exceeds the minimum token threshold"}]
        has_changes, reason = detector.has_significant_changes(turns, "test")
        self.assertTrue(has_changes)
    
    def test_no_state_change(self):
        detector = ChangeDetector()
        turns = [{"tokens": 1000, "content": "test content"}]
        
        # First call should detect change
        has_changes1, _ = detector.has_significant_changes(turns, "test")
        self.assertTrue(has_changes1)
        
        # Second call with same content should not
        has_changes2, reason = detector.has_significant_changes(turns, "test")
        self.assertFalse(has_changes2)
        self.assertIn("no state change", reason.lower())


class TestPriorityScorer(unittest.TestCase):
    
    def test_error_priority(self):
        scorer = PriorityScorer()
        score, category, reason = scorer.score("There was an error in the deployment")
        self.assertEqual(score, 1.0)
        self.assertEqual(category, "error")
    
    def test_decision_priority(self):
        scorer = PriorityScorer()
        score, category, reason = scorer.score("We decided to use PostgreSQL")
        self.assertEqual(score, 0.9)
        self.assertEqual(category, "decision")
    
    def test_next_step_priority(self):
        scorer = PriorityScorer()
        score, category, reason = scorer.score("Next action is to fix the bug")
        self.assertEqual(score, 0.8)
        self.assertEqual(category, "next_step")
    
    def test_general_priority(self):
        scorer = PriorityScorer()
        score, category, reason = scorer.score("Just a general chat message")
        self.assertEqual(score, 0.5)
        self.assertEqual(category, "general")
    
    def test_immediate_extraction_threshold(self):
        scorer = PriorityScorer()
        self.assertTrue(scorer.should_extract_immediately(0.8))
        self.assertTrue(scorer.should_extract_immediately(0.9))
        self.assertFalse(scorer.should_extract_immediately(0.5))
        self.assertFalse(scorer.should_extract_immediately(0.6))


class TestBatchProcessor(unittest.TestCase):
    
    def test_add_to_batch(self):
        processor = BatchProcessor(batch_interval_minutes=60)
        should_process = processor.add({"test": "item"})
        self.assertFalse(should_process)  # Not enough items yet
        self.assertEqual(len(processor.batched_items), 1)
    
    def test_batch_size_threshold(self):
        processor = BatchProcessor(batch_interval_minutes=60, max_batch_size=3)
        
        # Add 2 items - shouldn't trigger
        processor.add({"test": "item1"})
        processor.add({"test": "item2"})
        self.assertFalse(processor.should_process())
        
        # Add 3rd item - should trigger
        should_process = processor.add({"test": "item3"})
        self.assertTrue(should_process)
    
    def test_get_batch_clears_items(self):
        processor = BatchProcessor()
        processor.add({"test": "item"})
        
        batch = processor.get_batch()
        self.assertEqual(len(batch), 1)
        self.assertEqual(len(processor.batched_items), 0)


class TestSmartFilter(unittest.TestCase):
    
    def test_skip_small_conversation(self):
        filter = SmartFilter()
        turns = [{"tokens": 100, "content": "small"}]
        result = filter.should_extract(turns, "test")
        
        self.assertFalse(result.should_extract)
        self.assertEqual(result.priority, 0.0)
    
    def test_extract_high_priority(self):
        filter = SmartFilter(min_tokens=10)  # Lower threshold for testing
        turns = [{"tokens": 1000, "content": "Critical error in production system"}]
        result = filter.should_extract(turns, "test")
        
        self.assertTrue(result.should_extract)
        self.assertGreaterEqual(result.priority, 0.7)
        self.assertFalse(result.batched)  # High priority = immediate
    
    def test_batch_low_priority(self):
        filter = SmartFilter(min_tokens=10)
        turns = [{"tokens": 1000, "content": "Just some general conversation about nothing important"}]
        result = filter.should_extract(turns, "test")
        
        self.assertTrue(result.should_extract)
        self.assertLess(result.priority, 0.7)
        self.assertTrue(result.batched)  # Low priority = batch


class TestIntegration(unittest.TestCase):
    """Integration tests with mocked dependencies"""
    
    def test_full_filtering_workflow(self):
        """Test the complete filtering workflow"""
        filter = SmartFilter(min_tokens=50)
        
        # Simulate conversation turns
        turns = [
            {"tokens": 300, "content": "User asked about deployment"},
            {"tokens": 500, "content": "We decided to use Kubernetes for orchestration"},
            {"tokens": 200, "content": "Next step is to set up the cluster"}
        ]
        
        result = filter.should_extract(turns, "test-agent")
        
        # Should extract (significant content + decision)
        self.assertTrue(result.should_extract)
        self.assertGreaterEqual(result.priority, 0.8)
        
        # Should have context hash
        self.assertIsNotNone(result.context_hash)
        self.assertGreater(len(result.context_hash), 0)


if __name__ == "__main__":
    unittest.main()
