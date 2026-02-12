import unittest
from unittest.mock import Mock, patch, MagicMock
from context_compaction import (
    ContextCompactor,
    Level1Summarizer,
    Level2Summarizer,
    Level3Summarizer,
    RAGPreRanker,
    TokenCounter,
    CompactionLevel,
    TurnSummary
)


class TestTokenCounter(unittest.TestCase):
    
    def test_count_text(self):
        """Test token counting for prose"""
        text = "This is a simple sentence."
        # ~4 chars per token for text
        expected = len(text) // 4
        self.assertEqual(TokenCounter.count(text), expected)
    
    def test_count_code(self):
        """Test token counting for code"""
        code = "def main():\n    x = {1, 2, 3}\n    return x"
        # Code has more tokens per char
        result = TokenCounter.count(code)
        self.assertGreater(result, 0)
    
    def test_count_empty(self):
        """Empty string returns 0"""
        self.assertEqual(TokenCounter.count(""), 0)


class TestLevel1Summarizer(unittest.TestCase):
    
    def setUp(self):
        self.summarizer = Level1Summarizer()
    
    @patch('context_compaction.requests.post')
    def test_summarize_small_turn_no_change(self, mock_post):
        """Small turns should not be summarized"""
        content = "Hello"
        result = self.summarizer.summarize_turn("user", content, 0)
        
        self.assertEqual(result.summary, content)
        self.assertEqual(result.original_tokens, result.summary_tokens)
        mock_post.assert_not_called()
    
    @patch('context_compaction.requests.post')
    def test_summarize_large_turn(self, mock_post):
        """Large turns should be summarized"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Summary of user message"}
        mock_post.return_value = mock_response
        
        # Large content that needs summarization
        content = "This is a very long message. " * 100
        result = self.summarizer.summarize_turn("user", content, 0)
        
        self.assertNotEqual(result.summary, content)
        mock_post.assert_called_once()
    
    @patch('context_compaction.requests.post')
    def test_fallback_on_error(self, mock_post):
        """Should fallback to truncation on error"""
        mock_post.side_effect = Exception("Connection failed")
        
        content = "This is a very long message that needs summarization. " * 100
        result = self.summarizer.summarize_turn("user", content, 0)
        
        self.assertIn("[truncated]", result.summary)
    
    def test_summarize_turns_batch(self):
        """Test batch processing of turns"""
        turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        results = self.summarizer.summarize_turns(turns)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].role, "user")
        self.assertEqual(results[1].role, "assistant")


class TestLevel2Summarizer(unittest.TestCase):
    
    def setUp(self):
        self.summarizer = Level2Summarizer()
    
    def test_small_batch_no_change(self):
        """Small batches should not be resummarized"""
        summaries = [
            TurnSummary(0, "user", "content", "short", 50, 50),
            TurnSummary(1, "assistant", "content", "also short", 50, 50)
        ]
        
        result = self.summarizer.summarize_batch(summaries, 0)
        
        self.assertIn("short", result)
        self.assertIn("also short", result)
    
    @patch('context_compaction.requests.post')
    def test_large_batch_summarized(self, mock_post):
        """Large batches should be synthesized"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Synthesized conversation"}
        mock_post.return_value = mock_response
        
        summaries = [
            TurnSummary(i, "user" if i % 2 == 0 else "assistant", 
                       "content", "Summary " * 50, 200, 200)
            for i in range(5)
        ]
        
        result = self.summarizer.summarize_batch(summaries, 0)
        
        self.assertEqual(result, "Synthesized conversation")
        mock_post.assert_called_once()
    
    def test_summarize_all_single_batch(self):
        """Test processing all turns as single batch"""
        summaries = [
            TurnSummary(0, "user", "content", "First message", 50, 50),
            TurnSummary(1, "assistant", "content", "Response", 50, 50)
        ]
        
        result = self.summarizer.summarize_all(summaries)
        
        self.assertIn("First message", result)
        self.assertIn("Response", result)
    
    def test_multi_batch_processing(self):
        """Test processing multiple batches"""
        # Create 15 summaries (exceeds batch size of 10)
        summaries = [
            TurnSummary(i, "user" if i % 2 == 0 else "assistant",
                       "content", f"Summary {i}", 100, 100)
            for i in range(15)
        ]
        
        result = self.summarizer.summarize_all(summaries)
        
        # Should process in 2 batches and combine
        self.assertIsNotNone(result)


class TestLevel3Summarizer(unittest.TestCase):
    
    def test_no_api_key_fallback(self):
        """Should fallback to truncation without API key"""
        summarizer = Level3Summarizer(api_key=None)
        
        text = "A" * 10000
        result = summarizer.compact(text, target_tokens=100)
        
        self.assertLess(len(result), len(text))
        self.assertIn("[truncated", result)
    
    def test_already_small_no_change(self):
        """Small input should not be changed"""
        summarizer = Level3Summarizer(api_key=None)
        
        text = "Small text"
        result = summarizer.compact(text, target_tokens=1000)
        
        self.assertEqual(result, text)


class TestRAGPreRanker(unittest.TestCase):
    
    def setUp(self):
        self.ranker = RAGPreRanker()
    
    @patch('context_compaction.requests.post')
    def test_prerank_documents(self, mock_post):
        """Test document pre-ranking"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "8"}
        mock_post.return_value = mock_response
        
        query = "database connection"
        docs = [
            "About databases and connections",
            "Something unrelated",
            "Database configuration guide"
        ]
        
        result = self.ranker.prerank(query, docs, top_k=2)
        
        self.assertEqual(len(result), 2)
        mock_post.assert_called()
    
    def test_prerank_small_list(self):
        """Small document lists should return all"""
        query = "test"
        docs = ["doc1", "doc2"]
        
        result = self.ranker.prerank(query, docs, top_k=5)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][1], 1.0)  # Full score
    
    def test_keyword_fallback(self):
        """Test keyword fallback scoring"""
        query = "database error"
        doc = "This document is about database connection errors"
        
        score = self.ranker._keyword_fallback(query, doc)
        
        self.assertGreater(score, 0.5)  # Should have good overlap
    
    def test_keyword_fallback_no_overlap(self):
        """Test keyword fallback with no overlap"""
        query = "database"
        doc = "This is about something completely different"
        
        score = self.ranker._keyword_fallback(query, doc)
        
        self.assertEqual(score, 0.0)


class TestContextCompactor(unittest.TestCase):
    
    def setUp(self):
        self.compactor = ContextCompactor()
    
    @patch.object(Level1Summarizer, 'summarize_turns')
    @patch.object(Level2Summarizer, 'summarize_all')
    def test_compact_stops_at_level1(self, mock_l2, mock_l1):
        """Should stop at Level 1 if small enough"""
        mock_l1.return_value = [
            TurnSummary(0, "user", "content", "Short summary", 100, 50)
        ]
        
        history = [{"role": "user", "content": "Hello"}]
        result = self.compactor.compact(history, max_output_tokens=1000)
        
        self.assertEqual(result.levels_used, [CompactionLevel.LEVEL1_7B])
        mock_l2.assert_not_called()
    
    @patch.object(Level1Summarizer, 'summarize_turns')
    @patch.object(Level2Summarizer, 'summarize_all')
    def test_compact_reaches_level2(self, mock_l2, mock_l1):
        """Should reach Level 2 if Level 1 too large"""
        mock_l1.return_value = [
            TurnSummary(i, "user", "content", "Summary " * 100, 500, 400)
            for i in range(10)
        ]
        mock_l2.return_value = "Level 2 synthesis"
        
        history = [{"role": "user", "content": "Msg"} for _ in range(10)]
        result = self.compactor.compact(history, max_output_tokens=200)
        
        self.assertIn(CompactionLevel.LEVEL1_7B, result.levels_used)
        self.assertIn(CompactionLevel.LEVEL2_32B, result.levels_used)
    
    @patch.object(Level1Summarizer, 'summarize_turns')
    @patch.object(Level2Summarizer, 'summarize_all')
    @patch.object(Level3Summarizer, 'compact')
    def test_compact_reaches_level3(self, mock_l3, mock_l2, mock_l1):
        """Should reach Level 3 if Level 2 too large"""
        mock_l1.return_value = [
            TurnSummary(i, "user", "content", "Summary " * 100, 500, 400)
            for i in range(20)
        ]
        mock_l2.return_value = "Large level 2 output " * 100
        mock_l3.return_value = "Compact level 3 output"
        
        history = [{"role": "user", "content": "Msg"} for _ in range(20)]
        result = self.compactor.compact(history, max_output_tokens=100)
        
        self.assertIn(CompactionLevel.LEVEL3_CLAUDE, result.levels_used)
    
    def test_savings_calculation(self):
        """Test token savings calculation"""
        history = [
            {"role": "user", "content": "This is a test message"},
            {"role": "assistant", "content": "This is a response"}
        ]
        
        with patch.object(Level1Summarizer, 'summarize_turns') as mock_l1:
            mock_l1.return_value = [
                TurnSummary(0, "user", "content", "Test", 100, 10),
                TurnSummary(1, "assistant", "content", "Response", 100, 10)
            ]
            
            result = self.compactor.compact(history, max_output_tokens=1000)
            
            self.assertGreater(result.token_savings, 0)
            self.assertGreater(result.savings_percentage, 0)
    
    def test_prerank_integration(self):
        """Test RAG pre-ranking integration"""
        with patch.object(RAGPreRanker, 'prerank') as mock_prerank:
            mock_prerank.return_value = [(2, 0.9), (0, 0.8), (1, 0.7)]
            
            indices = self.compactor.prerank_for_rag(
                "query", ["doc1", "doc2", "doc3"], top_k=2
            )
            
            self.assertEqual(indices, [2, 0])


class TestCompactionResult(unittest.TestCase):
    
    def test_result_structure(self):
        """Test CompactionResult has all required fields"""
        result = CompactionResult(
            original_text="Original",
            compacted_text="Compacted",
            original_tokens=1000,
            compacted_tokens=500,
            levels_used=[CompactionLevel.LEVEL1_7B],
            level_summaries={1: "Level 1"},
            processing_time_ms=100.0,
            token_savings=500,
            savings_percentage=50.0
        )
        
        self.assertEqual(result.original_tokens, 1000)
        self.assertEqual(result.compacted_tokens, 500)
        self.assertEqual(result.token_savings, 500)
        self.assertEqual(result.savings_percentage, 50.0)


class TestIntegration(unittest.TestCase):
    """Integration-style tests"""
    
    @patch('context_compaction.requests.post')
    def test_end_to_end_compaction(self, mock_post):
        """Test full compaction pipeline"""
        # Mock Ollama responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Summarized content"}
        mock_post.return_value = mock_response
        
        compactor = ContextCompactor()
        
        # Create realistic conversation history
        history = [
            {"role": "user", "content": "How do I deploy this?"},
            {"role": "assistant", "content": "You can use docker-compose. " * 50},
            {"role": "user", "content": "What about Kubernetes?"},
            {"role": "assistant", "content": "Kubernetes is great. " * 50}
        ]
        
        result = compactor.compact(history, max_output_tokens=500)
        
        self.assertIsNotNone(result.compacted_text)
        self.assertGreater(result.original_tokens, result.compacted_tokens)
        self.assertGreater(result.savings_percentage, 0)
    
    def test_rag_workflow(self):
        """Test RAG pre-ranking workflow"""
        with patch.object(RAGPreRanker, '_score_relevance') as mock_score:
            mock_score.side_effect = [0.9, 0.5, 0.8, 0.3, 0.7]
            
            compactor = ContextCompactor()
            query = "database optimization"
            documents = [
                "About database optimization techniques",
                "Frontend CSS styling guide",
                "Database indexing strategies",
                "Cooking recipes",
                "Query performance tuning"
            ]
            
            top_indices = compactor.prerank_for_rag(query, documents, top_k=3)
            
            self.assertEqual(len(top_indices), 3)
            # Highest scores should be indices 0, 2, 4
            self.assertIn(0, top_indices)


class TestEdgeCases(unittest.TestCase):
    
    def test_empty_history(self):
        """Empty history should return empty result"""
        compactor = ContextCompactor()
        result = compactor.compact([], max_output_tokens=1000)
        
        self.assertEqual(result.original_tokens, 0)
        self.assertEqual(result.compacted_tokens, 0)
    
    def test_single_turn(self):
        """Single turn should work"""
        compactor = ContextCompactor()
        
        with patch.object(Level1Summarizer, 'summarize_turns') as mock_l1:
            mock_l1.return_value = [
                TurnSummary(0, "user", "Hello", "Hello", 10, 10)
            ]
            
            result = compactor.compact(
                [{"role": "user", "content": "Hello"}],
                max_output_tokens=1000
            )
            
            self.assertEqual(result.levels_used, [CompactionLevel.LEVEL1_7B])
    
    def test_very_large_history(self):
        """Very large history should be handled"""
        compactor = ContextCompactor()
        
        # Create 100 turns
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", 
             "content": f"Message {i}"}
            for i in range(100)
        ]
        
        with patch.object(Level1Summarizer, 'summarize_turns') as mock_l1, \
             patch.object(Level2Summarizer, 'summarize_all') as mock_l2:
            
            mock_l1.return_value = [
                TurnSummary(i, "user" if i % 2 == 0 else "assistant",
                           "content", f"Summary {i}", 100, 50)
                for i in range(100)
            ]
            mock_l2.return_value = "Batch synthesis of all turns"
            
            result = compactor.compact(history, max_output_tokens=500)
            
            self.assertIsNotNone(result.compacted_text)
            mock_l2.assert_called_once()


if __name__ == "__main__":
    unittest.main()
