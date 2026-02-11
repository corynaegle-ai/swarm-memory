import unittest
from unittest.mock import Mock, patch
from tool_prefilter import (
    ToolSummarizer,
    ToolRegistry,
    ToolConfig,
    SummaryFormatter,
    MetricsTracker,
    SummaryResult,
    DEFAULT_OLLAMA_ENDPOINT
)


class TestToolRegistry(unittest.TestCase):
    
    def test_get_existing_config(self):
        registry = ToolRegistry()
        config = registry.get_config("file_read")
        self.assertIsNotNone(config)
        self.assertEqual(config.name, "file_read")
        self.assertEqual(config.token_threshold, 1000)
    
    def test_get_missing_config(self):
        registry = ToolRegistry()
        config = registry.get_config("nonexistent_tool")
        self.assertIsNone(config)
    
    def test_should_summarize_above_threshold(self):
        registry = ToolRegistry()
        # file_read threshold is 1000 tokens
        large_output = "x" * 5000  # ~1250 tokens
        self.assertTrue(registry.should_summarize("file_read", large_output))
    
    def test_should_not_summarize_below_threshold(self):
        registry = ToolRegistry()
        small_output = "x" * 100  # ~25 tokens
        self.assertFalse(registry.should_summarize("file_read", small_output))
    
    def test_should_not_summarize_unknown_tool(self):
        registry = ToolRegistry()
        large_output = "x" * 10000
        self.assertFalse(registry.should_summarize("unknown", large_output))
    
    def test_token_estimation(self):
        registry = ToolRegistry()
        # 1 token ≈ 4 chars
        text = "x" * 400
        self.assertEqual(registry._estimate_tokens(text), 100)
    
    def test_register_new_tool(self):
        registry = ToolRegistry()
        config = ToolConfig(
            name="custom_tool",
            token_threshold=500,
            summary_focus="custom focus",
            preserve_patterns=[]
        )
        registry.register_tool(config)
        
        retrieved = registry.get_config("custom_tool")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.token_threshold, 500)


class TestSummaryFormatter(unittest.TestCase):
    
    def test_format_summary(self):
        formatter = SummaryFormatter()
        summary = formatter.format_summary(
            "test_tool",
            "This is a summary",
            {"original_tokens": 2000, "summary_tokens": 50}
        )
        
        self.assertIn("[Tool: test_tool - Summarized Output]", summary)
        self.assertIn("2000 tokens", summary)
        self.assertIn("This is a summary", summary)
    
    def test_format_summary_with_preserved(self):
        formatter = SummaryFormatter()
        summary = formatter.format_summary(
            "test_tool",
            "Summary text",
            {
                "original_tokens": 1000,
                "summary_tokens": 100,
                "preserved_items": ["import os", "def main():"]
            }
        )
        
        self.assertIn("import os", summary)
        self.assertIn("def main():", summary)
    
    def test_format_error(self):
        formatter = SummaryFormatter()
        original = "x" * 3000
        result = formatter.format_error("test_tool", "Connection failed", original)
        
        self.assertIn("Summary Failed", result)
        self.assertIn("Connection failed", result)
        self.assertIn(original[:100], result)  # Truncated original


class TestMetricsTracker(unittest.TestCase):
    
    def test_record_summarized(self):
        metrics = MetricsTracker()
        metrics.record("file_read", 2000, 200, True, 150.0)
        
        stats = metrics.get_stats()
        self.assertEqual(stats["total_calls"], 1)
        self.assertEqual(stats["summarized_calls"], 1)
        self.assertEqual(stats["total_tokens_saved"], 1800)
    
    def test_record_not_summarized(self):
        metrics = MetricsTracker()
        metrics.record("file_read", 2000, 2000, False, 10.0)
        
        stats = metrics.get_stats()
        self.assertEqual(stats["total_calls"], 1)
        self.assertEqual(stats["summarized_calls"], 0)
        self.assertEqual(stats["total_tokens_saved"], 0)
    
    def test_summarization_rate(self):
        metrics = MetricsTracker()
        metrics.record("file_read", 2000, 200, True, 100.0)
        metrics.record("file_read", 500, 500, False, 10.0)
        metrics.record("file_read", 3000, 300, True, 120.0)
        
        stats = metrics.get_stats()
        self.assertEqual(stats["summarization_rate"], 2/3)
    
    def test_tool_breakdown(self):
        metrics = MetricsTracker()
        metrics.record("file_read", 2000, 200, True, 100.0)
        metrics.record("directory_list", 1000, 150, True, 80.0)
        
        stats = metrics.get_stats()
        self.assertIn("file_read", stats["tool_breakdown"])
        self.assertIn("directory_list", stats["tool_breakdown"])
        self.assertEqual(stats["tool_breakdown"]["file_read"]["tokens_saved"], 1800)


class TestToolSummarizer(unittest.TestCase):
    
    def setUp(self):
        self.summarizer = ToolSummarizer()
    
    @patch('tool_prefilter.requests.post')
    def test_summarize_below_threshold_returns_original(self, mock_post):
        """Small outputs should not be summarized"""
        output = "Small output"  # Well below 1000 token threshold
        result = self.summarizer.summarize("file_read", output)
        
        self.assertFalse(result.was_summarized)
        self.assertEqual(result.summary, output)
        mock_post.assert_not_called()
    
    @patch('tool_prefilter.requests.post')
    def test_summarize_above_threshold_calls_ollama(self, mock_post):
        """Large outputs should be summarized"""
        # Mock Ollama response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Summary of large file"}
        mock_post.return_value = mock_response
        
        # Large output above threshold
        output = "x" * 5000  # ~1250 tokens
        result = self.summarizer.summarize("file_read", output, {"path": "/test.py"})
        
        self.assertTrue(result.was_summarized)
        self.assertIn("Summary of large file", result.summary)
        mock_post.assert_called_once()
    
    @patch('tool_prefilter.requests.post')
    def test_summarize_handles_ollama_error(self, mock_post):
        """Errors should fallback to truncated original"""
        mock_post.side_effect = Exception("Connection refused")
        
        output = "x" * 5000
        result = self.summarizer.summarize("file_read", output)
        
        self.assertFalse(result.was_summarized)
        self.assertIn("Summary Failed", result.summary)
        self.assertIn("Connection refused", result.summary)
    
    def test_extract_preserved_items(self):
        output = """
import os
import sys

def main():
    pass

# TODO: fix this
"""
        config = self.summarizer.registry.get_config("file_read")
        preserved = self.summarizer._extract_preserved_items(output, config.preserve_patterns)
        
        self.assertIn("import os", preserved)
        self.assertIn("import sys", preserved)
        self.assertIn("def main():", preserved)
        self.assertIn("# TODO: fix this", preserved)
    
    def test_unknown_tool_not_summarized(self):
        """Unknown tools should pass through unchanged"""
        output = "x" * 10000
        result = self.summarizer.summarize("unknown_tool", output)
        
        self.assertFalse(result.was_summarized)
        self.assertEqual(result.original_output, output)
    
    def test_metrics_tracking(self):
        """Metrics should be tracked"""
        # Record some events
        self.summarizer.metrics.record("file_read", 2000, 200, True, 100.0)
        self.summarizer.metrics.record("file_read", 500, 500, False, 10.0)
        
        stats = self.summarizer.get_metrics()
        self.assertEqual(stats["total_calls"], 2)
        self.assertEqual(stats["summarized_calls"], 1)


class TestIntegration(unittest.TestCase):
    """Integration-style tests"""
    
    @patch('tool_prefilter.requests.post')
    def test_directory_listing_summarization(self, mock_post):
        """Test summarizing a directory listing"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Directory contains 15 Python files, 3 config files, and a README"
        }
        mock_post.return_value = mock_response
        
        # Simulate a large directory listing
        listing = "\n".join([f"file_{i}.py" for i in range(100)])  # 100 files
        listing += "\n" + "x" * 2000  # Make it large enough
        
        summarizer = ToolSummarizer()
        result = summarizer.summarize("directory_list", listing)
        
        self.assertTrue(result.was_summarized)
        self.assertIn("Directory", result.summary)
    
    @patch('tool_prefilter.requests.post')
    def test_log_output_summarization(self, mock_post):
        """Test summarizing log output"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "3 ERRORs found: database connection failed, retrying..."
        }
        mock_post.return_value = mock_response
        
        # Simulate log output with errors
        log = "\n".join([
            "2024-01-01 INFO: Starting...",
            "2024-01-01 ERROR: Database connection failed",
            "2024-01-01 INFO: Retrying...",
        ] * 100)  # Large log
        
        summarizer = ToolSummarizer()
        result = summarizer.summarize("log_output", log)
        
        self.assertTrue(result.was_summarized)


class TestEdgeCases(unittest.TestCase):
    
    def test_empty_output(self):
        """Empty output should not be summarized"""
        summarizer = ToolSummarizer()
        result = summarizer.summarize("file_read", "")
        
        self.assertFalse(result.was_summarized)
        self.assertEqual(result.summary, "")
    
    def test_very_long_output_truncation(self):
        """Very long outputs should be truncated before sending to Ollama"""
        from unittest.mock import Mock
        summarizer = ToolSummarizer()
        
        # Mock the Ollama response
        with patch('tool_prefilter.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "Summary of truncated content"}
            mock_post.return_value = mock_response
            
            # Very long output (would be > 8000 chars)
            output = "x" * 20000
            result = summarizer.summarize("file_read", output)
            
            # Check that the prompt sent to Ollama was truncated
            call_args = mock_post.call_args
            prompt = call_args[1]['json']['prompt']
            self.assertLess(len(prompt), 15000)  # Prompt should include truncated output
            self.assertIn("truncated for summary", prompt)
    
    def test_preserves_exact_threshold(self):
        """Output exactly at threshold should be summarized"""
        registry = ToolRegistry()
        # file_read threshold is 1000 tokens = 4000 chars
        exact_output = "x" * 4000
        self.assertTrue(registry.should_summarize("file_read", exact_output))


if __name__ == "__main__":
    unittest.main()
