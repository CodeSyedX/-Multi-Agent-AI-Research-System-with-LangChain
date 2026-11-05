import unittest
from state import ResearchState
from vector_store import create_local_vector_db
from graph import writer_node, finalizing_node

class TestPipelineUnits(unittest.TestCase):

    def setUp(self):
        self.sample_scraped_content = """
Title: Tesla EV Market Expansion
URL: https://example.com/tesla-ev
Snippet: Tesla continues to lead the EV market with new automated assembly updates and LangGraph integration for research pipelines.

Title: Battery Tech Innovations
URL: https://example.com/battery-tech
Snippet: Next-gen lithium iron phosphate batteries offer 20% higher energy density in standard chassis models.
"""
        self.sample_state: ResearchState = {
            "topic": "Tesla EV market and LangGraph routing updates",
            "language": "English",
            "search_results": self.sample_scraped_content,
            "scraped_content": self.sample_scraped_content,
            "report": "",
            "feedback": "",
            "revision_count": 0,
            "source_map": {}
        }

    def test_01_vector_store_indexing(self):
        print("\n🧪 --- Test 1: Testing local RAG indexing & metadata parsing ---")
        retriever = create_local_vector_db(self.sample_scraped_content)
        docs = retriever.invoke("Tesla EV market")
        self.assertTrue(len(docs) > 0, "Retriever should return relevant document chunks.")
        self.assertIn("source_url", docs[0].metadata, "Metadata should preserve source_url.")
        print("✅ Test 1 Passed: Local vector store metadata indexed cleanly.")

    def test_02_writer_node_execution(self):
        print("\n🧪 --- Test 2: Testing Writer Node execution & source_map creation ---")
        result = writer_node(self.sample_state)
        
        self.assertIn("report", result, "Writer node output must contain 'report'.")
        self.assertIn("source_map", result, "Writer node output must contain 'source_map'.")
        self.assertIsInstance(result["report"], str, "'report' must be a string.")
        self.assertIsInstance(result["source_map"], dict, "'source_map' must be a dict.")
        
        print("📝 --- Generated Draft Snippet ---")
        print(result["report"][:250] + "...\n")
        print("✅ Test 2 Passed: Writer node executed cleanly without type errors.")

    def test_03_finalizing_node_citation_mapping(self):
        print("\n🧪 --- Test 3: Testing Finalizer Node citation resolution ---")
        state_with_draft: ResearchState = {
            "topic": "Tesla EV market",
            "language": "English",
            "search_results": "",
            "scraped_content": "",
            "report": "Tesla leads the market in automated assembly [1]. Battery density improved significantly [2].",
            "feedback": "APPROVED",
            "revision_count": 1,
            "source_map": {
                "[1]": {"index": 1, "title": "Tesla EV Market Expansion", "url": "https://example.com/tesla-ev"},
                "[2]": {"index": 2, "title": "Battery Tech Innovations", "url": "https://example.com/battery-tech"}
            }
        }
        
        final_result = finalizing_node(state_with_draft)
        final_report = final_result["report"]
        
        self.assertIn("## References & Source Links", final_report)
        self.assertIn("https://example.com/tesla-ev", final_report)
        self.assertIn("https://example.com/battery-tech", final_report)
        
        print("📝 --- Final Output Report with References ---")
        print(final_report)
        print("✅ Test 3 Passed: Finalizer node correctly mapped citation markers to real URLs.")

if __name__ == "__main__":
    unittest.main()