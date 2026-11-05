from langchain_core.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

# Tavily client
tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# -----------------------------
# Web Search Function (Structured for State & Source Map)
# -----------------------------

def web_search_with_sources(query: str, max_results: int = 5):
    """
    Performs web search and returns both structured source mapping 
    for citations and raw search text for LLM consumption.
    """
    try:
        results = tavily.search(
            query=query,
            max_results=max_results
        )
    except Exception as e:
        return "", {}

    output = []
    source_map = {}

    for idx, r in enumerate(results.get("results", []), start=1):
        citation_key = f"[{idx}]"
        
        # Save structured source metadata for the finalizer agent
        source_map[citation_key] = {
            "title": r.get("title", f"Source {idx}"),
            "url": r.get("url", ""),
            "snippet": r.get("content", "")
        }

        # Formatted string snippet labeled with citation key
        output.append(
            f"""{citation_key} Title: {r.get('title')}
URL: {r.get('url')}
Snippet: {r.get('content')[:300]}"""
        )

    search_text = "\n----------------\n".join(output)
    return search_text, source_map


# -----------------------------
# Web Search Tool (LangChain Tool Wrapper)
# -----------------------------

@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information.
    Returns titles, URLs and snippets.
    """
    search_text, _ = web_search_with_sources(query)
    return search_text


# -----------------------------
# URL Scraper Tool
# -----------------------------

@tool
def scrape_url(url: str) -> str:
    """
    Scrape and return clean text content from a URL.
    """
    try:
        response = requests.get(
            url,
            timeout=8,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unnecessary HTML elements
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        return text[:3000]

    except Exception as e:
        return f"Could not scrape URL: {str(e)}"