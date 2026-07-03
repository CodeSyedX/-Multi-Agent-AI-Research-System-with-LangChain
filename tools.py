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
# Web Search Tool
# -----------------------------

@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information.
    Returns titles, URLs and snippets.
    """

    results = tavily.search(
        query=query,
        max_results=5
    )

    output = []

    for r in results["results"]:
        output.append(
            f"""
Title: {r['title']}
URL: {r['url']}
Snippet: {r['content'][:300]}
"""
        )

    return "\n----------------\n".join(output)



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


        # Remove unnecessary HTML
        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "footer"
            ]
        ):
            tag.decompose()


        text = soup.get_text(
            separator=" ",
            strip=True
        )

        return text[:3000]


    except Exception as e:
        return f"Could not scrape URL: {str(e)}"