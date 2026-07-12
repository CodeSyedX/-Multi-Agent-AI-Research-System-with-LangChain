from graph import app

def run_research_pipeline(topic: str) -> dict:
    language = input("Enter preferred language (English/Hindi/etc): ").strip() or "English"

    initial_state = {
        "topic": topic,
        "language": language,
        "search_results": "",
        "scraped_content": "",
        "report": "",
        "feedback": "",
        "revision_count": 0
    }

    print("\n🤖 Initiating LangGraph Research Workflow Node System...")
    
    final_state = app.invoke(initial_state)

    print("\n" + "=" * 50)
    print("🏆 FINAL COMPILED REPORT")
    print("=" * 50)
    print(final_state["report"])
    print("=" * 50)

    return final_state

if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    run_research_pipeline(topic)