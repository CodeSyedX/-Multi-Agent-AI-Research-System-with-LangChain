import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Import your graph pieces
from graph import app as graph_app
from state import ResearchState

load_dotenv()

app = FastAPI(title="Matrix LangGraph Engine API")

# Perfect CORS adjustments for local file testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    topic: str
    language: str = "English"

@app.post("/api/research/run")
async def run_research_pipeline(request: ResearchRequest):
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        raise HTTPException(status_code=500, detail="API keys are unconfigured in your .env file.")

    initial_state: ResearchState = {
        "topic": request.topic,
        "language": request.language,
        "search_results": "",
        "scraped_content": "",
        "report": "",
        "feedback": "",
        "revision_count": 0,
        "source_map": {}
    }

    try:
        # Run the LangGraph execution flow synchronously to return the complete dataset
        final_state = graph_app.invoke(initial_state)
        
        return {
            "report": final_state.get("report"),
            "source_map": final_state.get("source_map", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)