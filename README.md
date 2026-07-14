# 🤖 Cyber-Matrix Multi-Agent AI Research Engine with LangGraph

An advanced, production-grade **Multi-Agent AI Research System** built with **LangGraph, LangChain, FastAPI, Hugging Face, ChromaDB**.

Unlike traditional single-prompt LLM applications, this project implements a **stateful autonomous research workflow** where multiple AI agents collaborate, critique, refine, and generate well-structured research reports with inline citations. The system combines **live web retrieval**, **local Retrieval-Augmented Generation (RAG)**, **iterative review loops**, and an interactive cyberpunk-inspired dashboard.

---

# 🚀 Features

- 🕵️ **Autonomous Search Agent**
  - Performs live web research using Tavily Search API.
  - Collects relevant URLs, snippets, and metadata.

- 🧠 **Privacy-First Local RAG**
  - Uses **sentence-transformers/all-MiniLM-L6-v2**
  - Generates embeddings locally.
  - Stores vectors in an in-memory Chroma database.
  - No cloud embedding APIs required.

- ✍️ **Citation-Aware Writer Agent**
  - Retrieves the most relevant semantic chunks.
  - Produces structured reports.
  - Automatically inserts inline citations like:
    ```
    [1]
    [2]
    [3]
    ```

- ⚖️ **Master Critic Agent**
  - Reviews grammar
  - Detects hallucinations
  - Validates chronology
  - Enforces timeline consistency
  - Supports iterative refinement (up to 3 revision loops)

- 🎨 **Finalizer Agent**
  - Generates polished Markdown reports
  - Creates a hyperlinked References section
  - Maps citations to original sources automatically

- 🌍 **Multi-Language Support**
  - English
  - Hindi
  - Spanish
  - French
  - Easily extensible to additional languages


---

# 🏗️ Project Architecture

```text
CAPS/
│
├── .venv/                         # Virtual environment (ignored by Git)
├── __pycache__/                   # Python cache
│
├── .env                           # API Keys
├── .gitignore
├── requirements.txt
│
├── state.py                       # TypedDict research state
├── tools.py                       # Tavily search & scraping utilities
├── vector_store.py                # Local Chroma + Hugging Face embeddings
├── graph.py                       # LangGraph workflow orchestration
├── main.py                        # FastAPI backend
├── pipeline.py                    # CLI research pipeline
├── test_pipeline_units.py         # Unit tests
│
└── README.md
```

---

# 🧠 Multi-Agent Workflow

```text
                    USER QUERY
                        │
                        ▼
            ┌──────────────────────┐
            │   Search Agent        │
            │  Live Web Retrieval   │
            └──────────────────────┘
                        │
                        ▼
            ┌──────────────────────┐
            │   Writer Agent        │
            │ Local RAG + Citations │
            └──────────────────────┘
                        │
                        ▼
            ┌──────────────────────┐
            │   Master Critic       │
            │ Grammar + Validation  │
            └──────────────────────┘
                     │
         Rejected ───┘
              ▲
              │
      Maximum 3 Revision Loops
              │
              ▼
            Approved
                │
                ▼
        ┌────────────────────┐
        │  Finalizer Agent   │
        │ References Builder │
        └────────────────────┘
                │
                ▼
         Final Research Report
```

---

# ⚙️ Core Agents

## 🕵️ Search Agent (`search_node`)

Responsibilities:

- Searches the web using Tavily
- Retrieves relevant URLs
- Extracts snippets
- Collects metadata
- Returns structured search results

---

## ✍️ Writer Agent (`writer_node`)

Responsibilities:

- Cleans retrieved content
- Splits text into chunks
- Generates embeddings locally
- Stores vectors inside ChromaDB
- Retrieves top semantic matches
- Generates a detailed report
- Inserts inline citations

Example:

```
Artificial Intelligence is rapidly transforming healthcare.[1]

Recent studies indicate significant improvements in diagnostic accuracy.[2]
```

---

## ⚖️ Master Critic Agent (`critic_node`)

Responsibilities:

- Grammar validation
- Hallucination detection
- Timeline validation
- Chronological consistency
- Research quality evaluation

If issues are detected:

```
Rejected
↓
Feedback Generated
↓
Writer Revises Draft
↓
Critic Reviews Again
```

Maximum revision loops:

```
3
```

---

## 🎨 Finalizer Agent (`finalizing_node`)

Responsibilities:

- Reads approved report
- Resolves citations
- Generates references
- Produces polished Markdown output

Example:

```markdown
## References

[1] https://example.com/article

[2] https://example2.com/report
```

---

# 📦 Research State Schema

The pipeline maintains all workflow information inside a centralized TypedDict.

```python
class ResearchState(TypedDict):
    topic: str
    language: str
    search_results: str
    scraped_content: str
    report: str
    feedback: str
    revision_count: int
    source_map: dict
```

State variables are automatically shared between every LangGraph node.

---

# 🧠 Local RAG Pipeline

```text
Live Search
      │
      ▼
Collected Pages
      │
      ▼
Text Cleaning
      │
      ▼
Chunk Generation
      │
      ▼
MiniLM Embeddings
      │
      ▼
Chroma Vector Store
      │
      ▼
Semantic Retrieval
      │
      ▼
Writer Agent
```

---


# 🛠️ Technology Stack

## AI

- LangGraph
- LangChain
- Google Gemini
- Hugging Face Transformers

## Retrieval

- ChromaDB
- sentence-transformers
- Local RAG

## Backend

- Python
- FastAPI



## APIs

- Tavily Search API
- Gemini API

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/CodeSyedX/-Multi-Agent-AI-Research-System-with-LangChain.git

cd -Multi-Agent-AI-Research-System-with-LangChain
```

---

## Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_gemini_api_key

TAVILY_API_KEY=your_tavily_api_key
```

---

# 🧪 Run Unit Tests

```bash
python test_pipeline_units.py
```

This verifies:

- Search utilities
- Regex extraction
- Local embeddings
- Chroma indexing
- Citation generation
- Pipeline integrity

---

# ▶️ Run CLI Pipeline

```bash
python pipeline.py
```

---

# 🚀 Run FastAPI Backend

```bash
uvicorn main:app --reload
```

---

# 🌐 Launch Frontend

```bash
cd frontend

npm install

npm run dev
```

---

# 📈 Workflow Summary

```text
User Query
      │
      ▼
Search Agent
      │
      ▼
Writer Agent
      │
      ▼
Local RAG
      │
      ▼
Critic Agent
      │
 Approved?
   │
   ├── No → Revision Loop
   │
   └── Yes
          │
          ▼
Finalizer Agent
          │
          ▼
Markdown Report
          │
          ▼
Interactive Dashboard
```

---

# 🔮 Future Enhancements

- Persistent ChromaDB storage
- PDF report generation
- Multi-document RAG
- Streaming agent execution
- Research history dashboard
- User authentication
- Docker deployment
- Kubernetes support
- Redis caching
- Multi-model LLM routing
- Real-time collaborative research
- Export to DOCX and PDF

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Syed Aafreen**

AI Engineer • Full Stack Developer • Multi-Agent Systems Enthusiast

If you found this project useful, consider giving it a ⭐ on GitHub.