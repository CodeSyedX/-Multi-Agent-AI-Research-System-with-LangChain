# 🤖 Multi-Agent AI Research System with LangChain

An intelligent **Multi-Agent AI Research System** built using **LangChain** that automates the complete research workflow.

The system leverages multiple AI agents that collaborate to:

- 🔍 Search the web for reliable information
- 📖 Scrape and extract detailed content
- ✍️ Generate a structured research report
- 📝 Critically review the generated report

Instead of relying on a single LLM prompt, this project follows a **Multi-Agent Architecture**, where each agent is responsible for a specialized task.

---

# 🚀 Features

- AI-powered Search Agent
- Intelligent Reader Agent for web scraping
- Report Writer Agent
- Critic Agent for quality review
- Multi-language report generation
- Modular LangChain architecture
- Easy to extend with new agents

---

# 🏗️ Project Architecture

```
CAPS/
│
├── .venv/
├── __pycache__/
│
├── .env
├── .gitignore
├── requirements.txt
│
├── agents.py          # Builds all AI agents
├── tools.py           # Search & scraping tools
├── pipeline.py        # Complete research workflow
│
└── README.md
```

---

# 🧠 Workflow

```
             User Topic
                  │
                  ▼
        🔍 Search Agent
                  │
                  ▼
       📖 Reader Agent
                  │
                  ▼
       ✍️ Writer Agent
                  │
                  ▼
       📝 Critic Agent
                  │
                  ▼
        Final Research Report
```

---

# 🤖 Agents

## 1. Search Agent

Responsible for:

- Searching the internet
- Finding recent and reliable information
- Collecting useful search results

### Output

- Relevant search summaries
- URLs
- Sources

---

## 2. Reader Agent

Responsible for:

- Selecting the most relevant URL
- Scraping detailed content
- Extracting important information

### Output

- Cleaned webpage content
- Detailed research material

---

## 3. Writer Agent

Responsible for:

- Combining search results
- Using scraped content
- Writing a professional research report

Supports multiple languages.

### Output

- Structured report
- Headings
- Explanations
- Well-organized content

---

## 4. Critic Agent

Responsible for reviewing the generated report.

Checks:

- Accuracy
- Completeness
- Grammar
- Structure
- Missing information
- Suggestions for improvement

---

# ⚙️ Research Pipeline

The workflow is implemented inside **pipeline.py**.

## Step 1 — Search

The Search Agent receives the research topic and retrieves reliable information from the web.

```python
search_agent.invoke(...)
```

Output is stored as:

```python
state["search_results"]
```

---

## Step 2 — Read

The Reader Agent selects the most relevant URL from the search results and scrapes detailed content.

```python
reader_agent.invoke(...)
```

Output is stored as:

```python
state["scraped_content"]
```

---

## Step 3 — Write

The Writer Agent combines:

- Search results
- Scraped webpage content

The user can also choose the preferred report language.

```python
writer_chain.invoke(...)
```

Output is stored as:

```python
state["report"]
```

---

## Step 4 — Critic

The Critic Agent evaluates the generated report and provides constructive feedback.

```python
critic_chain.invoke(...)
```

Output is stored as:

```python
state["feedback"]
```

---

# 📦 State Management

The pipeline maintains all intermediate outputs inside a dictionary.

```python
state = {
    "search_results": "...",
    "scraped_content": "...",
    "report": "...",
    "feedback": "..."
}
```

This makes the workflow easy to debug and extend.

---

# 🛠️ Technologies Used

- Python
- LangChain
- LangGraph
- OpenAI / LLMs
- Tavily Search
- Web Scraping Tools
- Prompt Engineering
- Multi-Agent Systems

---

# 📥 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Multi-Agent-AI-Research-System.git
```

Move into the project directory:

```bash
cd Multi-Agent-AI-Research-System
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file and add your API keys.

Example:

```env
OPENAI_API_KEY=your_openai_api_key

TAVILY_API_KEY=your_tavily_api_key
```

---

# ▶️ Running the Project

Run the pipeline:

```bash
python pipeline.py
```

Example:

```
Enter a research topic:

Artificial Intelligence in Healthcare
```

Then select a language:

```
English
```

The system automatically:

- Searches the web
- Reads the best source
- Generates a report
- Reviews the report

---

# 📌 Example Output

```
Enter research topic:
Artificial Intelligence

↓

Search Agent

↓

Reader Agent

↓

Writer Agent

↓

Critic Agent

↓

Final Research Report
```

---

# 🔮 Future Improvements

- PDF report generation
- Citation management
- Multi-source research
- Research memory
- Agent routing using LangGraph
- Streamlit web interface
- Export to Word/PDF
- Research history database

---

# 🎯 Learning Outcomes

This project demonstrates:

- Multi-Agent AI Systems
- LangChain Agent Development
- AI Workflow Orchestration
- Prompt Engineering
- Web Search Integration
- Web Scraping
- LLM Chains
- Modular AI Architecture

---

# 📜 License

This project is intended for educational and learning purposes.