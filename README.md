# RAG-Powered Medical Manual QA System

A Streamlit-based conversational AI assistant that answers questions about 
medical manuals, EHR documentation, and healthcare policy documents. 
Supports multi-provider LLM backends (OpenAI, Claude, Gemini) with 
semantic search, intent classification, confidence scoring, and 
persistent conversation history.

**Built for:** CPC Integrated Health  
**Authors:** Dahana Moz Ruiz & Marcelle Tamegnon

---

## Features

### Core QA Engine
- **Semantic search** — Uses `sentence-transformers/all-MiniLM-L6-v2` 
  to retrieve the most relevant chunks from the uploaded manual
- **Intent classification** — Automatically routes questions into 7 
  categories: Eligibility, Procedure, Coverage, Documentation, Contact, 
  Definition, Other
- **Confidence scoring** — Color-coded confidence indicator (🟢/🟡/🔴) 
  with automatic clarifying follow-up questions when confidence is low
- **Named entity recognition** — Extracts program names (Medicaid, NEMT, 
  LogistiCare, etc.) using spaCy

### LLM Integration (chatbox2.py)
- **Multi-provider support** — Switch between OpenAI, Anthropic (Claude), 
  and Google Gemini from the sidebar
- **Configurable settings** — Adjust model, temperature, and max tokens 
  per session
- **Factual LLM refinement** — Optional LLM post-processing to improve 
  answer quality

### Document Support
- Upload `.txt`, `.md`, or `.pdf` files (PDF text extraction via PyMuPDF)
- Paste manual text directly in the sidebar
- Automatic chunking with overlap (1200 chars, 160 char overlap) 
  for long documents

### Structured Responses
- **Key Takeaways** — Bulleted summary of the most important points
- **Full Answer** — Collapsed expander with complete response
- **Source citations** — Expandable view of retrieved manual sections 
  with relevance scores and text offsets
- **Intent-aware formatting** — Responses formatted differently per 
  intent type (steps for Procedure, bullets for Eligibility, 
  covered/not-covered split for Coverage, etc.)

### Conversation History
- Persistent session storage via SQLite
- Load and resume previous conversations from the sidebar
- New conversation button with timestamped session IDs

---

## Tech Stack

- **Python** — Core language
- **Streamlit** — Web UI framework
- **sentence-transformers** — Semantic embeddings (`all-MiniLM-L6-v2`)
- **spaCy** — NLP and named entity recognition (`en_core_web_sm`)
- **PyMuPDF (fitz)** — PDF text extraction
- **OpenAI / Anthropic / Google APIs** — LLM backends
- **python-dotenv** — API key management
- **SQLite** — Persistent conversation storage

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/dahanam/EHR-Manual-QA-Chatbot.git
cd EHR-Manual-QA-Chatbot
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install streamlit sentence-transformers spacy pymupdf \
            python-dotenv openai anthropic google-generativeai
python -m spacy download en_core_web_sm
```

### 4. Add API keys
Create a file at `files/keys.env`:
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

### 5. Run the app

**Full multi-provider version (recommended):**
```bash
streamlit run chatbox2.py
```

**Lightweight single-agent version:**
```bash
streamlit run chat.py
```

---

## File Structure

| File | Description |
|------|-------------|
| `chatbox2.py` | Full app — multi-provider LLM, persistent history, Key Takeaways |
| `chat.py` | Improved single-agent version with intent-aware formatting |
| `chatbox.py` | Original baseline agent (Marcelle's version) |
| `modules/magent.py` | ManualOnlyAgent — core QA logic |
| `modules/llm_registry.py` | LLM provider registry (OpenAI, Claude, Gemini) |
| `modules/functions.py` | spaCy loader, PDF reader utilities |
| `modules/message_db_functions.py` | SQLite conversation persistence |
| `files/keys.env` | API keys (not committed to repo) |

---

## How It Works

1. User uploads or pastes a medical manual
2. The document is chunked and embedded using sentence-transformers
3. User asks a question in the chat interface
4. The system classifies intent and retrieves the top-5 most relevant chunks
5. An LLM (or local retrieval) generates a structured answer with sources
6. Response is formatted based on intent type and displayed with 
   confidence score, key takeaways, and source citations

---

## Authors

Dahana Moz Ruiz & Marcelle Tamegnon — CPC Integrated Health
