# 📜 DocIntel — Document Intelligence System

---

## 🧠 What is DocIntel?

DocIntel is a **Document Intelligence System** that lets you upload your own documents and ask questions about them in plain English. It reads your files, understands their content, and gives you accurate answers — always citing which document the answer came from.

You can upload PDFs, Word files, spreadsheets, presentations, and more. DocIntel processes them locally and uses a powerful AI model to answer your queries.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Interface                       │
│              Streamlit Web App  (app.py)                 │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    RAG Backend (rag.py)                  │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐  ┌─────────────┐  │
│  │ File Loaders │───▶│ Text Chunker │-▶│  FAISS      │  │
│  │ (10 formats) │    │ (LangChain)  │  │  Vector DB  │  │
│  └──────────────┘    └──────────────┘  └──────┬──────┘  │
│                                               │          │
│  ┌──────────────────────────────────────────┐ │          │
│  │  HuggingFace Embeddings                  │◀┘          │
│  │  (Alibaba-NLP/gte-base-en-v1.5)  local   │            │
│  └──────────────────────────────────────────┘            │
│                                                          │
│  ┌──────────────────────────────────────────┐            │
│  │  Groq LLM  (llama-3.3-70b-versatile)     │            │
│  │  Answer generation from retrieved context│            │
│  └──────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```
Upload Files
    │
    ▼
Format-Specific Loaders
    │   PDF → PyPDFLoader
    │   TXT → TextLoader
    │   DOCX → Docx2txtLoader
    │   CSV → Custom row-by-row parser
    │   XLSX/XLS → openpyxl / xlrd parser
    │   PPTX → UnstructuredPowerPointLoader
    │   MD → UnstructuredMarkdownLoader
    │   HTML → UnstructuredHTMLLoader
    ▼
RecursiveCharacterTextSplitter
    │   chunk_size=800, overlap=100
    ▼
HuggingFace Embeddings (local)
    ▼
FAISS Vector Index (saved to disk)
    │
    │   [At query time]
    ▼
similarity_search_with_score (top-12 chunks)
    │
    ▼
Token-budget-aware chunk selection
    │   (fits within LLM context window)
    ▼
Groq LLM → Grounded Answer + Sources
```

---

## 📁 Project Structure

```
DocIntel/
│
├── app.py                  # Streamlit UI — upload, chat, sidebar
├── rag.py                  # RAG backend — indexing, retrieval, generation
├── .env                    # API keys (not committed to version control)
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
└── resource/
    └── faiss_index/        # Persisted FAISS vector index (auto-created)
        ├── index.faiss
        └── index.pkl
```

---

## ⚙️ RAG Configuration

These parameters are tunable in `rag.py`:

| Parameter | Value | Purpose |
|---|---|---|
| `CHUNK_SIZE` | 800 chars | Size of each text chunk |
| `CHUNK_OVERLAP` | 100 chars | Overlap between chunks to preserve context |
| `RETRIEVAL_K` | 12 | Number of top chunks retrieved per query |
| `MAX_ANSWER_TOKENS` | 600 | Max tokens for LLM answer |
| `LLM_CONTEXT_LIMIT` | 4096 | Total context window of the LLM |
| `CONTEXT_BUDGET` | 3246 | Tokens budgeted for document context |
| `EMBEDDING_MODEL` | `gte-base-en-v1.5` | Local HuggingFace embedding model |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq LLM for answer generation |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / UI** | [Streamlit](https://streamlit.io/) |
| **RAG Framework** | [LangChain](https://www.langchain.com/) |
| **Vector Database** | [FAISS](https://github.com/facebookresearch/faiss) (local) |
| **Embeddings** | [HuggingFace](https://huggingface.co/) — `Alibaba-NLP/gte-base-en-v1.5` |
| **LLM** | [Groq](https://groq.com/) — `llama-3.3-70b-versatile` |
| **PDF Parsing** | PyPDF (via LangChain) |
| **Excel Parsing** | openpyxl, xlrd |
| **PPTX / HTML / MD** | Unstructured (via LangChain) |
| **DOCX Parsing** | docx2txt (via LangChain) |
| **Environment** | python-dotenv |

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/snairaadarsh/doc_intel.git
cd docintel
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com).

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## 📦 requirements.txt

```
streamlit
langchain
langchain-community
langchain-text-splitters
langchain-huggingface
langchain-groq
faiss-cpu
sentence-transformers
transformers
torch
huggingface-hub
python-dotenv
pypdf
docx2txt
openpyxl
xlrd
unstructured
```

> **Note:** `torch` and `sentence-transformers` are required for the local embedding model. On CPU-only machines, use `faiss-cpu`. If you have a CUDA GPU, you may swap to `faiss-gpu` for faster indexing.

---

## 📖 How to Use

### Step 1 — Upload Documents
- Open the sidebar on the left
- Click **"Choose files"** and select one or more supported documents (up to 100)
- Click **"⌛ Process files"** to begin ingestion

### Step 2 — Wait for Indexing
- A progress bar tracks each file being processed
- On completion, a summary shows total files indexed and total chunks created
- Indexed files appear in the **🗂️ Indexed Documents** section of the sidebar

### Step 3 — Ask Questions
- Type your question in the chat input at the bottom
- DocIntel retrieves the most relevant document chunks and generates a grounded answer
- Each answer includes:
  - **Sources** — which files the answer came from
  - **Supporting Context** (expandable) — the exact chunks used, with scores and page numbers

### Step 4 — Iterate
- The index persists across sessions — no need to re-upload on restart
- Use **🧹 Clear Chat** to reset the conversation without losing the index
