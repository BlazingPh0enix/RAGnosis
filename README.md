# DocuLens - Multi-Modal RAG System

A production-ready Multi-Modal Retrieval-Augmented Generation (RAG) system for PDF documents with text, tables, and images.

## Features

- 📄 **PDF Parsing** - Uses LlamaParse for accurate document parsing
- 🖼️ **Image Understanding** - Extracts and summarizes images with GPT-4
- 🔍 **Semantic Search** - Sentence Transformers embeddings with Qdrant vector store
- 💬 **Conversational AI** - Chat with your documents using GPT
- 📊 **Source Citations** - View the exact sources for each response
- 🚀 **FastAPI Backend** - Async document processing with real-time status

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                         │
│  (file upload, chat interface, source inspector)        │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP calls
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                       │
│  /api/documents/upload  │  /api/query                   │
│  /api/documents/status  │  /api/collections             │
└─────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   ┌─────────┐    ┌───────────┐    ┌───────────┐
   │ Qdrant  │    │ LlamaParse│    │  OpenAI   │
   │  Vector │    │    API    │    │    API    │
   │   DB    │    │           │    │           │
   └─────────┘    └───────────┘    └───────────┘
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Docker (for Qdrant)
- OpenAI API key
- LlamaCloud API key

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
LLAMA_CLOUD_API_KEY=your_llamacloud_api_key
QDRANT_URL=http://localhost:6333
```

### 4. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 5. Run the Application

**Option A: Run both servers (recommended)**
```bash
python run.py
```

**Option B: Run separately**
```bash
# Terminal 1 - Start FastAPI backend
uvicorn api.main:app --reload --port 8000

# Terminal 2 - Start Streamlit frontend
streamlit run app/streamlit_app.py
```

### 6. Open the App

- **Streamlit UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

## Usage

1. **Upload a PDF** - Click "Choose a PDF file" in the sidebar
2. **Wait for Processing** - Watch the progress as DocuLens:
   - Parses the document with LlamaParse
   - Extracts and summarizes images
   - Generates embeddings
   - Indexes to Qdrant
3. **Chat with Your Document** - Ask questions in natural language
4. **View Sources** - See which pages and images were used to answer

## Project Structure

```
doculens/
├── api/                    # FastAPI backend
│   ├── main.py            # App entry point
│   ├── schemas.py         # Pydantic models
│   ├── routes/            # API endpoints
│   │   ├── documents.py   # Upload, status, cancel
│   │   └── query.py       # Query endpoint
│   └── services/          # Business logic
│       └── processor.py   # Document processing
├── app/                    # Streamlit frontend
│   ├── streamlit_app.py   # Main app
│   └── components/        # UI components
├── config/                # Configuration
├── ingestion/             # Document parsing
├── index/                 # Chunking & indexing
├── retrieval/             # Query engine
├── evaluation/            # RAGAS metrics
└── run.py                 # Run script
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `LLAMA_CLOUD_API_KEY` | LlamaCloud API key | Required |
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `API_URL` | FastAPI server URL | `http://localhost:8000` |
| `MAX_UPLOAD_SIZE_MB` | Max upload size | `50` |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/documents/upload` | POST | Upload PDF for processing |
| `/api/documents/{job_id}/status` | GET | Get processing status |
| `/api/documents/{job_id}/cancel` | POST | Cancel processing |
| `/api/documents/collections` | GET | List Qdrant collections |
| `/api/query` | POST | Query documents |
| `/health` | GET | Health check |

## Models Used

- **Embeddings**: `all-MiniLM-L6-v2` (384 dimensions)
- **LLM**: `gpt-5-nano-2025-08-07`
- **Reranker**: `BAAI/bge-reranker-base`
- **Parser**: LlamaParse

## License

MIT
