# Domain-Specific-Document-Intelligence-QA-Engine-
A specialized RAG-powered document intelligence engine for accurate semantic search, automated information extraction, and grounded Q&amp;A over domain-specific files.


# RAG Chatbot — PDF/TXT Question Answering with LangChain, ChromaDB & Groq

# Chatbot Live link: [ Optimizing the full backend process. Take some time ]


A learning project that builds a Retrieval-Augmented Generation (RAG) pipeline. It loads documents, splits their text into chunks, converts the chunks into vector embeddings, stores them in ChromaDB, retrieves relevant content for a user question, and passes that context to a Groq-hosted language model to generate an answer.

> **Project status:** This repository contains notebook-based experimentation and implementation. Review the notebooks and configure local paths/API credentials before running. It is not presented as a production-ready service.

## Contents

- [Overview](#overview)
- [RAG architecture](#rag-architecture)
- [Pipeline walkthrough](#pipeline-walkthrough)
- [Technology stack](#technology-stack)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Running the notebooks](#running-the-notebooks)
- [How retrieval and generation work](#how-retrieval-and-generation-work)
- [Configuration notes](#configuration-notes)
- [Current limitations and next steps](#current-limitations-and-next-steps)

## Overview

The project explores two parts of a RAG system:

1. **Document ingestion:** creating LangChain `Document` objects and loading TXT/PDF files, including loading multiple files from a directory.
2. **PDF-to-answer pipeline:** loading PDF pages, splitting content into smaller chunks, generating embeddings with Sentence Transformers, persisting vectors in ChromaDB, retrieving relevant chunks, and generating a response with a Groq LLM.

The notebooks include a simple RAG function and an enhanced function that is designed to return answer/source information and an optional context preview.

## RAG architecture

![RAG chatbot architecture](assets/rag_architecture.png)

The workflow has two broad phases:

- **Indexing:** load source documents → split into chunks → create embeddings → store vectors and metadata.
- **Question answering:** embed a question → retrieve similar chunks → build context → ask the LLM to answer using that context.

## Pipeline walkthrough

### 1. Document ingestion

The first notebook demonstrates:

- LangChain `Document` objects with `page_content` and `metadata`.
- `TextLoader` for loading a single TXT file.
- `DirectoryLoader` for loading TXT files from a directory.
- `PyMuPDFLoader` / `PyPDFLoader` for loading PDF documents.

The metadata can preserve information such as the source file and page number, which is useful when tracing retrieved content.

### 2. PDF loading and text splitting

The second notebook loads a PDF with `PyMuPDFLoader` and uses `RecursiveCharacterTextSplitter` to split documents into smaller pieces.

The example call uses:

```python
chunks = split_documents(
    all_pdf_documents,
    chunk_size=500,
    chunk_overlap=150
)
```

- **Chunk size:** limits the size of each text segment.
- **Chunk overlap:** repeats a portion of neighboring text to help preserve context across chunk boundaries.

![RAG query workflow](assets/rag_workflow.png)

### 3. Embedding generation

The `EmbeddingManager` class loads the Sentence Transformers model:

```python
all-MiniLM-L6-v2
```

It converts each chunk into a numerical vector representation. The same embedding model is also used to encode a user query before retrieval.

### 4. Vector storage with ChromaDB

The `VectorStore` class creates a persistent ChromaDB client and a collection named `pdf_documents` by default.

For each chunk, the notebook stores:

- A generated document ID.
- The embedding vector.
- The original chunk text.
- Metadata, including the document index and content length.

### 5. Similarity-based retrieval

The `RAGRetriever` class:

1. Embeds the user's question.
2. Queries the ChromaDB collection for the nearest vectors.
3. Reads the returned text, metadata, IDs, and distances.
4. Converts distance to a similarity-style value using `1 - distance`.
5. Optionally filters results using a score threshold.

The retrieval method supports `top_k` to control how many results are requested.

### 6. LLM response generation

The notebook integrates `ChatGroq` and uses the retrieved text as context for the language model.

- `rag_simple()` retrieves context and asks the model to answer the question.
- `rag_advanced()` is intended to return an answer, source details, a confidence value based on the highest returned similarity score, and optionally the combined context.

The confidence value in this notebook is a retrieval-similarity heuristic; it is **not** a calibrated probability that the answer is correct.

## Technology stack

| Component | Technology | Purpose |
|---|---|---|
| Language | Python | Main implementation |
| Notebook environment | Jupyter Notebook | Experimentation and execution |
| Document handling | LangChain `Document` | Text and metadata representation |
| Document loading | LangChain Community loaders | TXT and PDF ingestion |
| PDF parsing | PyMuPDF / PyPDF | PDF text extraction |
| Text splitting | RecursiveCharacterTextSplitter | Chunk creation |
| Embeddings | Sentence Transformers | Text/query vector generation |
| Embedding model | `all-MiniLM-L6-v2` | Dense text embeddings |
| Vector database | ChromaDB | Persistent vector storage and search |
| LLM integration | `langchain-groq` | Calling a Groq-hosted model |
| LLM shown in notebook | `qwen/qwen3.8-27b` | Response generation |

## Repository structure

Arrange the files in your GitHub repository like this:

```text
RAG-Chatbot/
│
├── 01_RAG_chatbot.ipynb
├── 02_pdf_loader.ipynb
├── README.md
│
├── assets/
│   ├── rag_architecture.png
│   └── rag_workflow.png
│
├── Data/                  # Optional: keep private/source PDFs out of Git
│   ├── pdf/
│   └── text_files/
│
├── chroma_db/             # Generated vector database; normally gitignored
│
└── .env                   # Local secrets; never commit this file
```

The notebook filenames in your uploaded files are `01_RAG_chatbot(1).ipynb` and `02_pdf_loader.ipynb`. You can rename the first notebook to `01_RAG_chatbot.ipynb` when arranging the repository, or keep its existing filename and update the links below.

### Notebooks

- [01 — RAG chatbot and document ingestion](01_RAG_chatbot.ipynb)
- [02 — PDF loader, chunking, embeddings, vector store and retrieval](02_pdf_loader.ipynb)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository-name>.git
cd <your-repository-name>
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

The notebooks use the following main packages:

```bash
pip install jupyter ipykernel
pip install langchain langchain-community langchain-text-splitters
pip install sentence-transformers chromadb
pip install pymupdf pypdf
pip install langchain-groq python-dotenv
pip install numpy scikit-learn
```

Depending on your environment and package versions, you may need to resolve compatibility issues. For reproducible runs, pin the versions that work in your environment in a `requirements.txt` file.

### 4. Configure the Groq API key

Create a `.env` file in the project root:

```env
GROQ_APIS_KEY=your_groq_api_key_here
```

The current notebook reads the environment variable named `GROQ_APIS_KEY`:

```python
load_dotenv()
api_key = os.getenv("GROQ_APIS_KEY")
```

Keep `.env` private. Add it to `.gitignore` and never upload API keys to GitHub. If you change the variable name in the notebook, make sure the `.env` key matches it.

### 5. Configure document and vector-store paths

The notebooks currently contain Windows-specific absolute paths (for example, paths under `D:\RAG Project\...` and `C:\ML Study\...`). Update these paths to match your machine before running the cells.

For portability, consider using `pathlib.Path` and paths relative to the repository root.

## Running the notebooks

Start Jupyter:

```bash
jupyter notebook
```

Recommended order:

1. Open `01_RAG_chatbot.ipynb` and run the document-ingestion examples.
2. Open `02_pdf_loader.ipynb`.
3. Set the PDF input path and persistent ChromaDB directory.
4. Run the PDF loading and text-splitting cells.
5. Initialize the embedding manager and vector store.
6. Generate embeddings and add chunks to the ChromaDB collection.
7. Initialize the retriever and test a query.
8. Configure the Groq API key, initialize the LLM, and run the RAG functions.

Example retrieval test shown in the notebook:

```python
rag_retriever.retrieve(
    "Explain The Term Variance Inflation Factor Mean?",
    top_k=5
)
```

Example RAG question shown in the notebook:

```python
question = "svm"
answer = rag_simple(question, rag_retriever, llm)
print(answer)
```

## How retrieval and generation work

Conceptually, the question-answering path is:

```text
User question
     |
     v
SentenceTransformer query embedding
     |
     v
ChromaDB similarity search
     |
     v
Top-K retrieved document chunks
     |
     v
Context + question prompt
     |
     v
Groq-hosted Qwen model
     |
     v
Generated answer
```

The answer depends on the quality of the source documents, chunking strategy, embedding model, retrieval results, prompt, and LLM.

## Configuration notes

Values used in the notebook include:

| Setting | Example value |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` |
| ChromaDB collection | `pdf_documents` |
| Text chunk size in example | `500` |
| Text chunk overlap in example | `150` |
| Simple RAG default `top_k` | `3` |
| Advanced RAG default `top_k` | `5` |
| Advanced RAG default minimum score | `0.2` |
| Groq model string | `qwen/qwen3.8-27b` |
| LLM temperature | `0.1` |
| LLM max tokens | `1024` |

These are notebook example values, not benchmark-optimized settings.

## Current limitations and next steps

This is a notebook-based project and some parts need refinement before it is used as a reliable application:

- Replace machine-specific absolute paths with configurable relative paths.
- Review and test the `process_all_pdfs()` function; its current implementation has path/document handling issues and should be corrected before relying on directory-wide PDF ingestion.
- Check the prompt construction in `rag_advanced()`; the current prompt references `answer` before a response has been generated.
- Add robust handling for empty retrieval results, API failures, and malformed PDFs.
- Persist and validate source metadata consistently (for example, source filename and page number).
- Add a tested `requirements.txt` with pinned dependency versions.
- Add evaluation questions and measure retrieval quality and answer faithfulness.
- Consider adding a user interface (for example, Streamlit or FastAPI) after the pipeline is stable.
- Add tests for document loading, chunking, vector insertion, retrieval, and answer generation.

## Learning outcomes

Through this project, the implementation explores:

- Document loading with LangChain.
- Text chunking and overlap.
- Semantic embeddings with Sentence Transformers.
- Persistent vector storage using ChromaDB.
- Similarity-based document retrieval.
- Context injection into an LLM prompt.
- Basic and enhanced RAG pipeline design.

---

**If you find this project useful, feel free to star the repository.**



## Contacts

- Name: Akash Mondal
- email: skyakash.ml@gmail.com
- Contact No. : +91 9083666706
