# Explainable RAG-Based Question Answering System

This project is an **explainable Retrieval-Augmented Generation (RAG) system** built as part of a technical assignment.  
It allows users to upload documents and ask questions, where answers are generated **strictly from the uploaded content**.

The main focus of this project is **clarity, correctness, and explainability**, rather than using complex or hidden abstractions.

---

## 🎯 What This Project Does

This system allows you to:

- Upload **PDF or TXT documents**
- Ask **natural language questions** about those documents
- Receive answers that are **grounded only in the uploaded data**
- View **source references** used to generate each answer

This approach helps reduce hallucinations and improves trust in the generated responses.

---

## 🧠 Why RAG (Retrieval-Augmented Generation)?

Large Language Models can generate incorrect information if they rely only on training data.  
RAG improves reliability by:

1. Retrieving relevant document content first  
2. Providing only that content to the LLM  
3. Generating answers grounded in retrieved context  

This project demonstrates a **clean and transparent RAG pipeline**.

---

## 🛠️ Tech Stack & Rationale

| Component | Technology | Reason |
|--------|------------|--------|
| API | FastAPI | Lightweight, fast, auto-generated docs |
| RAG Framework | LangChain | Standard patterns for RAG pipelines |
| LLM | Groq (LLaMA-3) | Fast inference, free tier |
| Embeddings | HuggingFace MiniLM | Open-source, fast, 384-dim |
| Vector Store | Pinecone | Managed, scalable vector database |
| Config | python-dotenv | Secure environment management |

---

## 📐 High-Level Architecture

```

User
↓
FastAPI API
↓
Document Upload (Background Task)
↓
Text Extraction (PDF / TXT)
↓
Chunking (450 chars, 50 overlap)
↓
Embedding Generation
↓
Pinecone Vector Store

Query Flow:
Question → Embedding → Top-K Retrieval → Context → Groq LLM → Answer

````

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.9+
- Free API keys:
  - Groq
  - Pinecone
  - HuggingFace

---

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate     
````

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.3-70b-versatile

HF_API_KEY=your_key
EMBEDDING_MODEL=all-MiniLM-L6-v2

PINECONE_API_KEY=your_key
PINECONE_ENV=us-east-1
PINECONE_INDEX_NAME=rag-documents-384

TOP_K=3
```

⚠️ Pinecone index dimension must be **384**.

---

### 4️⃣ Run the Server

```bash
python main.py
```

Open:

```
http://localhost:8000/docs
```

---

## 🌐 API Endpoints

* **GET `/health`** – Health check
* **POST `/upload`** – Upload PDF/TXT document
* **POST `/query`** – Ask a question

Responses include:

* Answer
* Source chunks
* Latency

---

## 📁 Project Structure

```
.
├── main.py        # FastAPI app and routes
├── rag.py         # Retrieval & generation logic
├── utils.py       # Chunking and loaders
├── requirements.txt
├── .env.example
├── README.md
└── EXPLANATION.md
```

---

## 🛡️ Safety Features

* Basic rate limiting
* Pydantic request validation
* File type & size validation
* Background ingestion for large files
* Graceful error handling

---

## 📊 Performance Notes

* Typical query latency: **1–3 seconds**
* Chunk size: **450 characters**
* Overlap: **50 characters**
