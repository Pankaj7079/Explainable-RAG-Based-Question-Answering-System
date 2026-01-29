"""
RAG (Retrieval-Augmented Generation) pipeline using LangChain.
Handles document ingestion, vector storage, retrieval, and answer generation.
"""

import os
from typing import Dict, List
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

from utils import load_document, get_text_splitter

# load environment variables
load_dotenv()


def get_embeddings():
    """
    Initialize HuggingFace embeddings model.
    Using all-MiniLM-L6-v2 because it's:
    - Fast (good for real-time APIs)
    - Lightweight (384 dimensions)
    - Open source (no vendor lock-in)
    - Well-tested for semantic search
    """
    return HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        model_kwargs={'device': 'cpu'}  # can switch to 'cuda' if GPU available
    )


def initialize_pinecone():
    """
    Initialize Pinecone client and ensure index exists.
    This is called once at startup to verify configuration.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-documents")
    
    # create pinecone client
    pc = Pinecone(api_key=api_key)
    
    # check if index exists, create if it doesn't
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        # create index with appropriate dimension for all-MiniLM-L6-v2 (384)
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=os.getenv("PINECONE_ENV", "us-east-1")
            )
        )
        print(f"Created new Pinecone index: {index_name}")
    
    return pc


def get_vector_store():
    """
    Get Pinecone vector store instance using LangChain integration.
    This handles embedding storage and retrieval automatically.
    """
    embeddings = get_embeddings()
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-documents")
    
    # langchain's pinecone wrapper handles all vector operations
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )
    
    return vector_store


def ingest_document(file_path: str, filename: str):
    """
    Complete document ingestion pipeline.
    
    Steps:
    1. Load document using appropriate loader
    2. Split into chunks with RecursiveCharacterTextSplitter
    3. Add metadata (source filename, chunk index)
    4. Store in Pinecone vector store
    
    Args:
        file_path: Path to the document file
        filename: Original filename for metadata
    """
    # step 1: load document
    documents = load_document(file_path)
    
    # step 2: split into chunks
    text_splitter = get_text_splitter()
    chunks = text_splitter.split_documents(documents)
    
    # step 3: add metadata to each chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = filename
        chunk.metadata["chunk_index"] = i
    
    # step 4: store in pinecone
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    
    print(f"Ingested {len(chunks)} chunks from {filename}")


def get_qa_chain():
    """
    Create a RAG chain for question answering using LangChain LCEL.
    
    This chain combines:
    - Retriever: searches Pinecone for relevant chunks
    - LLM: Groq's llama3-8b-8192 for fast, quality answers
    - Prompt: custom template to ensure grounded answers
    """
    # initialize groq llm
    # using llama-3.3-70b-versatile for fast inference and high quality answers
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0  # deterministic answers for consistency
    )
    
    # get vector store and configure as retriever
    vector_store = get_vector_store()
    top_k = int(os.getenv("TOP_K", 3))
    retriever = vector_store.as_retriever(
        search_kwargs={"k": top_k}  # retrieve top-k most relevant chunks
    )
    
    # custom prompt to keep answers grounded in retrieved context
    prompt_template = """You are a helpful assistant answering questions based solely on the provided context.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer the question using ONLY the information from the context above
- If the context doesn't contain enough information, say "I don't have enough information in the uploaded documents to answer this question"
- Be concise and specific
- Cite relevant parts of the context when possible

Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # helper function to format retrieved documents
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    # create rag chain using LCEL (LangChain Expression Language)
    # this is the modern way to build chains in LangChain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever



def query_documents(question: str) -> Dict:
    """
    Query the RAG system with a question.
    
    Args:
        question: User's question
        
    Returns:
        Dictionary with:
        - answer: Generated answer
        - sources: List of source documents with metadata
    """
    rag_chain, retriever = get_qa_chain()
    
    # execute the chain to get the answer
    answer = rag_chain.invoke(question)
    
    # get source documents separately
    source_docs = retriever.invoke(question)
    
    # extract answer and sources
    response = {
        "answer": answer,
        "sources": [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "chunk_index": doc.metadata.get("chunk_index", -1)
            }
            for doc in source_docs
        ]
    }
    
    return response

