"""
Utilities for document loading and text chunking.
Using LangChain components for industry-standard document processing.
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document(file_path: str):
    """
    Load a document using appropriate LangChain loader based on file extension.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List of LangChain Document objects
    """
    # determine file type and use appropriate loader
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        # use pypdf loader for pdf files
        loader = PyPDFLoader(file_path)
    elif file_extension == '.txt':
        # simple text loader for txt files
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    # load returns a list of Document objects
    documents = loader.load()
    return documents


def get_text_splitter():
    """
    Configure the text splitter for chunking documents.
    
    Using RecursiveCharacterTextSplitter because it intelligently splits on
    paragraph boundaries first, then sentences, then characters.
    This preserves semantic meaning better than simple character splitting.
    
    Chunk size 450 chars is a sweet spot:
    - Large enough to maintain context for questions
    - Small enough for precise retrieval (avoid noisy chunks)
    - Overlap of 50 chars helps preserve sentence boundaries across chunks
    
    Returns:
        Configured RecursiveCharacterTextSplitter
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # try these separators in order
    )
