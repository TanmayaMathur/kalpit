"""
AmbedkarGPT - Main RAG System
Assignment 1: Building a Functional Prototype

This module implements a simple command-line Q&A system using the RAG (Retrieval-Augmented Generation) 
pipeline with LangChain, ChromaDB, HuggingFace Embeddings, and Ollama Mistral 7B.

Components:
1. Document Loading - Load speech files
2. Text Chunking - Split text into manageable chunks
3. Embeddings - Generate embeddings using HuggingFace
4. Vector Store - Store embeddings in ChromaDB
5. Retrieval - Retrieve relevant chunks based on queries
6. Generation - Generate answers using Ollama Mistral 7B

Usage:
    python main.py
    Then enter questions interactively at the command prompt.
"""

import os
import sys
from typing import List, Dict, Tuple
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class AmbedkarGPT:
    """
    RAG System for Question-Answering on Ambedkar's works.
    
    This class orchestrates the entire pipeline:
    - Loading documents from corpus folder
    - Creating embeddings
    - Building vector store
    - Retrieving relevant chunks
    - Generating answers using LLM
    """
    
    def __init__(self, corpus_dir: str = "./corpus", chunk_size: int = 500, 
                 chunk_overlap: int = 50, persist_dir: str = "./chroma_db"):
        """
        Initialize the RAG system.
        
        Args:
            corpus_dir (str): Directory containing speech files
            chunk_size (int): Size of text chunks in characters
            chunk_overlap (int): Overlap between chunks
            persist_dir (str): Directory to persist ChromaDB
        """
        self.corpus_dir = corpus_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_dir = persist_dir
        
        self.documents = []
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        
        print("[*] Initializing AmbedkarGPT...")
        self._initialize_system()
    
    def _load_documents(self) -> List[Document]:
        """
        Load all documents from the corpus directory.
        
        Returns:
            List[Document]: List of loaded documents
        """
        print(f"[*] Loading documents from {self.corpus_dir}...")
        documents = []
        
        corpus_path = Path(self.corpus_dir)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus directory not found: {self.corpus_dir}")
        
        # Load all .txt files from corpus
        for txt_file in sorted(corpus_path.glob("*.txt")):
            print(f"    [+] Loading {txt_file.name}...")
            loader = TextLoader(str(txt_file), encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)
        
        if not documents:
            raise ValueError(f"No documents found in {self.corpus_dir}")
        
        print(f"[+] Loaded {len(documents)} documents")
        return documents
    
    def _split_text(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents (List[Document]): List of documents to split
            
        Returns:
            List[Document]: List of chunked documents
        """
        print(f"[*] Splitting text into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})...")
        
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        chunks = splitter.split_documents(documents)
        print(f"[+] Created {len(chunks)} chunks")
        return chunks
    
    def _initialize_embeddings(self):
        """Initialize HuggingFace embeddings (no API key needed)."""
        print("[*] Initializing HuggingFace embeddings...")
        print("    [+] Using: sentence-transformers/all-MiniLM-L6-v2")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}  # Use CPU for compatibility
        )
        print("[+] Embeddings initialized")
    
    def _initialize_vector_store(self, chunks: List[Document]):
        """
        Initialize ChromaDB vector store.
        
        Args:
            chunks (List[Document]): Chunks to store
        """
        print(f"[*] Initializing ChromaDB vector store (persist_dir={self.persist_dir})...")
        
        # Create vector store from chunks
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name="ambedkar_speeches"
        )
        
        # Persist to disk
        self.vector_store.persist()
        print(f"[+] Vector store created and persisted")
    
    def _initialize_llm(self):
        """Initialize Ollama LLM."""
        print("[*] Initializing Ollama LLM...")
        print("    [+] Model: Mistral 7B")
        print("    [!] Make sure Ollama is running: ollama serve")
        
        self.llm = Ollama(
            model="mistral",
            base_url="http://localhost:11434",
            temperature=0.7,
            top_p=0.9
        )
        print("[+] LLM initialized")
    
    def _initialize_retriever(self):
        """Initialize retriever from vector store."""
        print("[*] Initializing retriever...")
        
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Retrieve top 3 chunks
        )
        print("[+] Retriever initialized")
    
    def _initialize_qa_chain(self):
        """Initialize QA chain with prompt template."""
        print("[*] Initializing QA chain...")
        
        # Create a prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant answering questions about Dr. B.R. Ambedkar's speeches and writings.

Context from the documents:
{context}

Question: {question}

Answer: Provide a comprehensive answer based on the context above. If the context doesn't contain relevant information, say so."""
        )
        
        # Create the chain using modern LangChain approach
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        self.qa_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        print("[+] QA chain initialized")
    
    def _initialize_system(self):
        """Initialize the complete RAG system."""
        try:
            # Step 1: Load documents
            documents = self._load_documents()
            
            # Step 2: Split into chunks
            chunks = self._split_text(documents)
            
            # Step 3: Initialize embeddings
            self._initialize_embeddings()
            
            # Step 4: Create vector store
            self._initialize_vector_store(chunks)
            
            # Step 5: Initialize LLM
            self._initialize_llm()
            
            # Step 6: Initialize retriever
            self._initialize_retriever()
            
            # Step 7: Initialize QA chain
            self._initialize_qa_chain()
            
            print("[+] System initialization complete!\n")
            
        except Exception as e:
            print(f"[!] Error during initialization: {str(e)}")
            print("[!] Make sure:")
            print("    1. Corpus directory exists with speech files")
            print("    2. Ollama is running (ollama serve)")
            print("    3. Mistral model is installed (ollama pull mistral)")
            sys.exit(1)
    
    def query(self, question: str) -> Dict:
        """
        Ask a question and get an answer.
        
        Args:
            question (str): User's question
            
        Returns:
            Dict: Contains 'answer', 'source_documents', and other metadata
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized")
        
        try:
            # Get relevant documents
            docs = self.retriever.invoke(question)
            
            # Generate answer using the chain
            answer = self.qa_chain.invoke(question)
            
            return {
                "result": answer,
                "source_documents": docs
            }
        except Exception as e:
            return {
                "result": f"Error generating answer: {str(e)}",
                "source_documents": [],
                "error": str(e)
            }
    
    def run_interactive(self):
        """Run interactive Q&A session."""
        print("=" * 70)
        print("Welcome to AmbedkarGPT - Q&A System")
        print("=" * 70)
        print("\nYou can now ask questions about Dr. Ambedkar's speeches.")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                question = input("Question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not question:
                    print("Please enter a question.\n")
                    continue
                
                print("\n[*] Searching for relevant information...")
                result = self.query(question)
                
                print("\n" + "=" * 70)
                print("ANSWER:")
                print("=" * 70)
                print(result.get("result", result.get("answer", "No answer generated")))
                
                print("\n" + "=" * 70)
                print("SOURCES:")
                print("=" * 70)
                sources = result.get("source_documents", [])
                if sources:
                    for i, doc in enumerate(sources, 1):
                        source_file = doc.metadata.get("source", "Unknown")
                        print(f"\n[{i}] Source: {source_file}")
                        print(f"    Content: {doc.page_content[:200]}...")
                else:
                    print("No source documents retrieved")
                
                print("\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n[!] Error: {str(e)}\n")


def main():
    """Main entry point."""
    try:
        # Initialize system with default parameters
        rag_system = AmbedkarGPT(
            corpus_dir="./corpus",
            chunk_size=500,
            chunk_overlap=50,
            persist_dir="./chroma_db"
        )
        
        # Run interactive session
        rag_system.run_interactive()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
