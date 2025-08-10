import os
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import hashlib
from datetime import datetime
import torch
import time
import psutil
import functools
from contextlib import contextmanager
import threading
from collections import defaultdict
import gc
import weakref

# Direct library imports (no LangChain)
try:
    import fitz  # PyMuPDF - much faster alternative
    PYMUPDF_AVAILABLE = True
except ImportError:
    import PyPDF2
    import pdfplumber
    PYMUPDF_AVAILABLE = False

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import ollama
from dataclasses import dataclass
from typing import Any
from dotenv import load_dotenv
import pandas as pd
from enum import Enum
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolType(Enum):
    RETRIEVAL = "retrieval"
    ANALYSIS = "analysis"
    SEARCH = "search"
    SYNTHESIS = "synthesis"

@dataclass
class ToolResult:
    tool_name: str
    success: bool
    result: Any
    metadata: Dict[str, Any]
    execution_time: float

@dataclass
class QueryPlan:
    original_query: str
    sub_queries: List[str]
    tools_sequence: List[Dict[str, Any]]
    reasoning: str

class AgentTool:
    def __init__(self, name: str, description: str, tool_type: ToolType):
        self.name = name
        self.description = description
        self.tool_type = tool_type
    
    def execute(self, query: str, context: Dict[str, Any], rag_system) -> ToolResult:
        raise NotImplementedError("Subclasses must implement execute method")

class ProfilerStats:
    """Thread-safe profiler statistics collector"""
    
    def __init__(self):
        self.stats = defaultdict(lambda: {
            'calls': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'total_memory_mb': 0.0,
            'avg_memory_mb': 0.0,
            'peak_memory_mb': 0.0
        })
        self._lock = threading.Lock()
    
    def record(self, name: str, elapsed_time: float, memory_mb: float):
        with self._lock:
            stats = self.stats[name]
            stats['calls'] += 1
            stats['total_time'] += elapsed_time
            stats['avg_time'] = stats['total_time'] / stats['calls']
            stats['min_time'] = min(stats['min_time'], elapsed_time)
            stats['max_time'] = max(stats['max_time'], elapsed_time)
            stats['total_memory_mb'] += memory_mb
            stats['avg_memory_mb'] = stats['total_memory_mb'] / stats['calls']
            stats['peak_memory_mb'] = max(stats['peak_memory_mb'], memory_mb)
    
    def get_report(self) -> Dict:
        with self._lock:
            return dict(self.stats)
    
    def reset(self):
        with self._lock:
            self.stats.clear()

# Global profiler instance
PROFILER = ProfilerStats()

def get_memory_usage_mb() -> float:
    """Get current memory usage in MB"""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except Exception:
        return 0.0

def profile_function(name: Optional[str] = None):
    """Decorator to profile function execution time and memory usage"""
    def decorator(func):
        func_name = name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if instance has profiling enabled (for methods)
            enabled = True
            if args and hasattr(args[0], 'enable_profiling'):
                enabled = args[0].enable_profiling
            
            if not enabled:
                return func(*args, **kwargs)
            
            memory_start = get_memory_usage_mb()
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                memory_end = get_memory_usage_mb()
                elapsed_time = end_time - start_time
                memory_used = memory_end - memory_start
                
                PROFILER.record(func_name, elapsed_time, memory_used)
                
                if elapsed_time > 1.0:  # Log slow operations
                    logger.info(f"🕒 {func_name}: {elapsed_time:.2f}s, {memory_used:+.1f}MB")
        
        return wrapper
    return decorator

@contextmanager
def profile_context(name: str, enabled: bool = True):
    """Context manager for profiling code blocks"""
    if not enabled:
        yield
        return
    
    memory_start = get_memory_usage_mb()
    start_time = time.perf_counter()
    
    try:
        yield
    finally:
        end_time = time.perf_counter()
        memory_end = get_memory_usage_mb()
        elapsed_time = end_time - start_time
        memory_used = memory_end - memory_start
        
        PROFILER.record(name, elapsed_time, memory_used)
        
        if elapsed_time > 1.0:  # Log slow operations
            logger.info(f"🕒 {name}: {elapsed_time:.2f}s, {memory_used:+.1f}MB")
        
        # Clean up after profiling
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@contextmanager
def memory_managed_pdf_processing(pdf_path: str):
    """Context manager for memory-efficient PDF processing"""
    doc = None
    try:
        if PYMUPDF_AVAILABLE:
            doc = fitz.open(pdf_path)
        yield doc
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        raise
    finally:
        if doc is not None:
            doc.close()
            del doc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def print_profiling_report():
    """Print a detailed profiling report"""
    stats = PROFILER.get_report()
    
    if not stats:
        print("📊 No profiling data available")
        return
    
    print("\n" + "="*80)
    print("📊 PERFORMANCE PROFILING REPORT")
    print("="*80)
    
    # Sort by total time
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
    
    print(f"{'Operation':<40} {'Calls':<8} {'Total(s)':<10} {'Avg(s)':<10} {'Min(s)':<10} {'Max(s)':<10} {'Mem(MB)':<10}")
    print("-" * 108)
    
    for name, data in sorted_stats:
        # Shorten long names
        display_name = name[-38:] if len(name) > 38 else name
        print(f"{display_name:<40} {data['calls']:<8} {data['total_time']:<10.2f} {data['avg_time']:<10.3f} "
              f"{data['min_time']:<10.3f} {data['max_time']:<10.2f} {data['avg_memory_mb']:<10.1f}")
    
    print("="*80)
    
    # Show top time consumers
    print(f"\n🔥 TOP TIME CONSUMERS:")
    for name, data in sorted_stats[:5]:
        percentage = (data['total_time'] / sum(s['total_time'] for s in stats.values())) * 100
        print(f"   {percentage:5.1f}% - {name.split('.')[-1]}")
    
    print()

def reset_profiling():
    """Reset all profiling statistics"""
    PROFILER.reset()
    logger.info("📊 Profiling statistics reset")

@dataclass
class Document:
    """Simple document class to replace LangChain Document"""
    page_content: str
    metadata: Dict[str, Any]

class EnhancedPDFProcessor:
    """Enhanced PDF processor with table detection and smart chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Simple text splitter without LangChain
        self.separators = ["\n\n", "\n", ". ", " ", ""]
        self.use_pymupdf = PYMUPDF_AVAILABLE
        if self.use_pymupdf:
            logger.info("Using PyMuPDF for faster PDF processing")
        else:
            logger.info("PyMuPDF not available, using PyPDF2 + pdfplumber")
    
    @profile_function("EnhancedPDFProcessor.extract_tables_from_pdf")
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract tables from PDF using PyMuPDF or fallback to pdfplumber"""
        if self.use_pymupdf:
            return self._extract_tables_pymupdf(pdf_path)
        else:
            return self._extract_tables_pdfplumber(pdf_path)
    
    def _extract_tables_pymupdf(self, pdf_path: str) -> List[Dict]:
        """Fast table extraction using PyMuPDF with proper resource cleanup"""
        tables_data = []
        doc = None
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Find tables using PyMuPDF's built-in table detection
                try:
                    tables = page.find_tables()
                    
                    for table_idx, table in enumerate(tables):
                        # Extract table data
                        table_data = table.extract()
                        
                        if table_data and len(table_data) > 1:
                            # Convert to DataFrame for processing
                            df = pd.DataFrame(table_data[1:], columns=table_data[0] if table_data[0] else None)
                            df = df.dropna(how='all').reset_index(drop=True)
                            
                            if not df.empty:
                                tables_data.append({
                                    'page': page_num + 1,  # Convert to 1-based
                                    'table_index': table_idx,
                                    'content': df.to_string(),
                                    'raw_data': df.to_dict('records'),
                                    'num_rows': len(df),
                                    'num_cols': len(df.columns)
                                })
                            
                            # Clear DataFrame to free memory
                            del df
                        
                        # Clear table data to free memory
                        del table_data
                        
                except Exception as e:
                    logger.warning(f"Error extracting tables from page {page_num + 1}: {e}")
                
                # Clear page reference
                del page
        
        except Exception as e:
            logger.warning(f"Error opening PDF {pdf_path}: {e}")
        
        finally:
            # Ensure document is always closed
            if doc is not None:
                doc.close()
                del doc
            # Force garbage collection
            gc.collect()
        
        return tables_data
    
    def _extract_tables_pdfplumber(self, pdf_path: str) -> List[Dict]:
        """Fallback table extraction using pdfplumber"""
        tables_data = []
        
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                # Process pages sequentially
                for page_num in range(1, total_pages + 1):
                    try:
                        page_tables = self._extract_tables_from_page(pdf_path, page_num)
                        tables_data.extend(page_tables)
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num} in {pdf_path}: {e}")
                        
        except Exception as e:
            logger.warning(f"Error extracting tables from {pdf_path}: {e}")
        
        return tables_data
    
    def _extract_tables_from_page(self, pdf_path: str, page_num: int) -> List[Dict]:
        """Extract tables from a single page"""
        tables_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num <= len(pdf.pages):
                    page = pdf.pages[page_num - 1]  # Convert to 0-based index
                    tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 1:  # Ensure table has content
                            # Convert to pandas DataFrame for better processing
                            df = pd.DataFrame(table[1:], columns=table[0] if table[0] else None)
                            
                            # Clean up the dataframe
                            df = df.dropna(how='all').reset_index(drop=True)
                            
                            if not df.empty:
                                tables_data.append({
                                    'page': page_num,
                                    'table_index': table_idx,
                                    'content': df.to_string(),
                                    'raw_data': df.to_dict('records'),
                                    'num_rows': len(df),
                                    'num_cols': len(df.columns)
                                })
        except Exception as e:
            logger.warning(f"Error extracting tables from page {page_num} of {pdf_path}: {e}")
        
        return tables_data
    
    @profile_function("EnhancedPDFProcessor.process_pdf_with_tables")
    def process_pdf_with_tables(self, pdf_path: str) -> Tuple[List[Document], List[Document]]:
        """Process PDF to extract both text chunks and tables separately"""
        
        # Process text and tables sequentially
        text_chunks = self._process_text_pages(pdf_path)
        tables = self.extract_tables_from_pdf(pdf_path)
        
        # Create table documents
        table_documents = []
        for table in tables:
            doc = Document(
                page_content=f"Table from page {table['page']}:\n{table['content']}",
                metadata={
                    'source': pdf_path,
                    'paper_name': os.path.basename(pdf_path),
                    'source_type': 'table',
                    'page': table['page'],
                    'table_index': table['table_index'],
                    'num_rows': table['num_rows'],
                    'num_cols': table['num_cols']
                }
            )
            table_documents.append(doc)
        
        return text_chunks, table_documents
    
    @profile_function("EnhancedPDFProcessor._process_text_pages")
    def _process_text_pages(self, pdf_path: str) -> List[Document]:
        """Process text pages with chunking using PyMuPDF or fallback"""
        if self.use_pymupdf:
            return self._process_text_pages_pymupdf(pdf_path)
        else:
            return self._process_text_pages_pypdf2(pdf_path)
    
    def _process_text_pages_pymupdf(self, pdf_path: str) -> List[Document]:
        """Fast text processing using PyMuPDF with proper resource cleanup"""
        all_chunks = []
        doc = None
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    # Create document
                    doc_obj = Document(
                        page_content=text,
                        metadata={
                            'source': pdf_path,
                            'paper_name': os.path.basename(pdf_path),
                            'source_type': 'text',
                            'page': page_num + 1  # Convert to 1-based
                        }
                    )
                    # Chunk the page
                    chunks = self._chunk_page(doc_obj)
                    all_chunks.extend(chunks)
                    
                    # Clear intermediate objects
                    del doc_obj, chunks
                
                # Clear page reference and text
                del page, text
        
        except Exception as e:
            logger.error(f"Error processing {pdf_path} with PyMuPDF: {e}")
        
        finally:
            # Ensure document is always closed
            if doc is not None:
                doc.close()
                del doc
            # Force garbage collection
            gc.collect()
        
        return all_chunks
    
    def _process_text_pages_pypdf2(self, pdf_path: str) -> List[Document]:
        """Fallback text processing using PyPDF2"""
        all_chunks = []
        
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    
                    if text.strip():
                        # Create document
                        doc = Document(
                            page_content=text,
                            metadata={
                                'source': pdf_path,
                                'paper_name': os.path.basename(pdf_path),
                                'source_type': 'text',
                                'page': page_num
                            }
                        )
                        # Chunk the page
                        chunks = self._chunk_page(doc)
                        all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Error processing {pdf_path} with PyPDF2: {e}")
        
        return all_chunks
    
    def _chunk_page(self, page: Document) -> List[Document]:
        """Chunk a single page"""
        try:
            text = page.page_content
            chunks = self._split_text(text)
            
            # Create Document objects for each chunk
            chunk_docs = []
            for chunk in chunks:
                chunk_doc = Document(
                    page_content=chunk,
                    metadata=page.metadata.copy()
                )
                chunk_docs.append(chunk_doc)
            
            return chunk_docs
        except Exception as e:
            logger.warning(f"Error chunking page: {e}")
            return []
    
    def _split_text(self, text: str) -> List[str]:
        """Simple text splitter"""
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

class LiteratureRAG:
    def __init__(
        self,
        pdf_directory: str = "./pdfs",
        db_directory: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        enable_profiling: bool = True,
    ):
        self.pdf_directory = pdf_directory
        self.db_directory = db_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_profiling = enable_profiling
        
        load_dotenv()
        
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.pdf_processor = EnhancedPDFProcessor(chunk_size, chunk_overlap)
        
        # Collections for different content types
        self.text_collection_name = "literature_text"
        self.table_collection_name = "literature_tables"
        
        self._initialize_components()
    
    @profile_function("LiteratureRAG._initialize_components")
    def _initialize_components(self):
        logger.info("Initializing RAG components...")
        
        # Always use local models only (free)
        logger.info("Using local models (SentenceTransformers + Ollama)")
        
        # Auto-detect best device (GPU if available, otherwise CPU)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Configure model to use all available CPU cores for native multithreading
        import os
        num_threads = os.cpu_count()
        
        # Set PyTorch thread count for native multithreading
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        
        logger.info(f"Configured PyTorch to use {num_threads} threads for native multithreading")
        
        # Use SentenceTransformer with ONNX backend for native CPU multithreading
        try:
            # Try ModernBERT first for better performance
            self.embedding_model = SentenceTransformer(
                "answerdotai/ModernBERT-base",
                device=device,
                backend="onnx"  # Use ONNX for better CPU multithreading
            )
            logger.info("Using ModernBERT-base with ONNX backend")
        except Exception as e:
            logger.error(f"ModernBERT not available: {e}")
            raise
        # Use Ollama client directly
        self.llm_client = ollama.Client()
        self.llm_model = "qwen3:latest"
        
        os.makedirs(self.db_directory, exist_ok=True)
        os.makedirs(self.pdf_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=self.db_directory,
            settings=Settings(anonymized_telemetry=False)
        )
    
    @profile_function("LiteratureRAG._embed_texts")
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts with memory-efficient batch processing"""
        # Reduced batch sizes to prevent memory accumulation
        max_chunk_size = 50   # Process at most 50 texts at a time (reduced from 100)
        optimal_batch_size = 8  # Smaller batch size for better memory management (reduced from 16)
        
        all_embeddings = []
        
        for i in range(0, len(texts), max_chunk_size):
            chunk_texts = texts[i:i + max_chunk_size]
            chunk_batch_size = min(len(chunk_texts), optimal_batch_size)
            
            try:
                embeddings = self.embedding_model.encode(
                    chunk_texts,
                    batch_size=chunk_batch_size,
                    show_progress_bar=len(texts) > 100,
                    normalize_embeddings=True,
                    convert_to_tensor=False
                )
                all_embeddings.extend(embeddings.tolist())
                
                # Clear embedding tensors from memory
                del embeddings
                
                # Force garbage collection and clear GPU cache after each chunk
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error processing embedding chunk {i//max_chunk_size + 1}: {e}")
                # Clean up on error
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Log progress for large datasets
            if len(texts) > 1000:
                logger.info(f"Processed embeddings for {i + len(chunk_texts)}/{len(texts)} texts")
        
        return all_embeddings
    
    @profile_function("LiteratureRAG._process_single_pdf")
    def _process_single_pdf(self, pdf_file_path: str) -> Tuple[List[Document], List[Document]]:
        """Process a single PDF file - helper function for multiprocessing"""
        pdf_file = Path(pdf_file_path)
        logger.info(f"Processing {pdf_file.name}...")
        
        # Create a new processor instance for each process
        processor = EnhancedPDFProcessor(self.chunk_size, self.chunk_overlap)
        text_chunks, table_docs = processor.process_pdf_with_tables(str(pdf_file))
        
        # Add document hash for tracking
        doc_hash = hashlib.md5(pdf_file.name.encode()).hexdigest()[:8]
        for chunk in text_chunks:
            chunk.metadata['doc_id'] = doc_hash
            chunk.metadata['indexed_at'] = datetime.now().isoformat()
        
        for table_doc in table_docs:
            table_doc.metadata['doc_id'] = doc_hash
            table_doc.metadata['indexed_at'] = datetime.now().isoformat()
        
        return text_chunks, table_docs

    @profile_function("LiteratureRAG.load_and_process_documents")
    def load_and_process_documents(self) -> Tuple[List[Document], List[Document]]:
        """Load PDFs and process them sequentially with internal multithreading for each file"""
        logger.info(f"Loading and processing PDFs from {self.pdf_directory}...")
        
        if not os.path.exists(self.pdf_directory):
            logger.warning(f"PDF directory {self.pdf_directory} does not exist!")
            return [], []
        
        pdf_files = list(Path(self.pdf_directory).glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_directory}")
            return [], []
        
        all_text_chunks = []
        all_table_documents = []
        
        logger.info(f"Processing {len(pdf_files)} PDFs sequentially...")
        
        # Process PDFs sequentially
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing {pdf_file.name}...")
                text_chunks, table_docs = self._process_single_pdf(str(pdf_file))
                all_text_chunks.extend(text_chunks)
                all_table_documents.extend(table_docs)
                logger.info(f"Completed processing {pdf_file.name}")
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
        
        logger.info(f"Processed {len(pdf_files)} PDFs:")
        logger.info(f"  - {len(all_text_chunks)} text chunks")
        logger.info(f"  - {len(all_table_documents)} tables")
        
        return all_text_chunks, all_table_documents
    
    @profile_function("LiteratureRAG.create_vectorstores")
    def create_vectorstores(self, text_chunks: List[Document], table_documents: List[Document]) -> None:
        """Create separate vector stores for text and tables"""
        logger.info("Creating vector stores...")
        
        # Create text collection
        if text_chunks:
            logger.info(f"Creating text vector store with {len(text_chunks)} chunks...")
            with profile_context("vectorstore.create_text_collection", self.enable_profiling):
                self.text_collection = self.chroma_client.get_or_create_collection(
                    name=self.text_collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                
                # Add documents to collection
                texts = [doc.page_content for doc in text_chunks]
                metadatas = [doc.metadata for doc in text_chunks]
                ids = [f"text_{i}_{hashlib.md5(text.encode()).hexdigest()[:8]}" 
                       for i, text in enumerate(texts)]
            
            with profile_context("vectorstore.embed_text_chunks", self.enable_profiling):
                embeddings = self._embed_texts(texts)
            
            with profile_context("vectorstore.add_text_documents", self.enable_profiling):
                self.text_collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
        
        # Create table collection
        if table_documents:
            logger.info(f"Creating table vector store with {len(table_documents)} tables...")
            with profile_context("vectorstore.create_table_collection", self.enable_profiling):
                self.table_collection = self.chroma_client.get_or_create_collection(
                    name=self.table_collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                
                # Add documents to collection
                texts = [doc.page_content for doc in table_documents]
                metadatas = [doc.metadata for doc in table_documents]
                ids = [f"table_{i}_{hashlib.md5(text.encode()).hexdigest()[:8]}" 
                       for i, text in enumerate(texts)]
            
            with profile_context("vectorstore.embed_table_chunks", self.enable_profiling):
                embeddings = self._embed_texts(texts)
            
            with profile_context("vectorstore.add_table_documents", self.enable_profiling):
                self.table_collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
        
        logger.info("Vector stores created successfully")
    
    @profile_function("LiteratureRAG.load_vectorstores")
    def load_vectorstores(self) -> bool:
        """Load existing vector stores"""
        try:
            logger.info("Loading existing vector stores...")
            
            # Try to load text collection
            try:
                self.text_collection = self.chroma_client.get_collection(
                    name=self.text_collection_name
                )
                logger.info(f"Loaded text collection: {self.text_collection_name}")
            except Exception:
                logger.warning(f"Text collection {self.text_collection_name} does not exist")
                return False
            
            # Try to load table collection
            try:
                self.table_collection = self.chroma_client.get_collection(
                    name=self.table_collection_name
                )
                logger.info(f"Loaded table collection: {self.table_collection_name}")
            except Exception:
                logger.info("Table collection does not exist (optional)")
                self.table_collection = None
            
            logger.info("Vector stores loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector stores: {e}")
            return False
    
    def _clear_collections(self) -> None:
        """Clear existing collections for force reindexing"""
        try:
            # Try to delete text collection
            try:
                self.chroma_client.delete_collection(name=self.text_collection_name)
                logger.info(f"Cleared existing text collection: {self.text_collection_name}")
            except Exception as e:
                logger.info(f"Text collection {self.text_collection_name} doesn't exist or couldn't be deleted: {e}")
            
            # Try to delete table collection
            try:
                self.chroma_client.delete_collection(name=self.table_collection_name)
                logger.info(f"Cleared existing table collection: {self.table_collection_name}")
            except Exception as e:
                logger.info(f"Table collection {self.table_collection_name} doesn't exist or couldn't be deleted: {e}")
                
        except Exception as e:
            logger.warning(f"Error clearing collections: {e}")
    
    @profile_function("LiteratureRAG.index_pdfs")
    def index_pdfs(self, force_reindex: bool = False, batch_size: int = 500) -> None:
        """Index PDFs with memory-efficient batch processing"""
        if not force_reindex and self.load_vectorstores():
            logger.info("Using existing vector stores. Set force_reindex=True to rebuild.")
            return
        
        # If force reindexing, clear existing collections
        if force_reindex:
            logger.info("Force reindex enabled - clearing existing collections...")
            self._clear_collections()
        
        # Process documents in memory-efficient batches
        self._index_pdfs_in_batches(batch_size)
        logger.info("Indexing complete!")
    
    @profile_function("LiteratureRAG._index_pdfs_in_batches")
    def _index_pdfs_in_batches(self, batch_size: int = 500) -> None:
        """Process PDFs in memory-efficient batches"""
        logger.info(f"Starting batch processing with batch size: {batch_size}")
        
        if not os.path.exists(self.pdf_directory):
            logger.warning(f"PDF directory {self.pdf_directory} does not exist!")
            return
        
        pdf_files = list(Path(self.pdf_directory).glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_directory}")
            return
        
        logger.info(f"Processing {len(pdf_files)} PDFs in batches...")
        
        # Initialize collections
        self._initialize_collections()
        
        total_text_chunks = 0
        total_tables = 0
        
        # Process each PDF file individually to avoid memory accumulation
        for pdf_idx, pdf_file in enumerate(pdf_files, 1):
            try:
                logger.info(f"Processing PDF {pdf_idx}/{len(pdf_files)}: {pdf_file.name}")
                
                # Process single PDF
                text_chunks, table_docs = self._process_single_pdf(str(pdf_file))
                
                if text_chunks or table_docs:
                    # Process chunks in batches to avoid memory issues
                    self._process_chunks_in_batches(text_chunks, table_docs, batch_size)
                    
                    total_text_chunks += len(text_chunks)
                    total_tables += len(table_docs)
                
                # Clear memory after each PDF
                del text_chunks, table_docs
                
                # Force garbage collection and clear GPU cache
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"Completed PDF {pdf_idx}/{len(pdf_files)}: {pdf_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                # Clean up on error too
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        logger.info(f"Batch processing complete! Total: {total_text_chunks} text chunks, {total_tables} tables")
    
    def _initialize_collections(self) -> None:
        """Initialize ChromaDB collections"""
        # Create text collection
        self.text_collection = self.chroma_client.get_or_create_collection(
            name=self.text_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Create table collection  
        self.table_collection = self.chroma_client.get_or_create_collection(
            name=self.table_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("Initialized ChromaDB collections")
    
    @profile_function("LiteratureRAG._process_chunks_in_batches")
    def _process_chunks_in_batches(self, text_chunks: List[Document], table_docs: List[Document], batch_size: int) -> None:
        """Process document chunks in small batches"""
        
        # Process text chunks in batches
        if text_chunks:
            logger.info(f"Processing {len(text_chunks)} text chunks in batches of {batch_size}")
            for i in range(0, len(text_chunks), batch_size):
                batch = text_chunks[i:i + batch_size]
                self._add_text_batch_to_vectorstore(batch)
                
                if len(text_chunks) > batch_size:
                    logger.info(f"Processed text batch {i//batch_size + 1}/{(len(text_chunks) + batch_size - 1)//batch_size}")
        
        # Process table documents in batches
        if table_docs:
            logger.info(f"Processing {len(table_docs)} table documents in batches of {batch_size}")
            for i in range(0, len(table_docs), batch_size):
                batch = table_docs[i:i + batch_size]
                self._add_table_batch_to_vectorstore(batch)
                
                if len(table_docs) > batch_size:
                    logger.info(f"Processed table batch {i//batch_size + 1}/{(len(table_docs) + batch_size - 1)//batch_size}")
    
    @profile_function("LiteratureRAG._add_text_batch_to_vectorstore")
    def _add_text_batch_to_vectorstore(self, text_batch: List[Document]) -> None:
        """Add a batch of text documents to the vector store with memory cleanup"""
        if not text_batch:
            return
        
        texts = [doc.page_content for doc in text_batch]
        metadatas = [doc.metadata for doc in text_batch]
        ids = [f"text_{i}_{hashlib.md5(text.encode()).hexdigest()[:8]}" 
               for i, text in enumerate(texts)]
        
        try:
            # Generate embeddings for this batch
            embeddings = self._embed_texts(texts)
            
            # Add to collection
            self.text_collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
        finally:
            # Clean up intermediate variables
            if 'embeddings' in locals():
                del embeddings
            del texts, metadatas, ids
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    @profile_function("LiteratureRAG._add_table_batch_to_vectorstore")
    def _add_table_batch_to_vectorstore(self, table_batch: List[Document]) -> None:
        """Add a batch of table documents to the vector store with memory cleanup"""
        if not table_batch:
            return
        
        texts = [doc.page_content for doc in table_batch]
        metadatas = [doc.metadata for doc in table_batch]
        ids = [f"table_{i}_{hashlib.md5(text.encode()).hexdigest()[:8]}" 
               for i, text in enumerate(texts)]
        
        try:
            # Generate embeddings for this batch
            embeddings = self._embed_texts(texts)
            
            # Add to collection
            self.table_collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
        finally:
            # Clean up intermediate variables
            if 'embeddings' in locals():
                del embeddings
            del texts, metadatas, ids
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    @profile_function("LiteratureRAG._similarity_search")
    def _similarity_search(self, collection, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents in a collection with memory cleanup"""
        if not collection:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self._embed_texts([query])[0]
            
            # Search in collection
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            # Convert results to Document objects
            documents = []
            if results['documents'] and results['metadatas']:
                for text, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    doc = Document(
                        page_content=text,
                        metadata=metadata
                    )
                    documents.append(doc)
            
            return documents
        
        finally:
            # Clean up variables
            if 'query_embedding' in locals():
                del query_embedding
            if 'results' in locals():
                del results
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _filter_thinking_sections(self, text: str) -> str:
        """Remove <think></think> sections from model responses"""
        import re
        # Remove thinking sections (case insensitive, multiline)
        pattern = r'<think>.*?</think>'
        filtered_text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        # Clean up multiple whitespace/newlines
        filtered_text = re.sub(r'\n\s*\n', '\n\n', filtered_text)
        return filtered_text.strip()

    @profile_function("LiteratureRAG._generate_response")
    def _generate_response(self, query: str, context: str) -> str:
        """Generate response using Ollama"""
        prompt = f"""You are a helpful assistant for answering questions about academic literature.
Use the following pieces of context from research papers to answer the question.
The context may include both regular text and tables from the papers.
If you don't know the answer based on the context, just say that you don't know.
Always cite the paper name when providing information.

IMPORTANT: Provide a CONCISE and DIRECT answer. Keep your response brief and to the point - aim for 1-3 sentences unless the question specifically asks for more detail. Do not include thinking sections, reasoning steps, or any <think> tags in your response.

Context: {context}

Question: {query}

Answer: """
        
        # Use Ollama to generate response
        response = self.llm_client.chat(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.2}
        )
        
        # Filter out thinking sections before returning
        raw_content = response['message']['content']
        filtered_content = self._filter_thinking_sections(raw_content)
        return filtered_content
    
    @profile_function("LiteratureRAG.query")
    def query(self, question: str, include_tables: bool = True, k: int = 5) -> Dict:
        """Query with option to include table search"""
        if not hasattr(self, 'text_collection'):
            logger.error("Vector stores not initialized. Run index_pdfs() first.")
            return {
                "answer": "System not initialized. Please index PDFs first.",
                "sources": []
            }
        
        logger.info(f"Processing query: {question}")
        
        # Search in text collection
        text_docs = self._similarity_search(self.text_collection, question, k=k)
        
        # Search in tables if requested
        table_docs = []
        if include_tables and hasattr(self, 'table_collection') and self.table_collection:
            table_docs = self._similarity_search(self.table_collection, question, k=3)
        
        # Combine all documents
        all_docs = text_docs + table_docs
        
        # Create context from documents
        context_parts = []
        sources = []
        seen_sources = set()
        
        for doc in all_docs:
            paper_name = doc.metadata.get('paper_name', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            source_type = doc.metadata.get('source_type', 'text')
            source_key = f"{paper_name}_p{page}_{source_type}"
            
            if source_key not in seen_sources:
                context_parts.append(f"[{paper_name}, page {page}]: {doc.page_content}")
                
                sources.append({
                    "paper": paper_name,
                    "page": page,
                    "type": source_type,
                    "content": doc.page_content[:200] + "..."
                })
                seen_sources.add(source_key)
        
        # Generate response
        context = "\n\n".join(context_parts)
        answer = self._generate_response(question, context) if context else "No relevant documents found."
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def search_similar(self, query: str, k: int = 5, search_tables: bool = True) -> List[Dict]:
        """Search for similar content in both text and tables"""
        if not hasattr(self, 'text_collection'):
            logger.error("Vector stores not initialized. Run index_pdfs() first.")
            return []
        
        results = []
        
        # Search in text
        text_docs = self._similarity_search(self.text_collection, query, k=k)
        for doc in text_docs:
            results.append({
                "paper": doc.metadata.get('paper_name', 'Unknown'),
                "page": doc.metadata.get('page', 'N/A'),
                "type": "text",
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        # Search in tables
        if search_tables and hasattr(self, 'table_collection') and self.table_collection:
            table_docs = self._similarity_search(self.table_collection, query, k=k//2)
            for doc in table_docs:
                results.append({
                    "paper": doc.metadata.get('paper_name', 'Unknown'),
                    "page": doc.metadata.get('page', 'N/A'),
                    "type": "table",
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get statistics about the indexed content"""
        stats = {
            "text_chunks": 0,
            "tables": 0,
            "papers": set()
        }
        
        try:
            if hasattr(self, 'text_collection'):
                # Get text collection stats
                text_docs = self.text_collection.get()
                if text_docs and 'metadatas' in text_docs:
                    stats["text_chunks"] = len(text_docs['metadatas'])
                    for metadata in text_docs['metadatas']:
                        if metadata and 'paper_name' in metadata:
                            stats["papers"].add(metadata['paper_name'])
            
            if hasattr(self, 'table_collection') and self.table_collection:
                # Get table collection stats
                table_docs = self.table_collection.get()
                if table_docs and 'metadatas' in table_docs:
                    stats["tables"] = len(table_docs['metadatas'])
                    for metadata in table_docs['metadatas']:
                        if metadata and 'paper_name' in metadata:
                            stats["papers"].add(metadata['paper_name'])
            
            stats["papers"] = len(stats["papers"])
            
        except Exception as e:
            logger.warning(f"Error getting statistics: {e}")
        
        return stats
    
    def get_profiling_report(self) -> Dict:
        """Get performance profiling report"""
        return PROFILER.get_report()
    
    def print_profiling_report(self):
        """Print performance profiling report"""
        print_profiling_report()
    
    def reset_profiling(self):
        """Reset profiling statistics"""
        reset_profiling()
    
    def agentic_query(self, question: str, include_plan_details: bool = False) -> Dict[str, Any]:
        """Process query using agentic approach with tool planning"""
        if not hasattr(self, '_agentic_tools'):
            self._initialize_agentic_tools()
        
        logger.info(f"Processing agentic query: {question}")
        
        # Analyze query complexity
        analysis = self._analyze_query_complexity(question)
        logger.info(f"Query analysis: {analysis}")
        
        # Create execution plan
        plan = self._create_execution_plan(question, analysis)
        logger.info(f"Execution plan: {plan.reasoning}")
        
        # Execute plan
        tool_results = self._execute_plan(plan)
        
        # Get final result from synthesizer
        synthesis_result = None
        for result in tool_results:
            if result.tool_name == 'synthesizer':
                synthesis_result = result
                break
        
        if synthesis_result and synthesis_result.success:
            final_result = synthesis_result.result
            if isinstance(final_result, dict):
                answer = final_result.get('answer', 'No answer generated.')
                sources = final_result.get('sources', [])
                tools_used = final_result.get('tools_used', [])
            else:
                answer = str(final_result)
                sources = []
                tools_used = []
        else:
            # Fallback to basic RAG if synthesis fails
            logger.warning("Synthesis failed, falling back to basic RAG")
            basic_result = self.query(question)
            answer = basic_result['answer']
            sources = basic_result['sources']
            tools_used = ['basic_rag']
        
        response = {
            'answer': answer,
            'sources': sources,
            'tools_used': tools_used,
            'execution_stats': {
                'total_tools': len(tool_results),
                'successful_tools': sum(1 for r in tool_results if r.success),
                'total_time': sum(r.execution_time for r in tool_results),
                'query_complexity': analysis['complexity']
            }
        }
        
        if include_plan_details:
            response['execution_plan'] = {
                'reasoning': plan.reasoning,
                'tools_sequence': [t['tool'] for t in plan.tools_sequence],
                'tool_results': [
                    {
                        'tool': r.tool_name,
                        'success': r.success,
                        'execution_time': r.execution_time,
                        'metadata': r.metadata
                    }
                    for r in tool_results
                ]
            }
        
        return response
    
    def _initialize_agentic_tools(self):
        """Initialize agentic tools for advanced query processing"""
        self._agentic_tools = {
            'document_retriever': DocumentRetrieverTool(),
            'summarizer': SummarizationTool(),
            'comparator': ComparisonTool(),
            'refined_searcher': RefinedSearchTool(),
            'synthesizer': SynthesisTool()
        }
        logger.info("Agentic tools initialized")
    
    def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine complexity and required tools"""
        analysis = {
            'complexity': 'simple',
            'requires_comparison': False,
            'requires_synthesis': False,
            'requires_multi_step': False,
            'focus_areas': []
        }
        
        # Keywords that indicate different types of analysis needed
        comparison_keywords = ['compare', 'contrast', 'difference', 'versus', 'vs', 'similarities', 'differ']
        synthesis_keywords = ['overview', 'comprehensive', 'summary', 'synthesize', 'integrate', 'combine']
        complex_keywords = ['methodology', 'approach', 'framework', 'analysis', 'evaluation', 'assessment']
        
        query_lower = query.lower()
        
        # Check for comparison needs
        if any(keyword in query_lower for keyword in comparison_keywords):
            analysis['requires_comparison'] = True
            analysis['complexity'] = 'moderate'
        
        # Check for synthesis needs
        if any(keyword in query_lower for keyword in synthesis_keywords):
            analysis['requires_synthesis'] = True
            analysis['complexity'] = 'moderate'
        
        # Check for complex analysis
        if any(keyword in query_lower for keyword in complex_keywords):
            analysis['complexity'] = 'complex'
            analysis['requires_multi_step'] = True
        
        # Determine focus areas
        if 'method' in query_lower or 'approach' in query_lower:
            analysis['focus_areas'].append('methodology')
        if 'result' in query_lower or 'finding' in query_lower:
            analysis['focus_areas'].append('results')
        if 'data' in query_lower or 'dataset' in query_lower:
            analysis['focus_areas'].append('data')
        if 'limitation' in query_lower or 'challenge' in query_lower:
            analysis['focus_areas'].append('limitations')
        
        return analysis
    
    def _create_execution_plan(self, query: str, analysis: Dict[str, Any]) -> QueryPlan:
        """Create execution plan based on query analysis"""
        sub_queries = [query]
        tools_sequence = []
        
        # Always start with document retrieval
        tools_sequence.append({
            'tool': 'document_retriever',
            'context': {
                'retrieval_count': 7 if analysis['complexity'] == 'simple' else 10,
                'include_tables': True
            }
        })
        
        # Add refinement search for complex queries
        if analysis['complexity'] in ['moderate', 'complex']:
            search_focus = analysis['focus_areas'][0] if analysis['focus_areas'] else 'general'
            tools_sequence.append({
                'tool': 'refined_searcher',
                'context': {'search_focus': search_focus},
                'depends_on': ['document_retriever']
            })
        
        # Add comparison if needed
        if analysis['requires_comparison']:
            tools_sequence.append({
                'tool': 'comparator',
                'context': {},
                'depends_on': ['document_retriever']
            })
        
        # Add summarization for complex queries
        if analysis['complexity'] == 'complex' or len(analysis['focus_areas']) > 1:
            tools_sequence.append({
                'tool': 'summarizer',
                'context': {},
                'depends_on': ['document_retriever']
            })
        
        # Always end with synthesis
        tools_sequence.append({
            'tool': 'synthesizer',
            'context': {},
            'depends_on': ['document_retriever']
        })
        
        reasoning = f"Query complexity: {analysis['complexity']}. "
        if analysis['requires_comparison']:
            reasoning += "Comparison analysis needed. "
        if analysis['requires_synthesis']:
            reasoning += "Multi-source synthesis required. "
        if analysis['focus_areas']:
            reasoning += f"Focus areas: {', '.join(analysis['focus_areas'])}."
        
        return QueryPlan(
            original_query=query,
            sub_queries=sub_queries,
            tools_sequence=tools_sequence,
            reasoning=reasoning
        )
    
    def _execute_plan(self, plan: QueryPlan) -> List[ToolResult]:
        """Execute the planned sequence of tool calls"""
        results = {}
        execution_order = []
        
        # Build dependency graph and execution order
        remaining_tools = plan.tools_sequence.copy()
        
        while remaining_tools:
            # Find tools with no dependencies or all dependencies satisfied
            ready_tools = []
            for tool_config in remaining_tools:
                dependencies = tool_config.get('depends_on', [])
                if not dependencies or all(dep in execution_order for dep in dependencies):
                    ready_tools.append(tool_config)
            
            if not ready_tools:
                # Break dependency cycle by executing first remaining tool
                ready_tools = [remaining_tools[0]]
                logger.warning("Potential dependency cycle detected, proceeding with first available tool")
            
            # Execute ready tools
            for tool_config in ready_tools:
                tool_name = tool_config['tool']
                tool_context = tool_config['context'].copy()
                
                # Add results from dependency tools to context
                dependencies = tool_config.get('depends_on', [])
                for dep in dependencies:
                    if dep in results:
                        if dep == 'document_retriever':
                            tool_context['documents'] = results[dep].result
                            tool_context['initial_results'] = results[dep].result
                        else:
                            tool_context[f'{dep}_result'] = results[dep].result
                
                # Execute tool
                tool = self._agentic_tools[tool_name]
                logger.info(f"Executing tool: {tool_name}")
                result = tool.execute(plan.original_query, tool_context, self)
                results[tool_name] = result
                execution_order.append(tool_name)
                
                logger.info(f"Tool {tool_name} completed in {result.execution_time:.2f}s - Success: {result.success}")
            
            # Remove executed tools
            remaining_tools = [t for t in remaining_tools if t['tool'] not in execution_order]
        
        return list(results.values())


class DocumentRetrieverTool(AgentTool):
    def __init__(self):
        super().__init__(
            name="document_retriever",
            description="Retrieves relevant documents and passages from the literature",
            tool_type=ToolType.RETRIEVAL
        )
    
    def execute(self, query: str, context: Dict[str, Any], rag_system) -> ToolResult:
        import time
        start_time = time.perf_counter()
        
        try:
            k = context.get('retrieval_count', 7)
            include_tables = context.get('include_tables', True)
            
            # Search both text and tables
            text_docs = rag_system._similarity_search(rag_system.text_collection, query, k=k)
            table_docs = []
            
            if include_tables and hasattr(rag_system, 'table_collection') and rag_system.table_collection:
                table_docs = rag_system._similarity_search(rag_system.table_collection, query, k=3)
            
            all_docs = text_docs + table_docs
            
            # Format results
            retrieved_docs = []
            for doc in all_docs:
                retrieved_docs.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source_type': doc.metadata.get('source_type', 'text')
                })
            
            execution_time = time.perf_counter() - start_time
            return ToolResult(
                tool_name=self.name,
                success=True,
                result=retrieved_docs,
                metadata={'query': query, 'retrieved_count': len(retrieved_docs)},
                execution_time=execution_time
            )
        
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ToolResult(
                tool_name=self.name,
                success=False,
                result=str(e),
                metadata={'query': query, 'error': str(e)},
                execution_time=execution_time
            )


class SummarizationTool(AgentTool):
    def __init__(self):
        super().__init__(
            name="summarizer",
            description="Summarizes documents or passages to extract key information",
            tool_type=ToolType.ANALYSIS
        )
    
    def execute(self, query: str, context: Dict[str, Any], rag_system) -> ToolResult:
        import time
        start_time = time.perf_counter()
        
        try:
            documents = context.get('documents', [])
            if not documents:
                raise ValueError("No documents provided for summarization")
            
            # Create context from documents
            doc_contents = []
            for doc in documents:
                paper_name = doc.get('metadata', {}).get('paper_name', 'Unknown')
                page = doc.get('metadata', {}).get('page', 'N/A')
                content = doc.get('content', '')
                doc_contents.append(f"[{paper_name}, p.{page}]: {content}")
            
            combined_context = "\n\n".join(doc_contents)
            
            # Generate summary
            summary_prompt = f"""Summarize the key points from the following academic content in relation to: {query}

Content:
{combined_context}

Please provide a concise summary highlighting the main findings, methodologies, and conclusions relevant to the query."""
            
            response = rag_system.llm_client.chat(
                model=rag_system.llm_model,
                messages=[
                    {"role": "system", "content": "You are a research assistant that creates concise, accurate summaries."},
                    {"role": "user", "content": summary_prompt}
                ],
                options={"temperature": 0.2}
            )
            
            summary = rag_system._filter_thinking_sections(response['message']['content'])
            
            execution_time = time.perf_counter() - start_time
            return ToolResult(
                tool_name=self.name,
                success=True,
                result=summary,
                metadata={'query': query, 'doc_count': len(documents)},
                execution_time=execution_time
            )
        
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ToolResult(
                tool_name=self.name,
                success=False,
                result=str(e),
                metadata={'query': query, 'error': str(e)},
                execution_time=execution_time
            )


class ComparisonTool(AgentTool):
    def __init__(self):
        super().__init__(
            name="comparator",
            description="Compares findings, methodologies, or results across different papers",
            tool_type=ToolType.ANALYSIS
        )
    
    def execute(self, query: str, context: Dict[str, Any], rag_system) -> ToolResult:
        import time
        start_time = time.perf_counter()
        
        try:
            documents = context.get('documents', [])
            if len(documents) < 2:
                raise ValueError("Need at least 2 documents for comparison")
            
            # Group documents by paper
            papers = {}
            for doc in documents:
                paper_name = doc.get('metadata', {}).get('paper_name', 'Unknown')
                if paper_name not in papers:
                    papers[paper_name] = []
                papers[paper_name].append(doc)
            
            # Create comparison context
            paper_summaries = []
            for paper_name, paper_docs in papers.items():
                content_parts = []
                for doc in paper_docs:
                    page = doc.get('metadata', {}).get('page', 'N/A')
                    content = doc.get('content', '')[:500]  # Limit length
                    content_parts.append(f"Page {page}: {content}")
                
                paper_summary = f"**{paper_name}**:\n" + "\n".join(content_parts)
                paper_summaries.append(paper_summary)
            
            comparison_context = "\n\n".join(paper_summaries)
            
            # Generate comparison
            comparison_prompt = f"""Compare and contrast the information from the following papers regarding: {query}

Papers:
{comparison_context}

Please provide a structured comparison highlighting:
1. Similarities in approaches/findings
2. Key differences 
3. Conflicting results (if any)
4. Complementary insights"""
            
            response = rag_system.llm_client.chat(
                model=rag_system.llm_model,
                messages=[
                    {"role": "system", "content": "You are a research analyst expert at comparing academic literature."},
                    {"role": "user", "content": comparison_prompt}
                ],
                options={"temperature": 0.3}
            )
            
            comparison = rag_system._filter_thinking_sections(response['message']['content'])
            
            execution_time = time.perf_counter() - start_time
            return ToolResult(
                tool_name=self.name,
                success=True,
                result=comparison,
                metadata={'query': query, 'papers_compared': list(papers.keys())},
                execution_time=execution_time
            )
        
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ToolResult(
                tool_name=self.name,
                success=False,
                result=str(e),
                metadata={'query': query, 'error': str(e)},
                execution_time=execution_time
            )


class RefinedSearchTool(AgentTool):
    def __init__(self):
        super().__init__(
            name="refined_searcher",
            description="Performs targeted searches with refined queries based on initial results",
            tool_type=ToolType.SEARCH
        )
    
    def execute(self, query: str, context: Dict[str, Any], rag_system) -> ToolResult:
        import time
        start_time = time.perf_counter()
        
        try:
            initial_results = context.get('initial_results', [])
            search_focus = context.get('search_focus', 'methodology')
            
            # Generate refined search terms based on initial results
            if initial_results:
                # Extract key terms from initial results
                content_sample = "\n".join([
                    doc.get('content', '')[:200] for doc in initial_results[:3]
                ])
                
                refinement_prompt = f"""Based on this initial search result sample for the query "{query}", suggest 2-3 more specific search terms that could help find additional relevant information, focusing on {search_focus}:

Sample content:
{content_sample}

Provide only the search terms, one per line."""
                
                response = rag_system.llm_client.chat(
                    model=rag_system.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a research librarian expert at creating targeted search queries."},
                        {"role": "user", "content": refinement_prompt}
                    ],
                    options={"temperature": 0.4}
                )
                
                refined_terms = rag_system._filter_thinking_sections(response['message']['content']).strip().split('\n')
                refined_terms = [term.strip('- ').strip() for term in refined_terms if term.strip()]
            else:
                refined_terms = [query]
            
            # Perform searches with refined terms
            all_refined_docs = []
            for term in refined_terms[:3]:  # Limit to top 3 terms
                if term:
                    text_docs = rag_system._similarity_search(rag_system.text_collection, term, k=3)
                    for doc in text_docs:
                        all_refined_docs.append({
                            'content': doc.page_content,
                            'metadata': doc.metadata,
                            'source_type': doc.metadata.get('source_type', 'text'),
                            'search_term': term
                        })
            
            # Remove duplicates based on content hash
            seen_hashes = set()
            unique_docs = []
            for doc in all_refined_docs:
                content_hash = hash(doc['content'])
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    unique_docs.append(doc)
            
            execution_time = time.perf_counter() - start_time
            return ToolResult(
                tool_name=self.name,
                success=True,
                result=unique_docs,
                metadata={
                    'original_query': query,
                    'refined_terms': refined_terms,
                    'results_found': len(unique_docs)
                },
                execution_time=execution_time
            )
        
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ToolResult(
                tool_name=self.name,
                success=False,
                result=str(e),
                metadata={'query': query, 'error': str(e)},
                execution_time=execution_time
            )


class SynthesisTool(AgentTool):
    def __init__(self):
        super().__init__(
            name="synthesizer",
            description="Synthesizes information from multiple tool results to create comprehensive answers",
            tool_type=ToolType.SYNTHESIS
        )
    
    def execute(self, query: str, context: Dict[str, Any], rag_system) -> ToolResult:
        import time
        start_time = time.perf_counter()
        
        try:
            tool_results = context.get('tool_results', [])
            documents = context.get('documents', [])
            
            # Use documents from retrieval if available
            if documents:
                synthesis_content = []
                sources = []
                
                # Extract sources from retrieval results
                for doc in documents:
                    if isinstance(doc, dict):
                        metadata = doc.get('metadata', {})
                        sources.append({
                            'paper': metadata.get('paper_name', 'Unknown'),
                            'page': metadata.get('page', 'N/A'),
                            'type': metadata.get('source_type', 'text')
                        })
                
                # Create comprehensive synthesis
                context_parts = []
                for doc in documents[:7]:  # Use top 7 docs
                    if isinstance(doc, dict):
                        metadata = doc.get('metadata', {})
                        paper_name = metadata.get('paper_name', 'Unknown')
                        page = metadata.get('page', 'N/A')
                        content = doc.get('content', '')
                        context_parts.append(f"[{paper_name}, p.{page}]: {content}")
                
                combined_context = "\n\n".join(context_parts)
                
                synthesis_prompt = f"""Based on the following research literature, provide a CONCISE answer to: {query}

Literature:
{combined_context}

Please provide a brief, direct response (2-4 sentences) that synthesizes the key information from the sources and cites the papers appropriately. Avoid lengthy explanations unless specifically requested."""
                
                response = rag_system.llm_client.chat(
                    model=rag_system.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a research analyst creating comprehensive literature-based answers."},
                        {"role": "user", "content": synthesis_prompt}
                    ],
                    options={"temperature": 0.3}
                )
                
                synthesis_text = rag_system._filter_thinking_sections(response['message']['content'])
            else:
                synthesis_text = "No sufficient information found to synthesize a comprehensive answer."
                sources = []
            
            # Deduplicate sources
            unique_sources = []
            seen_sources = set()
            for source in sources:
                source_key = f"{source['paper']}_{source['page']}_{source['type']}"
                if source_key not in seen_sources:
                    unique_sources.append(source)
                    seen_sources.add(source_key)
            
            execution_time = time.perf_counter() - start_time
            return ToolResult(
                tool_name=self.name,
                success=True,
                result={
                    'answer': synthesis_text,
                    'sources': unique_sources,
                    'tools_used': ['document_retriever', 'synthesizer']
                },
                metadata={'query': query, 'sources_count': len(unique_sources)},
                execution_time=execution_time
            )
        
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ToolResult(
                tool_name=self.name,
                success=False,
                result=str(e),
                metadata={'query': query, 'error': str(e)},
                execution_time=execution_time
            )