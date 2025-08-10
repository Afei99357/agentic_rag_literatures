# 🤖 Agentic Literature RAG - Smart Document Chat Bot

An advanced **Agentic RAG** system that acts like a smart chat bot for your academic 
literature. Simply start it up and ask questions naturally - no complex commands needed! 
Features intelligent query planning, multi-tool execution, enhanced PDF processing, 
table extraction, and concise synthesis capabilities.

## ✨ What Makes This Special?

- **🤖 Always Agentic**: Every query automatically uses intelligent multi-tool processing
- **💬 Natural Chat Interface**: Just start it and ask questions like talking to a research assistant  
- **📄 Smart Document Processing**: Handles both text and tables from PDFs automatically
- **🎯 Concise Answers**: Gets straight to the point with 2-4 sentence responses
- **🔧 Zero Configuration**: Works out of the box with local models - no API keys needed!

## Features

### 🤖 Agentic Intelligence

- **Intelligent Query Planning**: Automatically analyzes query complexity and creates 
  execution plans
- **Multi-Tool Orchestration**: Coordinates multiple specialized tools for 
  comprehensive answers
- **Adaptive Reasoning**: Selects appropriate analysis strategies based on query type
- **Tool Dependencies**: Manages tool execution order and data flow between tools
- **Performance Monitoring**: Tracks tool execution and provides detailed analytics

### 🔧 Specialized Agent Tools

- **Document Retriever**: Advanced semantic search across text and tables
- **Summarization Tool**: Extracts key insights from retrieved documents
- **Comparison Tool**: Analyzes similarities and differences across papers
- **Refined Search Tool**: Generates and executes targeted follow-up queries
- **Synthesis Tool**: Creates comprehensive answers from multiple tool outputs

### Enhanced PDF Processing

- **Smart Chunking**: Improved text chunking with better context preservation
- **Table Detection**: Automatic extraction and indexing of tables from PDFs using 
  PyMuPDF (fast) or pdfplumber (fallback)
- **Dual Storage**: Separate vector stores for text content and tables in ChromaDB
- **Metadata Tracking**: Rich metadata including document IDs, indexing timestamps, 
  and content types
- **Native Multithreading**: ONNX backend provides optimal CPU multithreading 
  without Python GIL limitations
- **GPU Acceleration**: Automatic GPU detection and utilization for embeddings 
  when available
- **Memory-Efficient Processing**: Batch processing with automatic garbage collection 
  and GPU memory management

### Advanced Search Capabilities

- **Multi-Modal Search**: Search across both text and table content
- **Content Type Identification**: Clear distinction between text and table sources
- **Enhanced Statistics**: Detailed statistics about indexed content
- **Response Filtering**: Automatic removal of thinking sections from model responses 
  (e.g., `<think></think>`)

### Performance & Monitoring

- **Performance Profiling**: Built-in profiling system to monitor function execution 
  times and memory usage
- **Memory Management**: Automatic garbage collection and GPU memory clearing
- **Batch Processing**: Memory-efficient processing with configurable batch sizes
- **Threading Optimization**: Automatic detection and utilization of all CPU cores

### Core Features

- **PDF Processing**: Automatically extracts and processes text from PDF files
- **Vector Search**: Uses embeddings to find relevant passages from your literature
- **Question Answering**: Get answers to questions with citations from source papers
- **Local Models**: Uses free local models (Ollama + ModernBERT/MiniLM)
- **Interactive CLI**: User-friendly command-line interface with profiling support

## 🚀 Quick Start

### 1. Prerequisites

#### Install uv (Python Package Manager)
First, install uv for dependency management:

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with homebrew (macOS)
brew install uv
```

#### Install Ollama
Next, install Ollama for local LLM inference:

```bash
# Linux/WSL
curl -fsSL https://ollama.ai/install.sh | sh

# macOS
brew install ollama

# Or download from: https://ollama.ai
```

#### Install Qwen3 Model
After installing Ollama, pull the Qwen3 model:

```bash
# Install the Qwen3 model
ollama pull qwen3:latest

# Or for specific sizes if available:
# ollama pull qwen3:8b     # Specific size variant
# ollama pull qwen3:4b     # Smaller, faster variant
```

#### Install Python Dependencies
```bash
# Install dependencies with uv
uv sync
```

No virtual environment activation needed - uv handles everything!

### 2. Add Your PDFs

Place your PDF files in the `pdfs/` directory:
```bash
cp /path/to/your/papers/*.pdf pdfs/
```

### 3. Start Chatting!

```bash
# Start the interactive chat bot (default)
uv run python main.py
```

That's it! Just type your questions naturally like:
- "What are the main findings?"
- "Compare the different methodologies"
- "What are the limitations mentioned?"

### Alternative Usage

```bash
# Ask a single question and exit
uv run python main.py --question "What are the main findings?"

# Show detailed execution plans
uv run python main.py --question "Compare methods" --show-details

# Administrative tasks
uv run python main.py --action index --force-reindex
uv run python main.py --action stats
uv run python main.py --action profile
```

## 💬 How to Use the Chat Bot

### Interactive Commands

Once you start the chat bot with `python main.py`, you can use these commands:

- **Just ask questions naturally** - The system automatically uses agentic processing
- **`details`** - Toggle detailed execution plans on/off
- **`index`** - Reindex PDFs (use after adding new documents)
- **`stats`** - Show system statistics
- **`help`** - Show available commands
- **`quit`** or `exit`** - Stop the chat bot

### Sample Questions

The chat bot handles all types of questions intelligently:

**Simple Questions:**
```
What are the main findings?
What methods were used?
What datasets are mentioned?
```

**Complex Analysis:**
```
Compare the attention mechanisms across different papers
What are the common limitations mentioned in the literature?
How do the evaluation metrics differ between studies?
Summarize the evolution of techniques across papers
```

**Research-Focused:**
```
What gaps in research are identified?
What future work is suggested?
How do the results compare across different approaches?
What are the key contributions of each paper?
```

## ⚙️ Configuration

### Using Local Models (Default)

The system uses free, local models by default:

- **Embeddings**: ModernBERT-base (with ONNX backend for native multithreading) 
  or MiniLM-L6-v2 fallback
- **LLM**: Ollama with qwen3:latest 
- **Responses**: Automatically generates concise, direct answers (2-4 sentences)

No API keys required!

### Prerequisites

1. **Install Ollama**: Download from https://ollama.ai

2. **Install required model**:
   ```bash
   ollama pull qwen3:latest
   ```

3. **Optional - ONNX Runtime**: For better CPU performance:
   ```bash
   pip install sentence-transformers[onnx]
   ```

## Project Structure

```
agentic-rag/
 pdfs/              # Place your PDF files here
 chroma_db/         # Vector database storage
 main.py            # Main CLI interface
 rag_system.py      # Core RAG implementation
 .env               # Configuration (create from .env.example)
```

## 🛠️ Advanced Options

```bash
# Use custom directories
uv run python main.py --pdf-dir /path/to/pdfs --db-dir /path/to/db

# Force reindex (rebuild database)
uv run python main.py --action index --force-reindex

# Enable performance profiling
uv run python main.py --enable-profiling

# Custom batch size for memory management
uv run python main.py --action index --batch-size 200
```

## 📁 Project Structure

```
agentic-rag/
   pdfs/              # Place your PDF files here
   chroma_db/         # Vector database storage (auto-created)
   main.py            # Complete application - chat bot & CLI
   rag_system.py      # Core RAG implementation with agentic tools
   .env               # Configuration (optional)
```

## Troubleshooting

### No PDFs found

- Ensure PDF files are in the `pdfs/` directory
- Check file permissions

### Ollama not working

- Install from: https://ollama.ai
- Run: `ollama pull qwen3:latest`
- Ensure Ollama service is running: `ollama serve` (if needed)

### Out of memory

- Reduce batch_size: `--batch-size 50`
- Use fewer PDFs at once
- Enable profiling to identify memory-intensive operations: `--enable-profiling`

### Slow performance

- First indexing takes time (subsequent queries are faster)
- System now uses ONNX backend for optimal CPU multithreading
- ModernBERT provides better performance than standard BERT models
- Install ONNX support: `pip install sentence-transformers[onnx]`
- GPU automatically detected if available for faster embeddings
- Use performance profiling to identify bottlenecks: `--enable-profiling --profile-report`
- Increase batch size for better throughput: `--batch-size 500`

## Requirements

- Python 3.10+
- uv (Python package manager)
- 4GB+ RAM recommended
- Multi-core CPU (system automatically detects and uses all available cores)
- Optional: CUDA-compatible GPU for accelerated embeddings
- Ollama (for local LLM)

## New Features in Enhanced Version

### Performance Optimizations

- **ONNX Backend**: Uses ONNX runtime for native CPU multithreading without 
  Python GIL limitations
- **ModernBERT Architecture**: Advanced transformer model with RoPE and optimized 
  attention mechanisms
- **Intelligent Fallback**: ModernBERT-base → MiniLM-L6-v2 + ONNX → MiniLM-L6-v2 + PyTorch
- **Memory-Efficient Batch Processing**: Configurable batch sizes with automatic 
  memory cleanup
- **Native Multithreading**: PyTorch thread configuration leverages all CPU cores efficiently
- **GPU Acceleration**: Automatic CUDA detection with fallback to optimized CPU inference
- **Performance Profiling**: Built-in profiling system to monitor and optimize performance

### Table Processing

- **PyMuPDF First**: Uses PyMuPDF for fast table extraction with fallback to pdfplumber
- **Structured Table Storage**: Tables are stored separately in ChromaDB for 
  optimized retrieval
- **Table Metadata**: Rich metadata tracking including table dimensions and page locations

### Dual Vector Stores

- **Text Collection**: `literature_text` - Stores text chunks from papers
- **Table Collection**: `literature_tables` - Stores extracted tables with structured data
- **Smart Routing**: Queries intelligently search across both collections

### Enhanced Metadata

Each document now includes:

- `source_type`: Identifies content as 'text' or 'table'
- `doc_id`: Unique identifier for tracking document versions
- `indexed_at`: Timestamp for cache management
- Additional table-specific metadata (rows, columns, position)

### Response Quality

- **Thinking Section Filtering**: Automatically removes `<think></think>` sections 
  from model responses
- **Clean Output**: Ensures only the final answer is returned without internal reasoning

### Monitoring & Diagnostics

- **Performance Profiling**: Track function execution times and memory usage
- **Memory Management**: Automatic garbage collection and GPU memory cleanup
- **Statistics Dashboard**: Detailed statistics about indexed content and system performance

### API Improvements

```python
# New methods
rag.get_statistics()                    # System statistics
rag.query(question, include_tables=True) # Table-aware queries  
rag.search_similar(query, search_tables=True) # Multi-modal search
rag.get_profiling_report()              # Performance metrics
rag.print_profiling_report()            # Formatted performance report
```

## License

MIT