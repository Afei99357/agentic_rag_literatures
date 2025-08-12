# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Primary Development Workflow
```bash
# Install dependencies
uv sync

# Interactive development/testing
uv run python main.py

# Single query testing
uv run python main.py --question "test query" --show-details

# Force reindex for development
uv run python main.py --action index --force-reindex

# Performance analysis
uv run python main.py --enable-profiling --action profile

# System diagnostics
uv run python main.py --action stats
```

### Key Development Options
- `--enable-profiling`: Track performance metrics for optimization
- `--show-details`: Display execution plans for debugging agentic workflow
- `--force-reindex`: Complete database rebuild (useful after code changes)
- `--batch-size N`: Memory optimization during development

## Architecture Overview

### Core Design: Agentic RAG
This system implements an agentic approach where queries trigger intelligent multi-tool processing rather than simple retrieval+generation. The architecture consists of:

1. **Query Complexity Analysis** → **Tool Execution Planning** → **Multi-Tool Orchestration** → **Synthesis**
2. **Dual Vector Stores**: Separate collections for text (`literature_text`) and tables (`literature_tables`)
3. **Five Specialized Agent Tools**: DocumentRetriever, Summarizer, Comparator, RefinedSearcher, SynthesisTool

### Key Architectural Components

**`main.py`**: CLI interface with interactive chat mode, single-query mode, and administrative actions

**`rag_system.py`**: Core implementation containing:
- `LiteratureRAG`: Main orchestrator with agentic query processing
- `EnhancedPDFProcessor`: Dual text/table extraction with PyMuPDF (primary) + pdfplumber (fallback)
- Five `AgentTool` subclasses implementing the agentic workflow
- Comprehensive profiling system with memory and time tracking

### Tool Dependency System
Tools execute based on dependency graphs defined in `QueryPlan.tools_sequence`. The `_execute_plan()` method handles:
- Dependency resolution and execution ordering
- Context passing between tools (e.g., `documents` from DocumentRetriever to other tools)
- Error handling with informative responses when tools fail

### Query Complexity Analysis
Automatic categorization based on keywords:
- **Simple**: Basic retrieval (uses 7 docs, minimal tools)
- **Moderate**: Comparison/synthesis needed (uses 10 docs, adds RefinedSearcher)
- **Complex**: Multi-step analysis (adds Summarizer, multiple tool coordination)

## Technical Decisions

### No-LangChain Architecture
Direct integration with:
- **Ollama** (qwen3:latest) for local LLM inference
- **SentenceTransformers** (ModernBERT-base) with ONNX backend for embeddings
- **ChromaDB** for vector storage with cosine similarity
- **PyMuPDF/pdfplumber** for PDF processing

### Memory-Efficient Patterns
- Context managers (`memory_managed_pdf_processing`) for automatic resource cleanup
- Batch processing with configurable sizes and garbage collection
- GPU memory management with automatic CUDA cache clearing
- Sequential PDF processing to avoid memory accumulation

### Performance Profiling Integration
The `@profile_function` decorator and `profile_context` manager track:
- Function call counts, execution times (total/avg/min/max)
- Memory usage deltas and peak consumption
- Threading utilization and optimization opportunities

Use `print_profiling_report()` to identify bottlenecks during development.

## Development Patterns

### Adding New Agent Tools
1. Inherit from `AgentTool` with proper `ToolType`
2. Implement `execute(query, context, rag_system)` returning `ToolResult`
3. Add to `_agentic_tools` dictionary in `_initialize_agentic_tools()`
4. Update `_create_execution_plan()` logic for tool selection
5. Handle dependencies in `_execute_plan()` context passing

### Extending Query Analysis
Modify `_analyze_query_complexity()` to:
- Add new keyword categories for different analysis types
- Adjust complexity scoring based on query characteristics  
- Update `_create_execution_plan()` to use new analysis results

### Vector Store Extensions
The dual vector store pattern can be extended:
- Create new ChromaDB collections for different content types
- Add collection-specific search methods following `_similarity_search()` pattern
- Update metadata schemas for new content types

## Prerequisites for Development

### Required Services
- **Ollama** installed with `qwen3:latest` model (`ollama pull qwen3:latest`)
- **Python 3.10+** with UV package manager

### Optional Optimizations
- **CUDA GPU** for accelerated embeddings (auto-detected)
- **ONNX Runtime** for better CPU performance (`pip install sentence-transformers[onnx]`)

### Directory Structure
- `pdfs/`: Place test PDFs here (gitignored)
- `chroma_db/`: Vector database storage (gitignored)
- Generated log files and virtual environments are gitignored

## Key Files for Development

- **`rag_system.py:533`**: `LiteratureRAG.__init__()` - Main configuration
- **`rag_system.py:1216`**: `agentic_query()` - Entry point for agentic processing  
- **`rag_system.py:1288`**: `_initialize_agentic_tools()` - Tool registry
- **`rag_system.py:1343`**: `_create_execution_plan()` - Planning logic
- **`main.py:17`**: CLI argument parsing and modes

The system emphasizes local-first operation (no API keys required), comprehensive performance monitoring, and extensible agentic capabilities while maintaining memory efficiency for large document collections.