# RAG System Performance Profiling

The RAG system now includes comprehensive performance profiling to help identify bottlenecks and optimize performance.

## How to Enable Profiling

### Command Line Options

1. **Enable profiling**: Use `--enable-profiling` flag
2. **Show report**: Use `--profile-report` flag  
3. **Dedicated profiling**: Use `--action profile`

### Examples

```bash
# Run with profiling enabled and show report after indexing
python main.py --action index --enable-profiling --profile-report

# Run a dedicated profiling test
python main.py --action profile --question "What are the main findings?"

# Query with profiling
python main.py --action query --question "test" --enable-profiling --profile-report

# Interactive mode with profiling
python main.py --action interactive --enable-profiling
```

## Interactive Mode Commands

When profiling is enabled in interactive mode, you get additional commands:

- `profile` - Show current profiling report
- `reset-profile` - Reset profiling statistics

## What Gets Profiled

The profiling system tracks:

### Time Metrics
- **Calls**: Number of times each function was called
- **Total Time**: Total execution time across all calls
- **Average Time**: Average execution time per call  
- **Min/Max Time**: Fastest and slowest execution times

### Memory Metrics
- **Memory Usage**: Memory delta during function execution
- **Average Memory**: Average memory usage per call
- **Peak Memory**: Highest memory usage observed

### Key Operations Tracked

1. **PDF Processing**:
   - `EnhancedPDFProcessor.extract_tables_from_pdf`
   - `EnhancedPDFProcessor.process_pdf_with_tables`
   - `EnhancedPDFProcessor._process_text_pages`

2. **RAG Operations**:
   - `LiteratureRAG._initialize_components`
   - `LiteratureRAG.load_and_process_documents`
   - `LiteratureRAG._process_single_pdf`
   - `LiteratureRAG.index_pdfs`

3. **Vectorstore Operations**:
   - `LiteratureRAG.create_vectorstores`
   - `LiteratureRAG._embed_texts`
   - `vectorstore.create_text_collection`
   - `vectorstore.embed_text_chunks`
   - `vectorstore.add_text_documents`
   - `vectorstore.create_table_collection`
   - `vectorstore.embed_table_chunks`
   - `vectorstore.add_table_documents`

4. **Query Operations**:
   - `LiteratureRAG.query`
   - `LiteratureRAG._similarity_search`
   - `LiteratureRAG._generate_response`

## Sample Output

```
📊 PERFORMANCE PROFILING REPORT
================================================================================
Operation                                Calls    Total(s)   Avg(s)     Min(s)     Max(s)     Mem(MB)   
------------------------------------------------------------------------------------------------------------
LiteratureRAG._embed_texts               4        12.45      3.113      2.850      3.450      150.2     
vectorstore.embed_text_chunks            1        8.92       8.920      8.920      8.920      120.5     
EnhancedPDFProcessor.extract_tables_...  3        5.67       1.890      1.234      2.100      45.3      
LiteratureRAG.create_vectorstores        1        4.23       4.230      4.230      4.230      85.1      
LiteratureRAG._process_single_pdf        3        2.98       0.993      0.850      1.200      25.7      
================================================================================

🔥 TOP TIME CONSUMERS:
   35.2% - _embed_texts
   25.3% - embed_text_chunks
   16.1% - extract_tables_from_pdf
   12.0% - create_vectorstores
   8.4% - _process_single_pdf
```

## Interpreting Results

### Common Bottlenecks

1. **Embedding Generation** (`_embed_texts`): Usually the largest time consumer
   - Consider using GPU if available
   - Optimize batch sizes
   - Use ONNX backend for CPU optimization

2. **PDF Processing** (`extract_tables_from_pdf`): Can be slow for complex PDFs
   - Consider processing fewer pages
   - Optimize table extraction settings

3. **Vectorstore Operations**: Database operations can be I/O bound
   - Consider using faster storage
   - Optimize collection settings

### Performance Tips

1. **Enable GPU**: Use CUDA if available for embedding generation
2. **Batch Size**: Optimize embedding batch sizes (currently 128)
3. **Threading**: PyTorch threading is auto-configured
4. **Storage**: Use SSD storage for vector databases
5. **Memory**: Monitor memory usage for large document sets

## Programmatic Access

```python
from rag_system import LiteratureRAG

# Initialize with profiling
rag = LiteratureRAG(enable_profiling=True)

# Get profiling data
stats = rag.get_profiling_report()

# Print formatted report
rag.print_profiling_report()

# Reset statistics
rag.reset_profiling()
```

This profiling system helps identify exactly where time is being spent in your RAG pipeline, making optimization much more targeted and effective.