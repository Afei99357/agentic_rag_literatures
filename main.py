import argparse
from rag_system import LiteratureRAG, reset_profiling, print_profiling_report


def print_statistics(rag):
    """Print system statistics"""
    stats = rag.get_statistics()
    print("\n" + "="*60)
    print("SYSTEM STATISTICS")
    print("="*60)
    print(f"Papers indexed: {stats['papers']}")
    print(f"Text chunks: {stats['text_chunks']}")
    print(f"Tables extracted: {stats['tables']}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Agentic Literature RAG System - Smart Document Chat Bot")
    parser.add_argument(
        "--action",
        choices=["index", "stats", "profile"],
        help="Action to perform (optional - defaults to interactive chat)"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to ask and exit (otherwise starts interactive chat)"
    )
    parser.add_argument(
        "--pdf-dir",
        default="./pdfs",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--db-dir",
        default="./chroma_db",
        help="Directory for vector database"
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force reindexing of PDFs"
    )
    parser.add_argument(
        "--enable-profiling",
        action="store_true",
        default=False,
        help="Enable performance profiling"
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Show execution plan and detailed information"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Batch size for processing chunks during indexing (default: 200)"
    )
    
    args = parser.parse_args()
    
    # Reset profiling at start if enabled
    if args.enable_profiling:
        reset_profiling()
    
    rag = LiteratureRAG(
        pdf_directory=args.pdf_dir,
        db_directory=args.db_dir,
        enable_profiling=args.enable_profiling
    )
    
    # Handle special actions
    if args.action == "index":
        print("Indexing PDFs...")
        rag.index_pdfs(force_reindex=args.force_reindex, batch_size=args.batch_size)
        print("Indexing complete!")
        print_statistics(rag)
        return
    
    elif args.action == "stats":
        rag.load_vectorstores()
        print_statistics(rag)
        return
    
    elif args.action == "profile":
        print("🔄 Running profiling test...")
        reset_profiling()
        rag.index_pdfs(force_reindex=args.force_reindex, batch_size=args.batch_size)
        test_question = args.question or "What are the main findings?"
        print(f"Running test query: {test_question}")
        result = rag.agentic_query(test_question, include_plan_details=args.show_details)
        print("\n📊 PROFILING RESULTS:")
        print_profiling_report()
        return
    
    # Handle single question mode
    if args.question:
        print("🤖 Agentic Literature RAG - Processing your question...")
        rag.index_pdfs()
        result = rag.agentic_query(args.question, include_plan_details=args.show_details)
        
        if args.show_details and 'execution_plan' in result:
            print("\n" + "="*80)
            print("🔍 EXECUTION PLAN:")
            print("="*80)
            plan = result['execution_plan']
            print(f"Reasoning: {plan['reasoning']}")
            print(f"Tools: {' → '.join(plan['tools_sequence'])}")
            
            print("\nTool Results:")
            for tool_result in plan['tool_results']:
                status = "✅" if tool_result['success'] else "❌"
                print(f"  {status} {tool_result['tool']}: {tool_result['execution_time']:.2f}s")
        
        print("\n" + "="*80)
        print("🤖 ANSWER:")
        print("="*80)
        print(result["answer"])
        
        if result.get("sources"):
            print("\n" + "="*80)
            print("📚 SOURCES:")
            print("="*80)
            for i, source in enumerate(result["sources"], 1):
                content_type = "📊" if source.get('type') == 'table' else "📄"
                print(f"{i}. {content_type} {source['paper']}, Page {source['page']} ({source.get('type', 'text')})")
        
        if args.show_details and 'execution_stats' in result:
            stats = result['execution_stats']
            print(f"\n📊 Stats: {stats['successful_tools']}/{stats['total_tools']} tools, "
                  f"{stats['total_time']:.2f}s, {stats['query_complexity']} complexity")
        
        return
    
    # Default to interactive chat mode
    else:
        print("="*80)
        print("🤖 Agentic Literature RAG - Smart Document Chat Bot")
        print("="*80)
        print("\n🔧 Commands:")
        print("  - Just type your question to chat with your documents")
        print("  - Type 'details' to toggle detailed execution plans")
        print("  - Type 'index' to (re)index PDFs") 
        print("  - Type 'stats' to show system statistics")
        print("  - Type 'quit' or 'exit' to stop")
        print("="*80)
        
        rag.index_pdfs()
        print_statistics(rag)
        
        show_details = args.show_details
        
        while True:
            try:
                user_input = input("\n💬 Ask me about your documents: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'details':
                    show_details = not show_details
                    status = "enabled" if show_details else "disabled"
                    print(f"🔍 Detailed execution plans {status}")
                    continue
                
                elif user_input.lower() == 'index':
                    print("🔄 Reindexing PDFs...")
                    rag.index_pdfs(force_reindex=True)
                    print("✅ Indexing complete!")
                    print_statistics(rag)
                    continue
                
                elif user_input.lower() == 'stats':
                    print_statistics(rag)
                    continue
                
                elif user_input.lower() in ['help', 'h']:
                    print("\n🔧 Available commands:")
                    print("  - Just type your question to chat")
                    print("  - 'details' - toggle execution plan details")
                    print("  - 'index' - reindex PDFs")
                    print("  - 'stats' - show statistics")
                    print("  - 'quit' - exit the chat")
                    continue
                
                elif user_input:
                    print("\n🤖 Thinking...")
                    result = rag.agentic_query(user_input, include_plan_details=show_details)
                    
                    # Show execution plan if enabled
                    if show_details and 'execution_plan' in result:
                        print("\n" + "-"*60)
                        print("🔍 EXECUTION PLAN:")
                        print("-"*60)
                        plan = result['execution_plan']
                        print(f"Reasoning: {plan['reasoning']}")
                        print(f"Tools: {' → '.join(plan['tools_sequence'])}")
                    
                    print("\n" + "-"*60)
                    print("🤖 ANSWER:")
                    print("-"*60)
                    print(result["answer"])
                    
                    # Show execution stats if details enabled
                    if show_details and 'execution_stats' in result:
                        stats = result['execution_stats']
                        print(f"\n📊 {stats['successful_tools']}/{stats['total_tools']} tools, "
                              f"{stats['total_time']:.1f}s, {stats['query_complexity']} complexity")
                    
                    # Always show sources
                    if result.get("sources"):
                        print("\n" + "-"*60)
                        print("📚 SOURCES:")
                        print("-"*60)
                        for i, source in enumerate(result["sources"], 1):
                            content_type = "📊" if source.get('type') == 'table' else "📄"
                            print(f"{i}. {content_type} {source['paper']}, p.{source['page']} ({source.get('type', 'text')})")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again or type 'help' for commands.")


if __name__ == "__main__":
    main()
