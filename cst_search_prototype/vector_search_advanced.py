#!/usr/bin/env python3
"""
Vector Search - Clean implementation for searching code chunks
Works with data loaded by vector_load.py (ConcurrentVectorLoader)
"""

import sys
import time
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer


class CodeVectorSearcher:
    """Search code chunks using vector similarity"""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2", db_path: str = "./chroma_db"):
        self.model_name = model_name
        self.db_path = db_path
        
        print(f"🤖 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        print(f"🗄️ Connecting to ChromaDB at: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Collection name must match the loader
        self.collection_name = f"codebase_{model_name.replace('-', '_').replace('.', '_')}"
        print(f"🗂️ Looking for collection: {self.collection_name}")
        
        try:
            self.collection = self.client.get_collection(self.collection_name)
            count = self.collection.count()
            print(f"📂 Connected to collection: {self.collection_name} ({count} chunks)")
        except Exception as e:
            print(f"❌ Error: Collection '{self.collection_name}' not found")
            print(f"💡 Make sure you've run vector_load.py first with the same model and db-path")
            raise e
    
    def reconstruct_metadata(self, chromadb_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct original metadata format from ChromaDB storage"""
        metadata = {}
        
        for key, value in chromadb_metadata.items():
            if key == "keywords" and isinstance(value, str):
                # Convert string back to list
                if value.strip():
                    metadata[key] = [kw.strip() for kw in value.split(" | ") if kw.strip()]
                else:
                    metadata[key] = []
                    
            elif key == "dependencies" and isinstance(value, str):
                # Convert string back to list
                if value.strip():
                    metadata[key] = [dep.strip() for dep in value.split(" | ") if dep.strip()]
                else:
                    metadata[key] = []
                    
            else:
                metadata[key] = value
        
        return metadata
    
    def extract_code(self, enhanced_text: str) -> str:
        """Extract just the code portion from enhanced text"""
        if "Code:\n" in enhanced_text:
            return enhanced_text.split("Code:\n", 1)[1]
        return enhanced_text
    
    def search(self, query: str, n_results: int = 5, 
               language_filter: Optional[str] = None,
               chunk_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for code chunks"""
        
        # Check if database has data
        count = self.collection.count()
        if count == 0:
            print("❌ No data found in database")
            print(f"💡 Run: python vector_load.py <chunks_file> --model={self.model_name}")
            return []
        
        print(f"🔍 Searching '{query}' in {count} chunks...")
        
        # Build filter conditions
        where_conditions = {}
        if language_filter:
            where_conditions["language"] = language_filter
        if chunk_type_filter:
            where_conditions["chunk_type"] = chunk_type_filter
        
        where_clause = where_conditions if where_conditions else None
        
        try:
            # Generate query embedding with our model
            query_embedding = self.model.encode([query])[0].tolist()
            print(f"🔢 Generated query embedding: {len(query_embedding)} dimensions")
            
            # Perform vector search using pre-computed embedding
            results = self.collection.query(
                query_embeddings=[query_embedding],  # Use our embedding, not raw text
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            search_results = []
            for i in range(len(results["ids"][0])):
                # Reconstruct metadata
                chromadb_metadata = results["metadatas"][0][i]
                metadata = self.reconstruct_metadata(chromadb_metadata)
                
                # Extract code
                enhanced_text = results["documents"][0][i]
                code_text = self.extract_code(enhanced_text)
                
                # Calculate similarity score
                distance = results["distances"][0][i]
                similarity = 1.0 - distance  # Convert distance to similarity
                
                result = {
                    "id": results["ids"][0][i],
                    "similarity": similarity,
                    "code": code_text,
                    "metadata": metadata
                }
                
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def print_results(self, results: List[Dict[str, Any]], show_code_lines: int = 10):
        """Print search results in a formatted way"""
        if not results:
            print("No results found")
            return
        
        print(f"\nFound {len(results)} results:")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            metadata = result["metadata"]
            similarity_percent = result["similarity"] * 100
            
            # Header
            print(f"\n{i}. File: {metadata.get('filepath', 'Unknown file')}")
            print(f"   Similarity: {similarity_percent:.1f}%")
            print(f"   Type: {metadata.get('chunk_type', 'unknown')}: {metadata.get('name', 'unnamed')}")
            
            # Location info
            if metadata.get('line_start') and metadata.get('line_end'):
                print(f"   Lines: {metadata['line_start']}-{metadata['line_end']}")
            
            # Language and parent
            lang = metadata.get('language', 'unknown')
            parent = metadata.get('parent_name')
            if parent:
                print(f"   Language: {lang} (in {parent})")
            else:
                print(f"   Language: {lang}")
            
            # Summary
            if metadata.get('summary'):
                print(f"   Summary: {metadata['summary']}")
            
            # Keywords (show first 8)
            keywords = metadata.get('keywords', [])
            if keywords:
                keywords_display = keywords[:8]
                if len(keywords) > 8:
                    keywords_display.append(f"(+{len(keywords)-8} more)")
                print(f"   Keywords: {', '.join(keywords_display)}")
            
            # Dependencies (show first 5)
            dependencies = metadata.get('dependencies', [])
            if dependencies:
                deps_display = dependencies[:5]
                if len(dependencies) > 5:
                    deps_display.append(f"(+{len(dependencies)-5} more)")
                print(f"   Uses: {', '.join(deps_display)}")
            
            # Complexity
            if metadata.get('complexity_score'):
                complexity = metadata['complexity_score']
                print(f"   Complexity: {complexity:.1f}/10")
            
            # Code preview
            code_lines = result["code"].split('\n')
            preview_lines = code_lines[:show_code_lines]
            
            print(f"   Code preview ({len(code_lines)} total lines):")
            for line in preview_lines:
                print(f"      {line}")
            
            if len(code_lines) > show_code_lines:
                print(f"      ... ({len(code_lines) - show_code_lines} more lines)")
            
            print("-" * 80)
    
    def get_stats(self):
        """Get database statistics"""
        count = self.collection.count()
        print(f"📊 Database Statistics:")
        print(f"   Total chunks: {count}")
        print(f"   Model: {self.model_name}")
        print(f"   Database: {self.db_path}")
        print(f"   Collection: {self.collection_name}")
        
        if count == 0:
            print("   ❌ No data found")
            return
        
        # Get sample data for analysis
        sample_size = min(100, count)
        sample = self.collection.get(
            limit=sample_size, 
            include=["metadatas", "embeddings"]
        )
        
        if sample["embeddings"]:
            dims = len(sample["embeddings"][0])
            print(f"   🔢 Embedding dimensions: {dims}")
        
        # Analyze languages and chunk types
        if sample["metadatas"]:
            languages = {}
            chunk_types = {}
            
            for metadata in sample["metadatas"]:
                # Reconstruct to get proper data
                clean_meta = self.reconstruct_metadata(metadata)
                
                lang = clean_meta.get("language", "unknown")
                chunk_type = clean_meta.get("chunk_type", "unknown")
                
                languages[lang] = languages.get(lang, 0) + 1
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            print(f"   📈 Languages (sample of {sample_size}):")
            for lang, count_lang in sorted(languages.items(), key=lambda x: x[1], reverse=True):
                print(f"      {lang}: {count_lang}")
            
            print(f"   📋 Chunk types (sample of {sample_size}):")
            for chunk_type, count_type in sorted(chunk_types.items(), key=lambda x: x[1], reverse=True):
                print(f"      {chunk_type}: {count_type}")
        
        # Test query performance
        try:
            start_time = time.time()
            test_results = self.collection.query(
                query_texts=["test query"],
                n_results=5
            )
            query_time = time.time() - start_time
            print(f"   ⚡ Query performance: {query_time:.3f}s for 5 results")
        except Exception as e:
            print(f"   ⚠️ Performance test failed: {e}")
    
    def search_interactive(self):
        """Interactive search mode"""
        print(f"\n🔍 Interactive Search Mode")
        print(f"Database: {self.collection.count()} chunks loaded")
        print("Commands:")
        print("  <query>                    - Search for query")
        print("  lang:<language> <query>    - Filter by language")
        print("  type:<chunk_type> <query>  - Filter by chunk type")
        print("  stats                      - Show database stats")
        print("  quit                       - Exit")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n🔍 Search> ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                if user_input.lower() == 'stats':
                    self.get_stats()
                    continue
                
                # Parse filters
                language_filter = None
                chunk_type_filter = None
                query = user_input
                
                if user_input.startswith('lang:'):
                    parts = user_input.split(' ', 1)
                    if len(parts) == 2:
                        language_filter = parts[0][5:]  # Remove 'lang:'
                        query = parts[1]
                
                elif user_input.startswith('type:'):
                    parts = user_input.split(' ', 1)
                    if len(parts) == 2:
                        chunk_type_filter = parts[0][5:]  # Remove 'type:'
                        query = parts[1]
                
                # Perform search
                results = self.search(
                    query=query,
                    n_results=5,
                    language_filter=language_filter,
                    chunk_type_filter=chunk_type_filter
                )
                
                self.print_results(results, show_code_lines=6)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python vector_search.py <query> [options]")
        print("  python vector_search.py interactive [options]")
        print("  python vector_search.py stats [options]")
        print("\nOptions:")
        print("  --model=MODEL        Embedding model (default: all-mpnet-base-v2)")
        print("  --db-path=PATH       Database path (default: ./chroma_db)")
        print("  --results=N          Number of results (default: 5)")
        print("  --language=LANG      Filter by language (e.g., python, typescript)")
        print("  --type=TYPE          Filter by chunk type (e.g., function_definition)")
        print("  --code-lines=N       Lines of code to show (default: 10)")
        print("\nExamples:")
        print("  python vector_search.py 'authentication'")
        print("  python vector_search.py 'websocket' --language=typescript --results=10")
        print("  python vector_search.py 'database query' --type=function_definition")
        print("  python vector_search.py interactive")
        print("  python vector_search.py stats")
        sys.exit(1)
    
    command_or_query = sys.argv[1]
    
    # Parse arguments
    model_name = "all-mpnet-base-v2"
    db_path = "./chroma_db"
    n_results = 5
    language_filter = None
    chunk_type_filter = None
    code_lines = 10
    
    for arg in sys.argv[2:]:
        if arg.startswith("--model="):
            model_name = arg.split("=", 1)[1]
        elif arg.startswith("--db-path="):
            db_path = arg.split("=", 1)[1]
        elif arg.startswith("--results="):
            n_results = int(arg.split("=", 1)[1])
        elif arg.startswith("--language="):
            language_filter = arg.split("=", 1)[1]
        elif arg.startswith("--type="):
            chunk_type_filter = arg.split("=", 1)[1]
        elif arg.startswith("--code-lines="):
            code_lines = int(arg.split("=", 1)[1])
    
    try:
        searcher = CodeVectorSearcher(model_name=model_name, db_path=db_path)
        
        if command_or_query.lower() == "stats":
            # Show database statistics
            searcher.get_stats()
            
        elif command_or_query.lower() == "interactive":
            # Enter interactive mode
            searcher.search_interactive()
            
        else:
            # Perform search
            query = command_or_query
            
            print(f"🔍 Query: '{query}'")
            if language_filter:
                print(f"🏷️ Language filter: {language_filter}")
            if chunk_type_filter:
                print(f"🔹 Type filter: {chunk_type_filter}")
            
            results = searcher.search(
                query=query,
                n_results=n_results,
                language_filter=language_filter,
                chunk_type_filter=chunk_type_filter
            )
            
            searcher.print_results(results, show_code_lines=code_lines)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()