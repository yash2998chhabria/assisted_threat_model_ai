#!/usr/bin/env python3
"""
Vector Search - Search through code chunks stored in ChromaDB
Provides semantic search with filtering and ranking
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
import json


class VectorSearch:
    """Search code chunks in ChromaDB using semantic similarity"""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2", db_path: str = "./chroma_db"):
        self.model_name = model_name
        self.db_path = db_path
        
        print(f"🤖 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        print(f"🗄️ Connecting to ChromaDB at: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Collection name must match the one used in vector_load.py
        self.collection_name = f"codebase_{model_name.replace('-', '_').replace('.', '_')}"
        
        try:
            self.collection = self.client.get_collection(self.collection_name)
            count = self.collection.count()
            print(f"📊 Connected to collection: {self.collection_name} ({count} chunks)")
            
            # Validate that the collection has the right embedding dimensions
            if count > 0:
                self._validate_collection()
                
        except Exception as e:
            print(f"❌ Error: Collection '{self.collection_name}' not found")
            print(f"   Make sure you've run vector_load.py first with the same model")
            raise e
    
    def _validate_collection(self):
        """Validate that the collection is compatible with our model"""
        try:
            # Test with a small query to check dimensions
            test_embedding = self.model.encode(["test"])[0].tolist()
            sample = self.collection.get(limit=1, include=["embeddings"])
            
            # ✅ FIXED: Proper array checking
            if sample["embeddings"] is not None and len(sample["embeddings"]) > 0:
                stored_dim = len(sample["embeddings"][0])
                query_dim = len(test_embedding)
                
                if stored_dim != query_dim:
                    print(f"⚠️  Warning: Dimension mismatch!")
                    print(f"   Stored embeddings: {stored_dim}D")
                    print(f"   Current model: {query_dim}D")
                    print(f"   Make sure you're using the same model as during loading")
                else:
                    print(f"✅ Model compatibility verified ({query_dim}D embeddings)")
            else:
                print("⚠️  No embeddings found to validate")
                    
        except Exception as e:
            print(f"⚠️  Could not validate model compatibility: {e}")
    
    def search(
        self, 
        query: str, 
        limit: int = 10,
        language: Optional[str] = None,
        chunk_type: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for code chunks using semantic similarity
        
        Args:
            query: Search query (natural language or code)
            limit: Maximum number of results
            language: Filter by programming language (e.g., 'typescript', 'python')
            chunk_type: Filter by chunk type (e.g., 'function_declaration', 'class_declaration')
            min_score: Minimum similarity score (0.0 to 1.0)
        """
        print(f"🔍 Searching for: '{query}'")
        
        # Create query embedding
        query_embedding = self.model.encode([query])[0].tolist()
        
        # Build where clause for filtering
        where_clause = {}
        if language:
            where_clause["language"] = language
        if chunk_type:
            where_clause["chunk_type"] = chunk_type
        
        # Perform search
        try:
            # Use query_embeddings to ensure consistent model usage
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            processed_results = []
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                similarity = 1 - distance  # Convert distance to similarity
                
                if similarity < min_score:
                    continue
                
                result = {
                    "id": results["ids"][0][i],
                    "similarity": similarity,
                    "distance": distance,
                    "metadata": results["metadatas"][0][i],
                    "document": results["documents"][0][i]
                }
                
                # Extract original code from the enhanced document
                doc = result["document"]
                code_start = doc.find("Code:\n")
                if code_start != -1:
                    result["code"] = doc[code_start + 6:]  # Remove "Code:\n"
                else:
                    result["code"] = doc
                
                processed_results.append(result)
            
            print(f"✅ Found {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def search_by_function_name(self, function_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for functions by exact name match"""
        return self.search(
            query=f"function {function_name}",
            limit=limit,
            chunk_type="function_declaration"
        )
    
    def search_by_keywords(self, keywords: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Search using multiple keywords"""
        query = " ".join(keywords)
        return self.search(query, limit)
    
    def find_similar_code(self, code_snippet: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find code similar to a given snippet"""
        return self.search(f"Code: {code_snippet}", limit)
    
    def search_by_file(self, filepath: str, query: str = "", limit: int = 10) -> List[Dict[str, Any]]:
        """Search within a specific file"""
        # ChromaDB doesn't support complex where clauses with contains,
        # so we'll search and filter results
        all_results = self.search(query if query else filepath, limit=limit*3)
        
        # Filter results by filepath
        filtered_results = []
        for result in all_results:
            if filepath in result["metadata"].get("filepath", ""):
                filtered_results.append(result)
                if len(filtered_results) >= limit:
                    break
        
        return filtered_results
    
    def search_by_complexity(self, min_complexity: float, max_complexity: float = 100.0, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for code by complexity score range"""
        # Get more results and filter by complexity
        all_results = self.search("complex function method", limit=limit*5)
        
        filtered_results = []
        for result in all_results:
            complexity = result["metadata"].get("complexity_score", 0)
            try:
                complexity = float(complexity)
                if min_complexity <= complexity <= max_complexity:
                    filtered_results.append(result)
                    if len(filtered_results) >= limit:
                        break
            except (ValueError, TypeError):
                continue
        
        return filtered_results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by its ID"""
        try:
            result = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"]
            )
            
            if result["ids"]:
                doc = result["documents"][0]
                code_start = doc.find("Code:\n")
                code = doc[code_start + 6:] if code_start != -1 else doc
                
                return {
                    "id": result["ids"][0],
                    "metadata": result["metadatas"][0],
                    "document": doc,
                    "code": code
                }
            return None
            
        except Exception as e:
            print(f"❌ Error getting chunk {chunk_id}: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            total_count = self.collection.count()
            
            # Get sample to analyze metadata
            sample_size = min(100, total_count)
            sample = self.collection.get(limit=sample_size, include=["metadatas"])
            
            languages = {}
            chunk_types = {}
            
            # Handle case where metadatas might be None or empty
            if sample["metadatas"] and len(sample["metadatas"]) > 0:
                for meta in sample["metadatas"]:
                    if meta:  # Check if metadata dict is not None
                        lang = meta.get("language", "unknown")
                        chunk_type = meta.get("chunk_type", "unknown")
                        
                        languages[lang] = languages.get(lang, 0) + 1
                        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            return {
                "total_chunks": total_count,
                "sample_size": len(sample["metadatas"]) if sample["metadatas"] else 0,
                "languages": languages,
                "chunk_types": chunk_types,
                "collection_name": self.collection_name,
                "model": self.model_name
            }
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {
                "total_chunks": 0,
                "sample_size": 0,
                "languages": {},
                "chunk_types": {},
                "collection_name": self.collection_name,
                "model": self.model_name,
                "error": str(e)
            }
    
    def diagnose(self) -> Dict[str, Any]:
        """Diagnose potential issues with the database"""
        issues = []
        warnings = []
        
        try:
            count = self.collection.count()
            if count == 0:
                issues.append("Database is empty - no chunks loaded")
                return {"issues": issues, "warnings": warnings, "healthy": False}
            
            # Check sample data
            sample = self.collection.get(limit=5, include=["metadatas", "embeddings", "documents"])
            
            if not sample["embeddings"] or len(sample["embeddings"]) == 0:
                issues.append("No embeddings found in database")
            else:
                # Check embedding dimensions
                dims = len(sample["embeddings"][0])
                test_embedding = self.model.encode(["test"])[0]
                if len(test_embedding) != dims:
                    issues.append(f"Embedding dimension mismatch: stored={dims}, model={len(test_embedding)}")
            
            # Check metadata quality
            if sample["metadatas"]:
                required_fields = ["chunk_type", "language", "name"]
                missing_fields = []
                
                for field in required_fields:
                    if not any(meta.get(field) for meta in sample["metadatas"] if meta):
                        missing_fields.append(field)
                
                if missing_fields:
                    warnings.append(f"Some metadata fields are missing: {missing_fields}")
            
            # Check document content
            if sample["documents"]:
                avg_length = sum(len(doc) for doc in sample["documents"]) / len(sample["documents"])
                if avg_length < 50:
                    warnings.append(f"Documents seem very short (avg: {avg_length:.0f} chars)")
                elif avg_length > 10000:
                    warnings.append(f"Documents seem very long (avg: {avg_length:.0f} chars)")
            
            return {
                "issues": issues,
                "warnings": warnings,
                "healthy": len(issues) == 0,
                "total_chunks": count,
                "embedding_dims": dims if 'dims' in locals() else None,
                "avg_doc_length": avg_length if 'avg_length' in locals() else None
            }
            
        except Exception as e:
            issues.append(f"Diagnostic error: {e}")
            return {"issues": issues, "warnings": warnings, "healthy": False}


def format_search_result(result: Dict[str, Any], show_code: bool = True) -> str:
    """Format a search result for display"""
    metadata = result["metadata"]
    
    output = []
    output.append(f"📍 {result['id']}")
    output.append(f"   🎯 Similarity: {result['similarity']:.3f}")
    output.append(f"   📝 Type: {metadata.get('chunk_type', 'unknown')}")
    output.append(f"   🔤 Language: {metadata.get('language', 'unknown')}")
    output.append(f"   🏷️  Name: {metadata.get('name', 'unnamed')}")
    
    # Show file and line info
    filepath = metadata.get('filepath', 'unknown')
    line_start = metadata.get('line_start')
    line_end = metadata.get('line_end')
    
    if line_start and line_end:
        output.append(f"   📁 File: {filepath}:{line_start}-{line_end}")
    else:
        output.append(f"   📁 File: {filepath}")
    
    if metadata.get("parent_name"):
        output.append(f"   👤 Parent: {metadata['parent_name']}")
    
    if metadata.get("summary"):
        output.append(f"   📄 Summary: {metadata['summary']}")
    
    # Show complexity if available
    if metadata.get("complexity_score"):
        try:
            complexity = float(metadata["complexity_score"])
            output.append(f"   🧮 Complexity: {complexity:.1f}")
        except:
            pass
    
    if metadata.get("keywords"):
        keywords = metadata["keywords"].split(" | ")[:5]  # Show first 5 keywords
        output.append(f"   🔑 Keywords: {', '.join(keywords)}")
    
    if show_code:
        code = result["code"]
        # Add syntax highlighting hint
        lang = metadata.get('language', '')
        if len(code) > 500:
            code = code[:497] + "..."
        output.append(f"\n   💻 Code ({lang}):")
        # Indent the code
        indented_code = '\n'.join('      ' + line for line in code.split('\n'))
        output.append(indented_code)
    
    return "\n".join(output)


def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python vector_search.py <query> [options]")
        print("\nOptions:")
        print("  --limit=N              Number of results (default: 10)")
        print("  --language=LANG        Filter by language (e.g., typescript, python)")
        print("  --type=TYPE            Filter by chunk type (e.g., function_declaration)")
        print("  --min-score=SCORE      Minimum similarity score (0.0-1.0)")
        print("  --model=MODEL          Embedding model (default: all-mpnet-base-v2)")
        print("  --db-path=PATH         Database path (default: ./chroma_db)")
        print("  --no-code              Don't show code in results")
        print("  --json                 Output as JSON")
        print("  --stats                Show database statistics")
        print("  --validate             Validate database and model compatibility")
        print("  --diagnose             Run diagnostic checks on database")
        print("  --file=PATH            Search within specific file")
        print("  --complexity=MIN-MAX   Search by complexity range (e.g., --complexity=5-10)")
        print("\nExamples:")
        print("  python vector_search.py 'websocket authentication'")
        print("  python vector_search.py 'async function' --language=typescript --limit=5")
        print("  python vector_search.py 'error handling' --type=function_declaration")
        print("  python vector_search.py 'auth' --file='auth.test.ts'")
        print("  python vector_search.py 'complex algorithm' --complexity=8-15")
        print("  python vector_search.py --stats")
        print("  python vector_search.py --validate")
        sys.exit(1)
    
    # Parse arguments
    query = sys.argv[1]
    limit = 10
    language = None
    chunk_type = None
    min_score = 0.0
    model_name = "all-mpnet-base-v2"
    db_path = "./chroma_db"
    show_code = True
    output_json = False
    show_stats = False
    validate_only = False
    search_file = None
    complexity_range = None
    
    for arg in sys.argv[2:]:
        if arg.startswith("--limit="):
            limit = int(arg.split("=", 1)[1])
        elif arg.startswith("--language="):
            language = arg.split("=", 1)[1]
        elif arg.startswith("--type="):
            chunk_type = arg.split("=", 1)[1]
        elif arg.startswith("--min-score="):
            min_score = float(arg.split("=", 1)[1])
        elif arg.startswith("--model="):
            model_name = arg.split("=", 1)[1]
        elif arg.startswith("--db-path="):
            db_path = arg.split("=", 1)[1]
        elif arg.startswith("--file="):
            search_file = arg.split("=", 1)[1]
        elif arg.startswith("--complexity="):
            complexity_str = arg.split("=", 1)[1]
            if "-" in complexity_str:
                min_comp, max_comp = complexity_str.split("-")
                complexity_range = (float(min_comp), float(max_comp))
        elif arg == "--no-code":
            show_code = False
        elif arg == "--json":
            output_json = True
        elif arg == "--stats":
            show_stats = True
        elif arg == "--validate":
            validate_only = True
        elif arg == "--diagnose":
            show_stats = True  # Show both stats and diagnostics
    
    if show_stats or validate_only:
        query = ""  # Don't need query for stats/validation
    
    print(f"🚀 Vector Search Starting")
    if not show_stats and not validate_only:
        print(f"🔍 Query: '{query}'")
    print(f"🤖 Model: {model_name}")
    print(f"💾 Database: {db_path}")
    print("=" * 50)
    
    try:
        searcher = VectorSearch(model_name=model_name, db_path=db_path)
        
        if validate_only:
            print("\n✅ Database validation completed successfully!")
            return
        
        if show_stats:
            stats = searcher.get_stats()
            diagnostic = searcher.diagnose()
            
            if output_json:
                output_data = {"stats": stats, "diagnostic": diagnostic}
                print(json.dumps(output_data, indent=2))
            else:
                print("\n📊 Database Statistics:")
                print(f"   Total chunks: {stats.get('total_chunks', 0)}")
                print(f"   Sample analyzed: {stats.get('sample_size', 0)}")
                print(f"   Languages: {stats.get('languages', {})}")
                print(f"   Chunk types: {stats.get('chunk_types', {})}")
                
                print("\n🩺 Health Check:")
                if diagnostic.get("healthy"):
                    print("   ✅ Database is healthy")
                else:
                    print("   ❌ Issues detected:")
                    for issue in diagnostic.get("issues", []):
                        print(f"      • {issue}")
                
                if diagnostic.get("warnings"):
                    print("   ⚠️  Warnings:")
                    for warning in diagnostic.get("warnings", []):
                        print(f"      • {warning}")
                
                if diagnostic.get("embedding_dims"):
                    print(f"   📐 Embedding dimensions: {diagnostic['embedding_dims']}")
                if diagnostic.get("avg_doc_length"):
                    print(f"   📄 Average document length: {diagnostic['avg_doc_length']:.0f} chars")
        else:
            start_time = time.time()
            
            # Choose search method based on options
            if search_file:
                results = searcher.search_by_file(search_file, query, limit)
                print(f"🔍 Searching in file: {search_file}")
            elif complexity_range:
                results = searcher.search_by_complexity(complexity_range[0], complexity_range[1], limit)
                print(f"🔍 Searching by complexity: {complexity_range[0]}-{complexity_range[1]}")
            else:
                results = searcher.search(
                    query=query,
                    limit=limit,
                    language=language,
                    chunk_type=chunk_type,
                    min_score=min_score
                )
            
            search_time = time.time() - start_time
            
            if output_json:
                print(json.dumps(results, indent=2))
            else:
                print(f"\n📋 Results ({len(results)} found in {search_time:.3f}s):")
                print("=" * 50)
                
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {format_search_result(result, show_code)}")
                    print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()