#!/usr/bin/env python3
"""
Vector Load - Clean implementation for loading code chunks into ChromaDB
Handles complex metadata with lists, ensures proper storage
"""

import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


class VectorLoader:
    """Load code chunks into ChromaDB with proper metadata handling"""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2", db_path: str = "./chroma_db"):
        self.model_name = model_name
        self.db_path = db_path
        
        print(f"🤖 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        print(f"🗄️ Initializing ChromaDB at: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Collection name based on model to avoid conflicts
        self.collection_name = f"codebase_{model_name.replace('-', '_').replace('.', '_')}"
        
        # Clear existing collection if it exists
        try:
            self.client.delete_collection(self.collection_name)
            print(f"🔄 Cleared existing collection: {self.collection_name}")
        except:
            print(f"📝 Creating new collection: {self.collection_name}")
        
        # Create fresh collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={
                "model": model_name,
                "created_at": time.time()
            }
        )
    
    def prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert complex metadata to ChromaDB-compatible format"""
        clean_metadata = {}
        
        # Handle each field explicitly
        for key, value in metadata.items():
            if value is None:
                continue  # Skip None values entirely
            
            elif key == "keywords" and isinstance(value, list):
                # Convert keywords list to string
                if value:
                    clean_keywords = [str(kw).strip() for kw in value if kw is not None and str(kw).strip()]
                    clean_metadata["keywords"] = " | ".join(clean_keywords) if clean_keywords else ""
                
            elif key == "dependencies" and isinstance(value, list):
                # Convert dependencies list to string
                if value:
                    clean_deps = [str(dep).strip() for dep in value if dep is not None and str(dep).strip()]
                    clean_metadata["dependencies"] = " | ".join(clean_deps) if clean_deps else ""
                
            elif isinstance(value, (str, int, float, bool)):
                # Direct assignment for primitive types
                if isinstance(value, str) and value.strip():
                    clean_metadata[key] = value.strip()
                elif not isinstance(value, str):
                    clean_metadata[key] = value
                    
            elif isinstance(value, list):
                # Convert other lists to strings
                if value:
                    clean_items = [str(item).strip() for item in value if item is not None and str(item).strip()]
                    clean_metadata[key] = " | ".join(clean_items) if clean_items else ""
                    
            else:
                # Convert other types to string
                try:
                    str_value = str(value).strip()
                    if str_value and str_value.lower() not in ["none", "null"]:
                        clean_metadata[key] = str_value
                except:
                    continue
        
        return clean_metadata
    
    def create_enhanced_text(self, chunk: Dict[str, Any]) -> str:
        """Create enhanced text for better embedding"""
        metadata = chunk["metadata"]
        code_text = chunk["text"]
        
        parts = []
        
        # Add structured metadata for context
        if metadata.get("summary"):
            parts.append(f"Summary: {metadata['summary']}")
        
        parts.append(f"Type: {metadata.get('chunk_type', 'unknown')}")
        parts.append(f"Language: {metadata.get('language', 'unknown')}")
        parts.append(f"Name: {metadata.get('name', 'unnamed')}")
        
        if metadata.get("parent_name"):
            parts.append(f"Parent: {metadata['parent_name']}")
        
        # Add keywords for better searchability
        if metadata.get("keywords"):
            keywords = metadata["keywords"][:10]  # Limit to first 10 keywords
            parts.append(f"Keywords: {', '.join(keywords)}")
        
        # Add dependencies
        if metadata.get("dependencies"):
            deps = metadata["dependencies"][:5]  # Limit to first 5 dependencies
            parts.append(f"Uses: {', '.join(deps)}")
        
        # Add the actual code
        parts.append(f"Code:\n{code_text}")
        
        return "\n".join(parts)
    
    def load_chunks(self, chunks_file: str, batch_size: int = 100):
        """Load chunks from JSON file into ChromaDB"""
        print(f"📥 Loading chunks from: {chunks_file}")
        
        # Load data
        with open(chunks_file, 'r') as f:
            data = json.load(f)
        
        chunks = data["chunks"]
        total_chunks = len(chunks)
        print(f"📊 Found {total_chunks} chunks to process")
        
        # Process in batches for memory efficiency
        successful_loads = 0
        failed_loads = 0
        
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            
            print(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)")
            
            # Prepare batch data
            batch_ids = []
            batch_texts = []
            batch_metadatas = []
            
            for chunk in batch_chunks:
                try:
                    # Prepare data
                    chunk_id = chunk["id"]
                    enhanced_text = self.create_enhanced_text(chunk)
                    clean_metadata = self.prepare_metadata(chunk["metadata"])
                    
                    batch_ids.append(chunk_id)
                    batch_texts.append(enhanced_text)
                    batch_metadatas.append(clean_metadata)
                    
                except Exception as e:
                    print(f"   ⚠️ Error preparing chunk {chunk.get('id', 'unknown')}: {e}")
                    failed_loads += 1
                    continue
            
            if not batch_ids:
                print(f"   ❌ No valid chunks in batch {batch_num}")
                continue
            
            # Generate embeddings for batch
            try:
                print(f"   🔢 Generating embeddings for {len(batch_texts)} chunks...")
                embeddings = self.model.encode(batch_texts, show_progress_bar=False)
                print(f"   ✅ Generated {len(embeddings)} embeddings ({embeddings.shape[1]} dimensions)")
                
                # Store in ChromaDB
                self.collection.add(
                    ids=batch_ids,
                    embeddings=embeddings.tolist(),
                    documents=batch_texts,
                    metadatas=batch_metadatas
                )
                
                successful_loads += len(batch_ids)
                print(f"   💾 Stored {len(batch_ids)} chunks successfully")
                
            except Exception as e:
                print(f"   ❌ Error storing batch {batch_num}: {e}")
                failed_loads += len(batch_chunks)
        
        # Final statistics
        print(f"\n📊 Loading Summary:")
        print(f"   ✅ Successfully loaded: {successful_loads}")
        print(f"   ❌ Failed to load: {failed_loads}")
        print(f"   📈 Success rate: {successful_loads/total_chunks*100:.1f}%")
        print(f"   🗄️ Collection: {self.collection_name}")
        print(f"   💾 Storage: {self.db_path}")
        
        return successful_loads, failed_loads
    
    def verify_load(self):
        """Verify the loaded data"""
        count = self.collection.count()
        print(f"\n🔍 Verification:")
        print(f"   📊 Total chunks in database: {count}")
        
        if count > 0:
            # Test a sample query
            sample = self.collection.get(limit=1, include=["metadatas", "embeddings"])
            if sample["embeddings"]:
                dims = len(sample["embeddings"][0])
                print(f"   🔢 Embedding dimensions: {dims}")
                print(f"   🤖 Model: {self.model_name}")
                
                # Show sample metadata
                if sample["metadatas"]:
                    meta = sample["metadatas"][0]
                    print(f"   📝 Sample metadata fields: {list(meta.keys())}")
        
        return count


def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python vector_load.py <chunks_file.json> [--model=all-mpnet-base-v2] [--db-path=./chroma_db]")
        print("\nExamples:")
        print("  python vector_load.py chunks.json")
        print("  python vector_load.py chunks.json --model=all-MiniLM-L6-v2")
        print("  python vector_load.py chunks.json --model=all-mpnet-base-v2 --db-path=./my_db")
        sys.exit(1)
    
    chunks_file = sys.argv[1]
    
    # Parse arguments
    model_name = "all-mpnet-base-v2"  # Default
    db_path = "./chroma_db"  # Default
    
    for arg in sys.argv[2:]:
        if arg.startswith("--model="):
            model_name = arg.split("=", 1)[1]
        elif arg.startswith("--db-path="):
            db_path = arg.split("=", 1)[1]
    
    print(f"🚀 Vector Load Starting")
    print(f"📁 Chunks file: {chunks_file}")
    print(f"🤖 Model: {model_name}")
    print(f"💾 Database path: {db_path}")
    print("=" * 50)
    
    # Verify input file exists
    if not Path(chunks_file).exists():
        print(f"❌ Error: File not found: {chunks_file}")
        sys.exit(1)
    
    # Initialize loader and load data
    start_time = time.time()
    
    try:
        loader = VectorLoader(model_name=model_name, db_path=db_path)
        successful, failed = loader.load_chunks(chunks_file)
        loader.verify_load()
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️ Total time: {elapsed_time:.2f} seconds")
        print(f"🎯 Ready for search! Use:")
        print(f"   python vector_search.py 'your query' --model={model_name} --db-path={db_path}")
        
    except Exception as e:
        print(f"❌ Error during loading: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
