#!/usr/bin/env python3
"""
Vector Load - Concurrent implementation for loading code chunks into ChromaDB
High-performance loading with concurrent threading, batching, and reliability features
Note: Uses ThreadPoolExecutor for I/O concurrency (not true CPU parallelism due to Python's GIL)
"""

import json
import time
import sys
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import gc


class ConcurrentVectorLoader:
    """High-performance loader with concurrent threading and reliability"""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2", db_path: str = "./chroma_db", 
                 max_workers: int = 4, embedding_batch_size: int = 32):
        self.model_name = model_name
        self.db_path = db_path
        self.max_workers = max_workers
        self.embedding_batch_size = embedding_batch_size
        
        print(f"🤖 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        print(f"🗄️ Initializing ChromaDB at: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Collection name based on model to avoid conflicts
        self.collection_name = f"codebase_{model_name.replace('-', '_').replace('.', '_')}"
        
        # Thread-safe counters
        self.lock = threading.Lock()
        self.successful_loads = 0
        self.failed_loads = 0
        self.total_processed = 0
        
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
                "created_at": time.time(),
                "max_workers": max_workers
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
    
    def process_chunk_batch(self, chunks: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """Process a batch of chunks into IDs, texts, and metadata"""
        batch_ids = []
        batch_texts = []
        batch_metadatas = []
        
        for chunk in chunks:
            try:
                # Prepare data
                chunk_id = chunk["id"]
                enhanced_text = self.create_enhanced_text(chunk)
                clean_metadata = self.prepare_metadata(chunk["metadata"])
                
                batch_ids.append(chunk_id)
                batch_texts.append(enhanced_text)
                batch_metadatas.append(clean_metadata)
                
            except Exception as e:
                with self.lock:
                    self.failed_loads += 1
                print(f"   ⚠️ Error preparing chunk {chunk.get('id', 'unknown')}: {e}")
                continue
        
        return batch_ids, batch_texts, batch_metadatas
    
    def generate_embeddings_for_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts with sub-batching for memory efficiency"""
        if len(texts) <= self.embedding_batch_size:
            # Small batch, process directly
            return self.model.encode(texts, show_progress_bar=False)
        
        # Large batch, process in sub-batches
        all_embeddings = []
        for i in range(0, len(texts), self.embedding_batch_size):
            sub_batch = texts[i:i + self.embedding_batch_size]
            sub_embeddings = self.model.encode(sub_batch, show_progress_bar=False)
            all_embeddings.append(sub_embeddings)
        
        return np.vstack(all_embeddings)
    
    def store_batch_with_retry(self, batch_ids: List[str], embeddings: np.ndarray, 
                              batch_texts: List[str], batch_metadatas: List[Dict[str, Any]],
                              max_retries: int = 3) -> int:
        """Store batch with retry logic for reliability"""
        
        for attempt in range(max_retries):
            try:
                # Try to store the entire batch
                self.collection.add(
                    ids=batch_ids,
                    embeddings=embeddings.tolist(),
                    documents=batch_texts,
                    metadatas=batch_metadatas
                )
                
                with self.lock:
                    self.successful_loads += len(batch_ids)
                
                return len(batch_ids)
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"   ⚠️ Batch storage attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    print(f"   ❌ Batch storage failed after {max_retries} attempts: {e}")
                    # Try individual item storage as last resort
                    return self._store_items_individually(batch_ids, embeddings, batch_texts, batch_metadatas)
        
        return 0
    
    def _store_items_individually(self, batch_ids: List[str], embeddings: np.ndarray,
                                 batch_texts: List[str], batch_metadatas: List[Dict[str, Any]]) -> int:
        """Fallback: store items one by one"""
        print(f"   🔄 Falling back to individual storage for {len(batch_ids)} items...")
        successful_individual = 0
        
        for i, (chunk_id, embedding, text, metadata) in enumerate(zip(batch_ids, embeddings, batch_texts, batch_metadatas)):
            try:
                self.collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding.tolist()],
                    documents=[text],
                    metadatas=[metadata]
                )
                successful_individual += 1
                
            except Exception as e:
                print(f"   ⚠️ Failed to store individual item {i}: {e}")
                with self.lock:
                    self.failed_loads += 1
        
        with self.lock:
            self.successful_loads += successful_individual
        
        return successful_individual
    
    def process_batch_worker(self, batch_data: Tuple[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Worker function to process a single batch concurrently"""
        batch_num, chunks = batch_data
        start_time = time.time()
        
        try:
            # Step 1: Prepare batch data
            batch_ids, batch_texts, batch_metadatas = self.process_chunk_batch(chunks)
            
            if not batch_ids:
                return {
                    "batch_num": batch_num,
                    "success": False,
                    "error": "No valid chunks in batch",
                    "processed": 0,
                    "time": time.time() - start_time
                }
            
            # Step 2: Generate embeddings
            embeddings = self.generate_embeddings_for_batch(batch_texts)
            
            # Step 3: Store with retry logic
            stored_count = self.store_batch_with_retry(batch_ids, embeddings, batch_texts, batch_metadatas)
            
            processing_time = time.time() - start_time
            
            # Update progress
            with self.lock:
                self.total_processed += len(chunks)
            
            return {
                "batch_num": batch_num,
                "success": True,
                "processed": stored_count,
                "total_chunks": len(chunks),
                "time": processing_time,
                "embedding_dims": embeddings.shape[1] if len(embeddings) > 0 else 0
            }
            
        except Exception as e:
            with self.lock:
                self.failed_loads += len(chunks)
                self.total_processed += len(chunks)
            
            return {
                "batch_num": batch_num,
                "success": False,
                "error": str(e),
                "processed": 0,
                "total_chunks": len(chunks),
                "time": time.time() - start_time
            }
    
    def load_chunks(self, chunks_file: str, storage_batch_size: int = 100):
        """Load chunks with concurrent threading"""
        print(f"📥 Loading chunks from: {chunks_file}")
        
        # Load data
        with open(chunks_file, 'r') as f:
            data = json.load(f)
        
        chunks = data["chunks"]
        total_chunks = len(chunks)
        print(f"📊 Found {total_chunks} chunks to process")
        print(f"⚡ Using {self.max_workers} workers with {storage_batch_size} chunks per batch")
        print(f"🔢 Embedding batch size: {self.embedding_batch_size}")
        
        # Create batches
        batches = []
        for i in range(0, total_chunks, storage_batch_size):
            batch_chunks = chunks[i:i + storage_batch_size]
            batch_num = (i // storage_batch_size) + 1
            batches.append((batch_num, batch_chunks))
        
        total_batches = len(batches)
        print(f"📦 Split into {total_batches} batches")
        
        # Process batches concurrently
        start_time = time.time()
        completed_batches = 0
        
        print(f"\n🚀 Starting concurrent processing...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches for concurrent processing
            future_to_batch = {
                executor.submit(self.process_batch_worker, batch_data): batch_data[0]
                for batch_data in batches
            }
            
            # Process completed batches
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                result = future.result()
                completed_batches += 1
                
                # Update progress
                progress_percent = (completed_batches / total_batches) * 100
                
                if result["success"]:
                    dims = result.get("embedding_dims", "?")
                    print(f"   ✅ Batch {result['batch_num']}/{total_batches} "
                          f"({progress_percent:.1f}%) - "
                          f"Stored {result['processed']}/{result['total_chunks']} chunks "
                          f"in {result['time']:.2f}s ({dims}D)")
                else:
                    print(f"   ❌ Batch {result['batch_num']}/{total_batches} "
                          f"({progress_percent:.1f}%) - "
                          f"FAILED: {result.get('error', 'Unknown error')}")
                
                # Periodic garbage collection to manage memory
                if completed_batches % 10 == 0:
                    gc.collect()
        
        total_time = time.time() - start_time
        
        # Final statistics
        print(f"\n📊 Loading Summary:")
        print(f"   ⏱️ Total time: {total_time:.2f} seconds")
        print(f"   ⚡ Processing speed: {total_chunks/total_time:.1f} chunks/second")
        print(f"   ✅ Successfully loaded: {self.successful_loads}")
        print(f"   ❌ Failed to load: {self.failed_loads}")
        print(f"   📈 Success rate: {(self.successful_loads/total_chunks)*100:.1f}%")
        print(f"   🗄️ Collection: {self.collection_name}")
        print(f"   💾 Storage: {self.db_path}")
        
        return self.successful_loads, self.failed_loads
    



def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python vector_load.py <chunks_file.json> [options]")
        print("\nOptions:")
        print("  --model=MODEL            Embedding model (default: all-mpnet-base-v2)")
        print("  --db-path=PATH           Database path (default: ./chroma_db)")
        print("  --workers=N              Number of worker threads (default: 4)")
        print("  --batch-size=N           Chunks per storage batch (default: 100)")
        print("  --embedding-batch=N      Embedding batch size (default: 32)")
        print("\nExamples:")
        print("  python vector_load.py chunks.json")
        print("  python vector_load.py chunks.json --workers=8 --batch-size=200")
        print("  python vector_load.py chunks.json --model=all-MiniLM-L6-v2")
        sys.exit(1)
    
    chunks_file = sys.argv[1]
    
    # Parse arguments
    model_name = "all-mpnet-base-v2"  # Default
    db_path = "./chroma_db"  # Default
    max_workers = 4  # Default
    storage_batch_size = 100  # Default
    embedding_batch_size = 32  # Default
    
    for arg in sys.argv[2:]:
        if arg.startswith("--model="):
            model_name = arg.split("=", 1)[1]
        elif arg.startswith("--db-path="):
            db_path = arg.split("=", 1)[1]
        elif arg.startswith("--workers="):
            max_workers = int(arg.split("=", 1)[1])
        elif arg.startswith("--batch-size="):
            storage_batch_size = int(arg.split("=", 1)[1])
        elif arg.startswith("--embedding-batch="):
            embedding_batch_size = int(arg.split("=", 1)[1])
    
    print(f"🚀 Concurrent Vector Load Starting")
    print(f"📁 Chunks file: {chunks_file}")
    print(f"🤖 Model: {model_name}")
    print(f"💾 Database path: {db_path}")
    print(f"👥 Workers: {max_workers}")
    print(f"📦 Storage batch size: {storage_batch_size}")
    print(f"🔢 Embedding batch size: {embedding_batch_size}")
    print("=" * 60)
    
    # Verify input file exists
    if not Path(chunks_file).exists():
        print(f"❌ Error: File not found: {chunks_file}")
        sys.exit(1)
    
    # Initialize loader and load data
    try:
        loader = ConcurrentVectorLoader(
            model_name=model_name, 
            db_path=db_path,
            max_workers=max_workers,
            embedding_batch_size=embedding_batch_size
        )
        
        successful, failed = loader.load_chunks(chunks_file, storage_batch_size)
        
        print(f"\n🎯 Ready for search! Use:")
        print(f"   python vector_search.py 'your query' --model={model_name} --db-path={db_path}")
        
    except Exception as e:
        print(f"❌ Error during loading: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
