#!/usr/bin/env python3
"""
Basic Vector Search - Simple function to test database queries
"""

import chromadb
from sentence_transformers import SentenceTransformer

def basic_search(query, n_results=10, model_name="all-mpnet-base-v2", db_path="./chroma_db"):
    """
    Basic search function - no filters, no fancy processing
    Just query the database and return raw results
    """
    print(f"🔍 Basic search for: '{query}'")
    print(f"🤖 Using model: {model_name}")
    
    # Load model and connect to DB
    model = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=db_path)
    
    # Get collection
    collection_name = f"codebase_{model_name.replace('-', '_').replace('.', '_')}"
    collection = client.get_collection(collection_name)
    
    # Check database
    total_count = collection.count()
    print(f"📊 Database has {total_count} chunks")
    
    if total_count == 0:
        print("❌ Database is empty!")
        return []
    
    # Encode query
    query_embedding = model.encode([query])[0].tolist()
    print(f"🔢 Query embedding dimensions: {len(query_embedding)}")
    
    # Raw query - no filters, just search everything
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, total_count),  # Don't ask for more than available
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"✅ Found {len(results['ids'][0])} results")
        
        # Convert to simple format
        simple_results = []
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            similarity = 1 - distance
            
            simple_results.append({
                'id': results['ids'][0][i],
                'similarity': similarity,
                'distance': distance,
                'metadata': results['metadatas'][0][i],
                'document': results['documents'][0][i]
            })
        
        return simple_results
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return []


def print_results(results, show_full_doc=False):
    """Print results in a simple format"""
    if not results:
        print("No results to display")
        return
    
    print(f"\n📋 Results ({len(results)} found):")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        
        print(f"\n{i}. ID: {result['id']}")
        print(f"   Similarity: {result['similarity']:.3f}")
        print(f"   Distance: {result['distance']:.3f}")
        print(f"   Type: {metadata.get('chunk_type', 'unknown')}")
        print(f"   Language: {metadata.get('language', 'unknown')}")
        print(f"   Name: {metadata.get('name', 'unknown')}")
        print(f"   File: {metadata.get('filepath', 'unknown')}")
        
        if show_full_doc:
            doc = result['document']
            if len(doc) > 300:
                doc = doc[:297] + "..."
            print(f"   Document: {doc}")
        
        print("-" * 40)


def test_searches():
    """Test with various queries"""
    test_queries = [
        "function",
        "websocket",
        "auth",
        "test",
        "async",
        "vendor",
        "createWebSocketConn"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        results = basic_search(query, n_results=3)
        print_results(results)
        
        if results:
            print(f"Best match similarity: {results[0]['similarity']:.3f}")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python basic_search.py 'query text' [n_results]")
        print("  python basic_search.py test  # Run test searches")
        print("\nExamples:")
        print("  python basic_search.py 'websocket' 5")
        print("  python basic_search.py 'function' 10")
        sys.exit(1)
    
    if sys.argv[1] == "test":
        test_searches()
    else:
        query = sys.argv[1]
        n_results = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        
        results = basic_search(query, n_results)
        print_results(results, show_full_doc=True)