import uuid
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    SparseVectorParams, SparseIndexParams, 
    Modifier, Document, SearchParams,
    Prefetch, FusionQuery, Fusion
)

from config import Config
from constants import (
    COHERE_VECTOR_DIMENSIONS, QDRANT_COLLECTION_NAME,
    HNSW_M_VALUE, HNSW_EF_CONSTRUCT, DEFAULT_SEGMENTS
)
from ai_services import get_text_embedding, get_search_embedding, count_text_tokens


# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=Config.QDRANT_URL,
    api_key=Config.QDRANT_API_KEY,
)


def setup_vector_database():
    """
    Set up the Qdrant vector database with optimized settings.
    Creates collection if it doesn't exist or updates it if needed.
    """
    try:
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if QDRANT_COLLECTION_NAME in collection_names:
            # Check if existing collection has the right setup
            if _check_collection_setup():
                # Collection exists and is properly configured, no need to optimize again
                pass
            else:
                qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
                _create_optimized_collection()
        else:
            _create_optimized_collection()
            
    except Exception as e:
        print(f"ERROR: Error setting up vector database: {e}")
        # Try creating a basic collection as fallback
        _create_basic_collection()


def _check_collection_setup():
    """Check if the collection has the right configuration"""
    try:
        collection_info = qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        
        # Check if we have dense vectors with right dimensions
        has_dense = hasattr(collection_info.config.params, 'vectors')
        if has_dense:
            if hasattr(collection_info.config.params.vectors, 'size'):
                dense_size_ok = collection_info.config.params.vectors.size == COHERE_VECTOR_DIMENSIONS
            else:
                dense_vector = collection_info.config.params.vectors.get('dense')
                dense_size_ok = dense_vector and dense_vector.size == COHERE_VECTOR_DIMENSIONS
        else:
            dense_size_ok = False
        
        # Check if we have sparse vectors
        has_sparse = hasattr(collection_info.config.params, 'sparse_vectors')
        sparse_ok = has_sparse and collection_info.config.params.sparse_vectors
        
        return dense_size_ok and sparse_ok
        
    except Exception as e:
        print(f"Error checking collection setup: {e}")
        return False


def _create_optimized_collection():
    """Create a new collection with optimal settings for performance"""
    try:
        # Optimized settings for best performance
        hnsw_config = models.HnswConfigDiff(
            m=HNSW_M_VALUE,                # More connections for better accuracy
            ef_construct=HNSW_EF_CONSTRUCT, # Better index quality
            on_disk=False,                  # Keep in memory for speed
        )
        
        optimizer_config = models.OptimizersConfigDiff(
            default_segment_number=DEFAULT_SEGMENTS,  # Parallel processing
        )
        
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=COHERE_VECTOR_DIMENSIONS, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                    modifier=Modifier.IDF  # Built-in BM25-like scoring
                )
            },
            hnsw_config=hnsw_config,
            optimizers_config=optimizer_config
        )
        print(f"Created optimized vector database: {QDRANT_COLLECTION_NAME}")
        
    except Exception as e:
        print(f"ERROR: Error creating optimized collection: {e}")
        _create_basic_collection()


def _create_basic_collection():
    """Create a basic collection as fallback"""
    try:
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=COHERE_VECTOR_DIMENSIONS, distance=Distance.COSINE)
        )
        print(f"Created basic vector database: {QDRANT_COLLECTION_NAME}")
    except Exception as e:
        print(f"ERROR: Failed to create even basic collection: {e}")


def _apply_performance_optimizations():
    """Apply performance optimizations to existing collection"""
    try:
        hnsw_config = models.HnswConfigDiff(
            m=HNSW_M_VALUE,
            ef_construct=HNSW_EF_CONSTRUCT,
            on_disk=False,
        )
        
        optimizer_config = models.OptimizersConfigDiff(
            default_segment_number=DEFAULT_SEGMENTS,
        )
        
        qdrant_client.update_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            hnsw_config=hnsw_config,
            optimizers_config=optimizer_config
        )
        pass
        
    except Exception as e:
        print(f"ERROR: Could not apply optimizations: {e}")


def save_chunks_to_database(chunks, document_id=None):
    """
    Save text chunks to the vector database with embeddings.
    Returns the document ID if successful, None if failed.
    """
    if document_id is None:
        document_id = str(uuid.uuid4())
    
    points = []
    
    for i, chunk in enumerate(chunks):
        try:
            # Get vector embedding for this chunk
            dense_embedding = get_text_embedding(chunk["content"])
            
            if dense_embedding is None:
                print(f"WARNING: Failed to get embedding for chunk {i}")
                continue
            
            # Get sparse embedding using Qdrant's built-in BM25
            sparse_document = _get_sparse_embedding(chunk["content"])
            
            # Check if we have hybrid search support
            has_sparse_support = _has_sparse_support()
            
            # Prepare vector data
            if has_sparse_support and sparse_document:
                vector_data = {
                    "dense": dense_embedding,
                    "sparse": sparse_document
                }
            else:
                vector_data = dense_embedding
            
            # Create point for storage
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector_data,
                payload={
                    "document_id": document_id,
                    "chunk_index": i,
                    "heading": chunk.get("heading", ""),
                    "content": chunk["content"],
                    "keywords": chunk.get("keywords", []),
                    "summary": chunk.get("summary", ""),
                    "token_count": count_text_tokens(chunk["content"])
                }
            )
            points.append(point)
            
        except Exception as e:
            print(f"ERROR: Error processing chunk {i}: {e}")
            continue
    
    # Save all points to database
    if points:
        try:
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=points
            )
            return document_id
        except Exception as e:
            print(f"ERROR: Error saving chunks to database: {e}")
            return None
    else:
        print("WARNING: No valid chunks to save")
        return None


def search_similar_chunks(query, limit=5):
    """
    Search for chunks similar to the query using smart hybrid search.
    Returns list of matching chunks with scores.
    """
    try:
        # Get embeddings for the search query
        query_dense_embedding = get_search_embedding(query)
        
        if query_dense_embedding is None:
            print("ERROR: Failed to get query embedding")
            return {"results": [], "search_method": "failed"}
        
        # Use simple search parameters
        search_params = SearchParams(hnsw_ef=128, exact=False)
        

        
        # Try hybrid search first, fallback to dense-only if needed
        if _has_sparse_support():
            results = _hybrid_search(query, query_dense_embedding, limit, search_params)
        else:
            results = _dense_search(query_dense_embedding, limit, search_params)
        
        return results
        
    except Exception as e:
        print(f"ERROR: Search failed: {e}")
        return {"results": [], "search_method": "error"}


def _get_sparse_embedding(text):
    """Get sparse embedding using Qdrant's built-in BM25"""
    try:
        return Document(text=text, model="Qdrant/bm25")
    except Exception as e:
        print(f"WARNING: Sparse embedding failed: {e}")
        return None


def _has_sparse_support():
    """Check if the collection supports sparse vectors"""
    try:
        collection_info = qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        return hasattr(collection_info.config.params, 'sparse_vectors') and collection_info.config.params.sparse_vectors
    except:
        return False





def _hybrid_search(query, query_dense_embedding, limit, search_params):
    """Perform hybrid search using both dense and sparse vectors"""
    try:
        query_sparse_document = _get_sparse_embedding(query)
        
        if not query_sparse_document:
            return _dense_search(query_dense_embedding, limit, search_params)
        
        # Calculate prefetch limits  
        prefetch_limit = max(limit * 3, limit + 10)
        
        search_results = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=query_sparse_document,
                    using="sparse",
                    limit=prefetch_limit,
                    params=search_params
                ),
                Prefetch(
                    query=query_dense_embedding,
                    using="dense",
                    limit=prefetch_limit,
                    params=search_params
                )
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True
        )
        
        results = _format_search_results(search_results.points)
        return {
            "results": results, 
            "search_method": "hybrid"
        }
        
    except Exception as e:
        print(f"WARNING: Hybrid search failed, trying dense search: {e}")
        return _dense_search(query_dense_embedding, limit, search_params)


def _dense_search(query_dense_embedding, limit, search_params):
    """Perform dense-only vector search"""
    try:
        search_results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_dense_embedding,
            limit=limit,
            with_payload=True,
            search_params=search_params
        )
        
        results = _format_search_results(search_results)
        return {
            "results": results, 
            "search_method": "dense-only"
        }
        
    except Exception as e:
        print(f"ERROR: Dense search failed: {e}")
        return {"results": [], "search_method": "failed"}


def _format_search_results(search_results):
    """Format search results into a consistent structure"""
    results = []
    for result in search_results:
        results.append({
            "score": result.score,
            "heading": result.payload.get("heading", ""),
            "content": result.payload.get("content", ""),
            "keywords": result.payload.get("keywords", []),
            "summary": result.payload.get("summary", ""),
            "document_id": result.payload.get("document_id", ""),
            "chunk_index": result.payload.get("chunk_index", 0),
            "token_count": result.payload.get("token_count", 0)
        })
    return results


def validate_vector_database():
    """Test that the vector database is working properly"""
    try:
        # Test connection
        collections = qdrant_client.get_collections()
        
        # Test collection exists
        if QDRANT_COLLECTION_NAME in [col.name for col in collections.collections]:
            return True
        else:
            print(f"WARNING: Collection '{QDRANT_COLLECTION_NAME}' not found")
            return False
            
    except Exception as e:
        print(f"ERROR: Vector database validation failed: {e}")
        return False
