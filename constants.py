# Constants for the Semantic Chunker Application

# Word limits and chunk sizes (optimized for performance)
DEFAULT_CHUNK_SIZE = 10000  # Larger working chunks for efficiency 
MIN_CHUNK_REFILL_SIZE = 5000  # Higher threshold to reduce iterations 
MAX_SAFE_GEMINI_WORDS = 8000  # Increased to handle larger chunks 

# Embedding and vector settings
COHERE_VECTOR_DIMENSIONS = 1536  # Cohere embed v4 vector size
QDRANT_COLLECTION_NAME = "semantic_chunks"

# HNSW optimization parameters
HNSW_M_VALUE = 64  # Edges per node
HNSW_EF_CONSTRUCT = 512  # Higher quality index construction
DEFAULT_SEGMENTS = 8  # Match CPU cores for parallel processing

# Gemini AI settings
GEMINI_TEMPERATURE = 0.1
GEMINI_RESPONSE_TYPE = "application/json"


