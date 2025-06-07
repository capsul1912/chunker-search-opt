# Constants for the Semantic Chunker Application

# Token limits and chunk sizes
DEFAULT_CHUNK_SIZE = 10000  # Default working chunk size in tokens
MIN_CHUNK_REFILL_SIZE = 5000  # Stop chunking and refill when below this size

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

# Token estimation fallback (when API fails)
WORDS_TO_TOKENS_RATIO = 1.33  # 1 token = approx 0.75 words
