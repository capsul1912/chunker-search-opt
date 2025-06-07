# Semantic Chunker & Search

A FastAPI application that breaks large texts into meaningful chunks using AI and provides semantic search capabilities with vector embeddings.

## Can It Handle Large Documents?

**YES. This system can handle documents of any size, including 100k+ tokens.**

### How Large Document Processing Works

- **Memory Efficient**: Only loads 10,000 tokens in memory at once, not the entire document
- **Dynamic Chunking**: Splits large documents into working chunks automatically
- **No Size Limit**: Can process 100k, 200k, or larger documents without issues
- **Smart Refill**: When working chunk gets small (under 5,000 tokens), refills from remaining text
- **Error Safe**: If any chunk fails, saves it and continues with the rest

### Processing Example for 100k Tokens
- **Iterations needed**: ~10 
- **Memory usage**: Maximum 10,000 tokens at once
- **Output**: ~66 semantic chunks
- **Time**: ~2-3 minutes
- **AI calls**: ~66 requests to Gemini

## Features

- **Text Chunking**: Uses Gemini to split documents into semantic chunks
- **Large Document Support**: Handles documents of any size efficiently
- **Vector Search**: Azure Cohere Embed v4 and Qdrant vector database
- **Hybrid Search**: Combines dense and sparse vectors for optimal search results

## 📁 Project Structure

```
chunker/
├── app.py                 # Main FastAPI application 
├── config.py              # Configuration and environment variables
├── constants.py           # Important numbers and settings
├── ai_services.py         # Gemini and Cohere functions
├── text_tools.py          # Text processing and splitting utilities
├── vector_database.py     # Qdrant database operations
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Web interface
└── .env                   # Environment variables
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create `.env` file with your API keys:**
   ```env
   # Gemini
   GEMINI_API_KEY=your_gemini_api_key_here
   
   # Azure Cohere
   AZURE_COHERE_API_KEY=your_azure_cohere_key_here
   AZURE_COHERE_ENDPOINT=your_azure_endpoint_here
   COHERE_MODEL_NAME=embed-v-4-0
   
   # Qdrant Database
   QDRANT_URL=your_qdrant_url_here
   QDRANT_API_KEY=your_qdrant_api_key_here
   
   # Application Settings (optional)
   APP_HOST=0.0.0.0
   APP_PORT=8000
   DEBUG_MODE=true
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open your browser:**
   ```
   http://localhost:8000
   ```

## How It Works

### Text Chunking Process
1. **Small texts** (≤10k tokens): Processed directly with Gemini
2. **Large texts** (>10k tokens): Uses dynamic chunking:
   - Maintains 10k token working chunks
   - Extracts semantic chunks one by one
   - Refills working chunk when it gets below 5k tokens
   - Never loads entire document in memory

### Search Process
1. **Hybrid Search**: Combines dense vectors (meaning) + sparse vectors (keywords)
2. **Simple Parameters**: Uses fixed settings for consistent performance
3. **Fusion Ranking**: Uses Reciprocal Rank Fusion for best results

## API Endpoints

- `GET /` - Web interface
- `POST /chunk` - Process and chunk text
- `POST /search` - Search for similar chunks
- `POST /embed-and-store` - Store pre-chunked content
- `GET /health` - Health check for all services

## Key Functions

### AI Services (`ai_services.py`)
- `count_text_tokens()` - Count tokens in text
- `break_text_into_chunks()` - Semantic chunking with Gemini
- `get_text_embedding()` - Get vector embeddings from Cohere
- `validate_ai_services()` - Test connections

### Text Tools (`text_tools.py`)
- `split_text_by_tokens()` - Split text by exact token count
- `process_large_text()` - Dynamic chunking for large documents
- `clean_text_for_processing()` - Prepare text for processing

### Vector Database (`vector_database.py`)
- `setup_vector_database()` - Initialize Qdrant collection
- `save_chunks_to_database()` - Store chunks with embeddings
- `search_similar_chunks()` - Hybrid search with dense + sparse vectors
- `validate_vector_database()` - Test database connection

## Configuration

Key settings in `constants.py`:
- **Working chunk size**: 10,000 tokens
- **Refill threshold**: 5,000 tokens
- **Vector dimensions**: 1536 (Cohere Embed v4)

The system automatically validates all services on startup and shows clear error messages for any issues.

Use `GET /health` endpoint to check service status. 