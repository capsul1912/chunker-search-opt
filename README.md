# Semantic Chunker & Search

A FastAPI application that breaks text into meaningful chunks using AI and provides semantic search with vector embeddings.

## What It Does

This application takes any text document and splits it into semantic chunks that preserve meaning. It then stores these chunks in a vector database so you can search through them later.

## Key Features

- **Smart Chunking**: Uses Gemini AI to create meaningful chunks instead of arbitrary splits
- **Handles Any Size**: Works with small documents (10+ words) and large documents (100k+ words)
- **Fast Processing**: Optimized for performance with efficient chunking strategies
- **Vector Search**: Uses Cohere embeddings and Qdrant database for semantic search
- **Web Interface**: Simple web UI for testing and usage

## How It Works

### For Small Documents (under 10k words)
- Sends the entire document to AI for semantic analysis
- Creates 1-3 meaningful chunks with proper headings, keywords, and summaries

### For Large Documents (over 10k words)
- Processes in 10k word working chunks to avoid memory issues
- Extracts semantic chunks iteratively
- Refills working chunks when they get below 5k words
- Never loads the entire document into memory at once

### Performance
- Small docs: Process in 5-15 seconds
- Large docs: About 100-150 words per second
- Creates substantial chunks (500-2000 words each) instead of tiny fragments

## Setup

1. **Install Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create a `.env` file with your API keys:**
   ```env
   # Required: Gemini AI for chunking
   GEMINI_API_KEY=your_gemini_api_key

   # Required: Azure Cohere for embeddings
   AZURE_COHERE_API_KEY=your_azure_cohere_key
   AZURE_COHERE_ENDPOINT=your_azure_endpoint
   COHERE_MODEL_NAME=cohere-embed-v-4

   # Required: Qdrant for vector storage
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key

   # Optional: App settings
   APP_HOST=0.0.0.0
   APP_PORT=8000
   DEBUG_MODE=true
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open in browser:**
   ```
   http://localhost:8000
   ```

## API Endpoints

- **GET /** - Web interface for testing
- **POST /chunk** - Submit text to be chunked and stored
- **POST /search** - Search for similar chunks
- **POST /embed-and-store** - Store pre-processed chunks
- **GET /health** - Check if all services are working

## File Structure

```
chunker/
├── app.py                 # Main FastAPI application
├── config.py              # Environment variable handling
├── constants.py           # Configuration settings
├── ai_services.py         # Gemini and Cohere API calls
├── text_tools.py          # Text processing and chunking logic
├── vector_database.py     # Qdrant database operations
├── requirements.txt       # Python dependencies
├── templates/index.html   # Web interface
└── .env                   # Your API keys (create this)
```

## Main Functions

### Text Processing
- **process_large_text()** - Main chunking function for any document size
- **count_words()** - Count words in text
- **clean_text_for_processing()** - Prepare text for processing

### AI Services
- **break_text_into_chunks()** - Send text to Gemini for semantic chunking
- **get_text_embedding()** - Get vector embeddings from Cohere
- **get_search_embedding()** - Get embeddings optimized for search queries

### Database Operations
- **save_chunks_to_database()** - Store chunks with embeddings in Qdrant
- **search_similar_chunks()** - Find similar chunks using hybrid search
- **setup_vector_database()** - Initialize the database collection

## Configuration

Key settings you can adjust in `constants.py`:
- **DEFAULT_CHUNK_SIZE**: 10,000 words (working chunk size)
- **MIN_CHUNK_REFILL_SIZE**: 5,000 words (when to refill working chunk)
- **COHERE_VECTOR_DIMENSIONS**: 1536 (Cohere Embed v4 vector size)

## Troubleshooting

- Use **GET /health** to check if all services are connected
- Check the console output for detailed processing logs
- Make sure all API keys are valid and have proper permissions
- Qdrant collection is created automatically on first run

## Performance Notes

The system is optimized for both speed and quality:
- Creates fewer, larger chunks instead of many tiny ones
- Processes documents efficiently without loading everything into memory
- Uses hybrid search combining semantic similarity and keyword matching
- Handles documents from 10 words to 100k+ words effectively 