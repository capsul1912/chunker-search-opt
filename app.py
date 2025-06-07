import json
import uvicorn
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# Import our organized modules
from config import Config
from constants import DEFAULT_CHUNK_SIZE, MAX_SAFE_GEMINI_WORDS
from ai_services import break_text_into_chunks, validate_ai_services
from text_tools import process_large_text, clean_text_for_processing, count_words
from vector_database import (
    setup_vector_database, 
    save_chunks_to_database, 
    search_similar_chunks,
    validate_vector_database
)

# Create FastAPI app
app = FastAPI(title="Semantic Chunker & Search")

# Add CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up HTML templates
templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
async def startup_event():
    """Initialize everything when the app starts"""
    # Validate configuration
    if not Config.validate_config():
        print("ERROR: Configuration validation failed")
        return
    
    # Test AI services
    if not validate_ai_services():
        print("ERROR: AI services validation failed")
        return
    
    # Set up vector database
    setup_vector_database()
    
    # Test vector database
    if not validate_vector_database():
        print("WARNING: Vector database validation failed")


@app.get("/", response_class=HTMLResponse)
async def show_homepage(request: Request):
    """Show the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post('/chunk')
async def chunk_text(text: str = Form(...)):
    """
    Break text into semantic chunks and store them in the database.
    Works with both small and large texts automatically.
    Now uses word counting instead of token counting.
    """
    try:
        # Clean up the input text
        cleaned_text = clean_text_for_processing(text)
        word_count = count_words(cleaned_text)
        
        # Use the safe limit for Gemini to prevent timeouts
        if word_count <= MAX_SAFE_GEMINI_WORDS:
            # Small text: use AI directly (up to 7.5k words)
            print(f"Processing {word_count:,} words directly with Gemini")
            semantic_result = break_text_into_chunks(cleaned_text)
            
            # Parse the AI response
            if isinstance(semantic_result, str):
                parsed_result = json.loads(semantic_result)
            else:
                parsed_result = semantic_result
            
            if "chunks" in parsed_result:
                # Save chunks to database
                document_id = save_chunks_to_database(parsed_result["chunks"])
                parsed_result["document_id"] = document_id
            
            return {"result": json.dumps(parsed_result)}
        else:
            # Large text: use dynamic chunking process
            print(f"Processing {word_count:,} words with dynamic chunking")
            result = process_large_text(cleaned_text)
            
            # Save chunks to database
            if "chunks" in result:
                document_id = save_chunks_to_database(result["chunks"])
                result["document_id"] = document_id
            
            return {"result": json.dumps(result)}
            
    except Exception as e:
        print(f"ERROR: Error in chunk_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/embed-and-store')
async def store_existing_chunks(chunks_json: str = Form(...)):
    """
    Take already-chunked text and store it in the database with embeddings.
    Useful for importing pre-processed content.
    """
    try:
        # Parse the chunks data
        if isinstance(chunks_json, str):
            chunks_data = json.loads(chunks_json)
        else:
            chunks_data = chunks_json
        
        # Extract chunks array from different possible formats
        if "chunks" in chunks_data:
            chunks = chunks_data["chunks"]
        elif isinstance(chunks_data, list):
            chunks = chunks_data
        else:
            raise HTTPException(status_code=400, detail="Invalid chunks format")
        
        # Save to database with embeddings
        document_id = save_chunks_to_database(chunks)
        
        if document_id:
            return {
                "success": True,
                "document_id": document_id,
                "chunks_stored": len(chunks),
                "message": f"Successfully stored {len(chunks)} chunks in database"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to store chunks in database")
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        print(f"ERROR: Error in store_existing_chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/search')
async def search_chunks(query: str = Form(...), limit: int = Form(5)):
    """
    Search for chunks similar to the query using smart hybrid search.
    Automatically optimizes search strategy based on query and result count.
    """
    try:
        # Clean up the search query
        cleaned_query = clean_text_for_processing(query)
        
        # Perform smart search
        search_result = search_similar_chunks(cleaned_query, limit)
        
        return search_result
        
    except Exception as e:
        print(f"ERROR: Error in search_chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/health')
async def health_check():
    """Check if the application and all services are working"""
    try:
        health_status = {
            "app": "running",
            "config": Config.validate_config(),
            "ai_services": validate_ai_services(),
            "vector_database": validate_vector_database()
        }
        
        overall_health = all(health_status.values())
        
        if overall_health:
            return {"status": "healthy", "details": health_status}
        else:
            return {"status": "unhealthy", "details": health_status}
            
    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == '__main__':
    # Run the application
    uvicorn.run(
        "app:app", 
        host=Config.APP_HOST, 
        port=Config.APP_PORT, 
        reload=Config.DEBUG_MODE
    ) 