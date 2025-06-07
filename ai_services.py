import json
import requests
import time
from google import genai
from config import Config
from constants import (
    GEMINI_TEMPERATURE, 
    GEMINI_RESPONSE_TYPE, 
    WORDS_TO_TOKENS_RATIO
)


# Initialize Gemini AI with new client-based API
gemini_client = genai.Client(api_key=Config.GEMINI_API_KEY)


def break_text_into_chunks(text):
    """
    Use Gemini AI to split text into meaningful semantic chunks.
    Returns chunks with headings, content, keywords, and summaries.
    Only retries on timeout (no response in 30 seconds).
    """
    prompt = """

You are an expert in semantic text segmentation. Your mission is to break down text into SUBSTANTIAL, meaningful chunks that are thematically complete and independent. 

CRITICAL PERFORMANCE REQUIREMENTS:
1. Create FEWER, LARGER chunks (aim for 500-2000 words per chunk when possible)
2. Avoid micro-chunking - do not split every paragraph into separate chunks
3. Prioritize efficiency - fewer chunks mean better performance

Core Directive: Create Substantial Thematic Units

Only create a new chunk when there is a MAJOR thematic shift or topic change. Minor topic variations, examples, elaborations, and related subtopics should stay together in the same chunk.

Guiding Principles:
1. Consolidate Related Content: Group multiple related paragraphs, sections, and concepts into single comprehensive chunks.
2. Structural Unity: Keep structurally connected elements together:
   - Introduction + body + conclusion of a topic
   - Concept + examples + applications
   - Problem + analysis + solution
   - Process steps that work together
3. Avoid Over-Segmentation: Strongly favor larger chunks. Only split when absolutely necessary for thematic coherence.
4. Preserve Original Text: Copy content exactly as written, no modifications.
5. Target Chunk Size: Aim for substantial chunks of 500-2000 words when the content allows it.

Output Structure:
Heading: A comprehensive title that covers the entire chunk's scope
Content: The exact, unaltered original text (favor longer sections)
Keywords: 7-10 key terms representing the chunk's main concepts
Summary: A 2-3 sentence overview of the chunk's complete content

Text to process:
""" + text
    
    # Optimized retry settings for better performance
    max_retries = 2  # Fewer retries for faster failure
    timeout = 20     # Shorter timeout for faster response
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            # Set up request config
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=prompt,
                config={
                    "temperature": GEMINI_TEMPERATURE,
                    "response_mime_type": GEMINI_RESPONSE_TYPE,
                    "response_schema": {
                        "type": "object",
                        "properties": {
                            "chunks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "heading": {"type": "string"},
                                        "content": {"type": "string"},
                                        "keywords": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        },
                                        "summary": {"type": "string"}
                                    },
                                    "required": ["heading", "content", "keywords", "summary"]
                                }
                            }
                        },
                        "required": ["chunks"]
                    }
                }
            )
            
            # Parse and validate the results
            try:
                result = json.loads(response.text)
                if "chunks" not in result:
                    return json.dumps({"error": "Response missing chunks array"})
            except json.JSONDecodeError:
                return json.dumps({"error": "Response not in JSON format"})
            
            return response.text
            
        except Exception as e:
            elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
            error_message = str(e).lower()
            
            # Only retry on timeout-related errors
            is_timeout = any(keyword in error_message for keyword in [
                'timeout', 'timed out', 'deadline exceeded', 'connection timeout',
                'read timeout', 'request timeout'
            ])
            
            if is_timeout:
                # If this was the last attempt, return error
                if attempt == max_retries - 1:
                    return json.dumps({"error": f"Timeout after {max_retries} attempts: {str(e)}"})
                
                # Wait before retrying
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait_time)
            else:
                # Non-timeout error - don't retry, return immediately
                return json.dumps({"error": f"API error: {str(e)}"})


def get_text_embedding(text):
    """
    Get vector embedding from Azure Cohere for the given text.
    Returns a list of numbers that represents the text's meaning.
    Only retries on timeout errors.
    """
    max_retries = 2
    timeout = 30  # 30 second timeout
    
    for attempt in range(max_retries):
        try:
            headers = Config.get_cohere_headers()
            
            payload = {
                "model": Config.COHERE_MODEL_NAME,
                "input": [text],
                "input_type": "document"
            }
            
            response = requests.post(
                Config.get_cohere_endpoint_url(),
                headers=headers,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["data"][0]["embedding"]
            else:
                return None  # Don't retry on API errors, only timeouts
                
        except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
            if attempt == max_retries - 1:
                return None
            time.sleep(1)  # Brief wait before retry
        except Exception as e:
            return None  # Don't retry on non-timeout errors
    
    return None


def get_search_embedding(text):
    """
    Get vector embedding optimized for search queries.
    Uses 'query' input type for better search performance.
    Only retries on timeout errors.
    """
    max_retries = 2
    timeout = 30  # 30 second timeout
    
    for attempt in range(max_retries):
        try:
            headers = Config.get_cohere_headers()
            
            payload = {
                "model": Config.COHERE_MODEL_NAME,
                "input": [text],
                "input_type": "query"  # Correct input type for queries
            }
            
            response = requests.post(
                Config.get_cohere_endpoint_url(),
                headers=headers,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["data"][0]["embedding"]
            else:
                return None  # Don't retry on API errors, only timeouts
                
        except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
            if attempt == max_retries - 1:
                return None
            time.sleep(1)  # Brief wait before retry
        except Exception as e:
            return None  # Don't retry on non-timeout errors
    
    return None


def validate_ai_services():
    """
    Test that all AI services are working properly.
    Returns True if everything is working, False otherwise.
    """
    # Test Gemini chunking
    try:
        test_chunk = break_text_into_chunks("This is a test. This is another test sentence.")
        if "error" in test_chunk:
            print(f"ERROR: Gemini chunking failed: {test_chunk}")
            return False
    except Exception as e:
        print(f"ERROR: Gemini chunking failed: {e}")
        return False
    
    # Test Cohere embeddings
    try:
        embedding = get_text_embedding("test text")
        if not embedding or len(embedding) == 0:
            print("ERROR: Cohere embeddings failed")
            return False
    except Exception as e:
        print(f"ERROR: Cohere embeddings failed: {e}")
        return False
    
    return True
