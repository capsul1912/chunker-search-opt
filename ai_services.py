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
    from text_tools import count_words
    
    # Count words and log the chunking attempt
    word_count = count_words(text)
    print(f"Sending {word_count:,} words to Gemini for chunking...")
    
    prompt = """
    You are an expert in semantic text segmentation. Your primary mission is to break down a document into its fundamental, self-contained units of meaning.
The guiding principle is thematic coherence. A chunk represents a single, complete topic or concept. You should create a new chunk only when the text transitions to a new, distinct topic.

Guiding Principles:

Keep the EXACT original text in each chunk - don't change or summarize anything.
Focus on Thematic Coherence: The length of a chunk is not important. What matters is that it contains one complete idea. A new chunk starts only when the topic changes.
Preserve Original Text: The Content for each chunk must be the exact original text. Do not alter, summarize, or omit anything.
Maintain Source Language: All generated output (Heading, Keywords, Summary) must be in the same language as the source text. Do not translate.
Group Related Elements: Keep closely related information together. For example, a concept and its examples, a problem and its solution, or sequential steps in a process should all be in the same chunk.

For each chunk you identify, provide the following structure:
Heading: A concise title that captures the core topic of the chunk.
Content: The exact, unaltered original text for this chunk.
Keywords: 7-10 important words or phrases from the chunk that are central to its meaning.
Summary: A brief 1-2 sentence description of the information presented in the chunk.

Text to process:
""" + text
    
    # Try the API call first without retry messaging
    max_retries = 3
    timeout = 30  # 30 seconds timeout
    
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
            
            elapsed_time = time.time() - start_time
            print(f"Gemini response received in {elapsed_time:.1f}s")
            
            # Parse and validate the results
            try:
                result = json.loads(response.text)
                if "chunks" not in result:
                    print("WARNING: Gemini: Response missing chunks array")
                else:
                    print(f"SUCCESS: Received {len(result['chunks'])} chunks from Gemini")
            except json.JSONDecodeError:
                print("WARNING: Gemini: Response not in JSON format")
            
            return response.text
            
        except Exception as e:
            elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
            error_message = str(e).lower()
            
            # Only retry on timeout-related errors
            is_timeout = any(keyword in error_message for keyword in [
                'timeout', 'timed out', 'deadline exceeded', 'connection timeout',
                'read timeout', 'request timeout', 'no response'
            ])
            
            if is_timeout:
                print(f"TIMEOUT: Gemini timed out after {elapsed_time:.1f}s: {e}")
                
                # If this was the last attempt, return error
                if attempt == max_retries - 1:
                    print(f"ERROR: All {max_retries} timeout attempts failed. Giving up.")
                    return json.dumps({"error": f"Timeout after {max_retries} attempts: {str(e)}"})
                
                # Show retry attempt info only when actually retrying
                print(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                print(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                # Non-timeout error - don't retry, return immediately
                print(f"ERROR: Gemini API error after {elapsed_time:.1f}s: {e}")
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
                print(f"Cohere API error: {response.status_code} - {response.text}")
                return None  # Don't retry on API errors, only timeouts
                
        except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
            print(f"TIMEOUT: Cohere embedding timed out: {e}")
            if attempt == max_retries - 1:
                print(f"ERROR: All {max_retries} timeout attempts failed.")
                return None
            print(f"Retrying... (attempt {attempt + 2}/{max_retries})")
            time.sleep(1)  # Brief wait before retry
        except Exception as e:
            print(f"ERROR: Cohere embedding API error: {e}")
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
                print(f"Cohere search embedding error: {response.status_code} - {response.text}")
                return None  # Don't retry on API errors, only timeouts
                
        except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
            print(f"TIMEOUT: Cohere search embedding timed out: {e}")
            if attempt == max_retries - 1:
                print(f"ERROR: All {max_retries} timeout attempts failed.")
                return None
            print(f"Retrying... (attempt {attempt + 2}/{max_retries})")
            time.sleep(1)  # Brief wait before retry
        except Exception as e:
            print(f"ERROR: Cohere search embedding API error: {e}")
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
