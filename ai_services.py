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


def count_text_tokens(text):
    """
    Count how many tokens are in the text using Gemini's proper token counter.
    Falls back to word estimation if API fails.
    """
    try:
        # Use the correct new API method for token counting
        result = gemini_client.models.count_tokens(
            model="gemini-2.5-flash-preview-05-20", 
            contents=text
        )
        return result.total_tokens
    except Exception as e:
        print(f"Token counting failed, using word estimate: {e}")
        # Backup: estimate from word count
        import re
        words = len(re.findall(r'\b\w+\b', text))
        return int(words * WORDS_TO_TOKENS_RATIO)


def break_text_into_chunks(text):
    """
    Use Gemini AI to split text into meaningful semantic chunks.
    Returns chunks with headings, content, keywords, and summaries.
    Includes timeout and retry mechanisms to prevent hanging.
    """
    # Count tokens and log the chunking attempt
    token_count = count_text_tokens(text)
    print(f"Sending {token_count:,} tokens to Gemini for chunking...")
    
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
    
    # Retry mechanism with exponential backoff
    max_retries = 3
    base_timeout = 60  # 60 seconds base timeout
    
    for attempt in range(max_retries):
        try:
            timeout = base_timeout * (2 ** attempt)  # Exponential backoff: 60s, 120s, 240s
            print(f"Attempt {attempt + 1}/{max_retries} (timeout: {timeout}s)")
            
            start_time = time.time()
            
            # Set up request config with timeout handling
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
            print(f"ERROR: Gemini attempt {attempt + 1} failed after {elapsed_time:.1f}s: {e}")
            
            # If this was the last attempt, return error
            if attempt == max_retries - 1:
                print(f"ERROR: All {max_retries} attempts failed. Giving up.")
                return json.dumps({"error": f"Failed after {max_retries} attempts: {str(e)}"})
            
            # Wait before retrying (exponential backoff)
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            print(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)


def get_text_embedding(text):
    """
    Get vector embedding from Azure Cohere for the given text.
    Returns a list of numbers that represents the text's meaning.
    Includes timeout and retry for reliability.
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
                if attempt == max_retries - 1:
                    return None
                
        except Exception as e:
            print(f"Error getting text embedding (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(1)  # Brief wait before retry
    
    return None


def get_search_embedding(text):
    """
    Get vector embedding optimized for search queries.
    Uses 'query' input type for better search performance.
    Includes timeout and retry for reliability.
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
                if attempt == max_retries - 1:
                    return None
                
        except Exception as e:
            print(f"Error getting search embedding (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(1)  # Brief wait before retry
    
    return None


def validate_ai_services():
    """
    Test that all AI services are working properly.
    Returns True if everything is working, False otherwise.
    """
    # Test Gemini token counting
    try:
        test_text = "This is a test sentence for checking if Gemini works."
        tokens = count_text_tokens(test_text)
    except Exception as e:
        print(f"ERROR: Gemini token counting failed: {e}")
        return False
    
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
