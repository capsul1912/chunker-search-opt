import json
import requests
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
    """
    # Count tokens and log the chunking attempt
    token_count = count_text_tokens(text)
    
    prompt = """
You are an expert at breaking documents into meaningful pieces. Your job is to split the text below into chunks that make sense together - like complete ideas, topics, or concepts.

IMPORTANT RULES:
1. Keep the EXACT original text in each chunk - don't change or summarize anything
2. Each chunk should be a complete thought or topic
3. Split at natural topic changes, not random places
4. Make chunks of reasonable size, but meaning is more important than size
5. Keep related examples and explanations together
6. Don't split closely related concepts across different chunks

For each chunk, provide:
- Heading: A clear title that describes what this chunk is about
- Content: The exact original text from the document
- Keywords: 7-10 important words for searching
- Summary: A brief 1-2 sentence description of what this chunk contains

Text to process:
""" + text
    
    try:
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
                print("WARNING: Gemini: Response missing chunks array")
        except json.JSONDecodeError:
            print("WARNING: Gemini: Response not in JSON format")
        
        return response.text
        
    except Exception as e:
        print(f"ERROR: Gemini: AI chunking failed - {e}")
        return json.dumps({"error": str(e)})


def get_text_embedding(text):
    """
    Get vector embedding from Azure Cohere for the given text.
    Returns a list of numbers that represents the text's meaning.
    """
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
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["data"][0]["embedding"]
        else:
            print(f"Cohere API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error getting text embedding: {e}")
        return None


def get_search_embedding(text):
    """
    Get vector embedding optimized for search queries.
    Uses 'query' input type for better search performance.
    """
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
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["data"][0]["embedding"]
        else:
            print(f"Cohere search embedding error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error getting search embedding: {e}")
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
