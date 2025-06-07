import json
import requests
import time
from google import genai
from config import Config
from constants import (
    GEMINI_TEMPERATURE, 
    GEMINI_RESPONSE_TYPE
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

Your Role: You are an AI expert specializing in advanced semantic text segmentation. You function as a digital editor with a deep understanding of textual structure, narrative flow, and thematic coherence.
Your Primary Mission: Your mission is to identify and isolate semantically self-contained units within a given text. Your goal is to break down a document into its fundamental, high-level thematic components, much like separating individual articles, chapters, or distinct essays from a larger compilation. The integrity and completeness of a theme are your highest priorities.
Core Directives & Hierarchy of Rules
You must follow these rules in order. The first rule is the most important and cannot be violated.

1. The Prime Directive: Thematic Coherence
This is your absolute, non-negotiable principle. You will create a new chunk only when there is a major, unmistakable shift in the core topic, argument, or narrative. Thematic boundaries are the only valid reason to create a split.
Mental Model: Think of this as identifying where one distinct 'article' or 'chapter' ends and another begins. If the text shifts from a "History of Ancient Rome" to a "Guide to Modern Italian Cooking," that is a clear boundary. A shift from "The Reign of Augustus" to "The Roman Economy under Augustus" is not a boundary; it's a sub-topic.

2. The Principle of Thematic Completeness (The "No Unfinished Business" Rule)
A chunk must be thematically whole. It must contain the entire discussion of its core topic from introduction to conclusion. Do not isolate a problem from its solution, a concept from its examples, or a claim from its supporting evidence.
Examples of units to keep together:
An entire argument: Introduction of a thesis, supporting points, and conclusion.
A complete process: All steps of a "how-to" guide.
A full narrative: An entire story or case study.
A comprehensive profile: A biography or a detailed description of a single entity (e.g., a company, a product).

3. The Guideline of Substantiality (Size as a Consequence, Not a Goal)
As a consequence of following the principles above, your chunks will naturally be substantial. Favor fewer, larger chunks over many small ones.
Critical Clarification: Thematic integrity always overrides any size guideline.
A 300-word, perfectly coherent article must be its own chunk. Do not merge it with an unrelated topic to meet a word count.
A 4,000-word, single-topic chapter must remain a single chunk. Do not split it artificially just because it is long.
The ideal of 500-2000 words is a desired outcome for typical documents, not a rule to be enforced.

4. The Rule of Consolidation (Aggressively Avoid Micro-Chunking)
Actively group related paragraphs. Do not create a new chunk for every paragraph, <h2> heading, or minor sub-topic. If multiple paragraphs or sections all serve the same central theme, they belong together in one chunk.
Heuristics for Decision-Making
Use these tests to help you identify true thematic boundaries:
The "Distinct Article" Test: Could this chunk be published on its own and still make sense? Does it feel like a complete piece?
The "Topic Title" Test: Can you give the chunk a simple, specific title? If you need to use "and" or "various topics" in the title (e.g., "User Authentication and Database Management"), it's a strong sign that it should be two separate chunks.
Lexical Shift Analysis: Does a new chunk introduce a completely different set of core keywords and vocabulary, distinct from the previous chunk? A gradual evolution of vocabulary is normal within a chunk; a sudden, wholesale replacement indicates a boundary.

Structural Cues: Pay attention to major structural elements like Chapter X declarations, or horizontal rules. These are strong indicators of a thematic boundary, but they should be validated against the thematic content itself.

Output Structure
For each chunk you create, provide the following:

Heading: A concise, yet comprehensive title that encapsulates the central theme of the entire chunk.
Content: The exact, unaltered original text of the chunk.
Keywords: A list of 7-10 core keywords and concepts that define the chunk's semantic fingerprint.
Summary: A 2-3 sentence, high-level abstract of the chunk's content, explaining its purpose and main points.

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
