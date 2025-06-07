import re
from constants import DEFAULT_CHUNK_SIZE, WORDS_TO_TOKENS_RATIO, MIN_CHUNK_REFILL_SIZE, MAX_SAFE_GEMINI_TOKENS
from ai_services import count_text_tokens


def split_text_by_tokens(text, target_tokens):
    """
    Split text to get approximately target_tokens worth of content.
    Returns the extracted part and remaining text.
    """
    if count_text_tokens(text) <= target_tokens:
        return text, ""
    
    # Try to split by paragraphs first to keep meaning together
    paragraphs = text.split('\n\n')
    extracted_parts = []
    current_tokens = 0
    
    for i, paragraph in enumerate(paragraphs):
        paragraph_tokens = count_text_tokens(paragraph)
        
        if current_tokens + paragraph_tokens <= target_tokens:
            extracted_parts.append(paragraph)
            current_tokens += paragraph_tokens
        else:
            # If paragraph is too big, try splitting by sentences
            if current_tokens < target_tokens * 0.8:  # Use 80% of available space
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    sentence_tokens = count_text_tokens(sentence)
                    if current_tokens + sentence_tokens <= target_tokens:
                        extracted_parts.append(sentence)
                        current_tokens += sentence_tokens
                    else:
                        break
            break
    
    extracted_text = '\n\n'.join(extracted_parts)
    
    # Find what's left after extraction
    if extracted_text:
        extracted_length = len(extracted_text)
        remaining_text = text[extracted_length:].lstrip('\n ')
    else:
        remaining_text = text
    
    return extracted_text, remaining_text


def process_large_text(text, chunk_size=DEFAULT_CHUNK_SIZE):
    """
    Process large text by maintaining working chunks and extracting semantic pieces.
    This is the main function for handling big documents.
    """
    from ai_services import break_text_into_chunks, count_text_tokens  # Import here to avoid circular import
    
    # Log the dynamic chunking process
    total_tokens = count_text_tokens(text)
    print(f"Dynamic chunking: {total_tokens:,} tokens")
    
    all_chunks = []
    remaining_text = text
    
    while remaining_text.strip():
        # Get a working chunk (10k tokens)
        working_chunk, remaining_text = split_text_by_tokens(remaining_text, chunk_size)
        
        if not working_chunk.strip():
            break
        
        working_tokens = count_text_tokens(working_chunk)
        print(f"Working chunk: {working_tokens:,} tokens, Remaining: {count_text_tokens(remaining_text):,} tokens")
            
        # Process this working chunk until it gets below 5k tokens
        while working_chunk.strip():
            working_tokens = count_text_tokens(working_chunk)
            print(f"Current working chunk: {working_tokens:,} tokens")
            
            # Check if we need to refill (below 5k tokens)
            if working_tokens < MIN_CHUNK_REFILL_SIZE:
                if remaining_text.strip():
                    # Refill with 5k tokens from remaining text
                    refill_text, remaining_text = split_text_by_tokens(remaining_text, MIN_CHUNK_REFILL_SIZE)
                    if refill_text.strip():
                        working_chunk = (working_chunk + "\n\n" + refill_text).strip()
                        working_tokens = count_text_tokens(working_chunk)
                        print(f"Refilled working chunk: {working_tokens:,} tokens, Remaining: {count_text_tokens(remaining_text):,} tokens")
                        continue
                else:
                    # No more text to refill, save remaining and break
                    if working_chunk.strip():
                        all_chunks.append({
                            "heading": "Final Content",
                            "content": working_chunk.strip(),
                            "keywords": [],
                            "summary": "Final remaining text"
                        })
                    break
            
            # Send working chunk to Gemini for semantic chunking
            semantic_result = break_text_into_chunks(working_chunk)
            
            try:
                # Handle the AI response
                if isinstance(semantic_result, str):
                    import json
                    parsed_result = json.loads(semantic_result)
                else:
                    parsed_result = semantic_result
                
                if "error" in parsed_result:
                    # If AI had problems, save the whole chunk
                    print(f"ERROR: Gemini error: {parsed_result['error']}")
                    all_chunks.append({
                        "heading": "Processing Error",
                        "content": working_chunk,
                        "keywords": [],
                        "summary": f"Error during processing: {parsed_result['error']}"
                    })
                    break
                
                if "chunks" not in parsed_result or not parsed_result["chunks"]:
                    # If no chunks came back, save the whole piece
                    print("WARNING: No semantic chunks returned, saving as single piece")
                    all_chunks.append({
                        "heading": "Unprocessed Content",
                        "content": working_chunk,
                        "keywords": [],
                        "summary": "Content that could not be broken into chunks"
                    })
                    break
                
                # Take the first semantic chunk that AI found
                first_chunk = parsed_result["chunks"][0]
                all_chunks.append(first_chunk)
                print(f"Extracted chunk: '{first_chunk['heading'][:50]}...' ({count_text_tokens(first_chunk['content']):,} tokens)")
                
                # Remove this chunk from working text
                first_chunk_content = first_chunk["content"]
                working_chunk = _remove_chunk_from_text(working_chunk, first_chunk_content)
                
                # Check token count after extracting chunk
                remaining_working_tokens = count_text_tokens(working_chunk) if working_chunk.strip() else 0
                print(f"After extraction, working chunk has: {remaining_working_tokens:,} tokens")
                
            except Exception as e:
                print(f"ERROR: Exception during processing: {e}")
                # Save the whole chunk if something went wrong
                all_chunks.append({
                    "heading": "Processing Error",
                    "content": working_chunk,
                    "keywords": [],
                    "summary": f"Error during processing: {str(e)}"
                })
                break
    
    print(f"Created {len(all_chunks)} semantic chunks")
    return {"chunks": all_chunks}


def _remove_chunk_from_text(current_chunk, chunk_content):
    """Remove the processed chunk content from the working text"""
    # Find where this chunk appears in our working text
    chunk_start = current_chunk.find(chunk_content)
    if chunk_start != -1:
        # Remove it cleanly
        return current_chunk[chunk_start + len(chunk_content):].lstrip('\n ')
    else:
        # Fallback: estimate how much to remove based on tokens
        chunk_tokens = count_text_tokens(chunk_content)
        words_to_remove = int(chunk_tokens * 0.75)  # Rough guess
        words = current_chunk.split()
        if len(words) > words_to_remove:
            return ' '.join(words[words_to_remove:])
        else:
            return ""


def estimate_tokens_from_words(text):
    """
    Backup method to estimate tokens when AI token counting fails.
    Uses word count approximation.
    """
    words = len(re.findall(r'\b\w+\b', text))
    return int(words * WORDS_TO_TOKENS_RATIO)


def clean_text_for_processing(text):
    """Clean and prepare text for processing"""
    # Remove excessive whitespace but keep paragraph breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces and tabs
    text = text.strip()
    return text
