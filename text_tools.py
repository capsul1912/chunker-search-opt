import re
from constants import DEFAULT_CHUNK_SIZE, MIN_CHUNK_REFILL_SIZE, MAX_SAFE_GEMINI_WORDS


def count_words(text):
    """Count words in text using regex"""
    return len(re.findall(r'\b\w+\b', text))


def split_text_by_words(text, target_words):
    """
    Split text to get approximately target_words worth of content.
    Returns the extracted part and remaining text.
    """
    if count_words(text) <= target_words:
        return text, ""
    
    # Try to split by paragraphs first to keep meaning together
    paragraphs = text.split('\n\n')
    extracted_parts = []
    current_words = 0
    
    for i, paragraph in enumerate(paragraphs):
        paragraph_words = count_words(paragraph)
        
        if current_words + paragraph_words <= target_words:
            extracted_parts.append(paragraph)
            current_words += paragraph_words
        else:
            # If paragraph is too big, try splitting by sentences
            if current_words < target_words * 0.8:  # Use 80% of available space
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    sentence_words = count_words(sentence)
                    if current_words + sentence_words <= target_words:
                        extracted_parts.append(sentence)
                        current_words += sentence_words
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
    Now uses word counting instead of token counting.
    """
    from ai_services import break_text_into_chunks  # Import here to avoid circular import
    
    # Log the dynamic chunking process
    total_words = count_words(text)
    print(f"=== STARTING CHUNKING PROCESS ===")
    print(f"Total document: {total_words:,} words")
    print(f"Working chunk size: {chunk_size:,} words")
    print(f"Refill threshold: {MIN_CHUNK_REFILL_SIZE:,} words")
    
    all_chunks = []
    remaining_text = text
    iteration = 1
    
    while remaining_text.strip():
        print(f"\n--- Iteration {iteration} ---")
        
        # Get a working chunk (7.5k words)
        working_chunk, remaining_text = split_text_by_words(remaining_text, chunk_size)
        
        if not working_chunk.strip():
            break
        
        working_words = count_words(working_chunk)
        remaining_words = count_words(remaining_text)
        print(f"Working chunk: {working_words:,} words")
        print(f"Remaining text: {remaining_words:,} words")
        
        chunk_extraction_count = 0
            
        # Process this working chunk until it gets below 3.75k words
        while working_chunk.strip():
            working_words = count_words(working_chunk)
            
            # Check if we need to refill (below threshold) but still process smaller docs
            if working_words < MIN_CHUNK_REFILL_SIZE and remaining_text.strip():
                print(f"Working chunk below threshold ({working_words:,} < {MIN_CHUNK_REFILL_SIZE:,} words)")
                
                # Refill with remaining text
                refill_text, remaining_text = split_text_by_words(remaining_text, MIN_CHUNK_REFILL_SIZE)
                if refill_text.strip():
                    working_chunk = (working_chunk + "\n\n" + refill_text).strip()
                    working_words = count_words(working_chunk)
                    remaining_words = count_words(remaining_text)
                    print(f"Refilled: {working_words:,} words in working chunk, {remaining_words:,} words remaining")
                    continue
            
            # Process the working chunk if it has substantial content
            if working_words < 10:  # Only skip AI for extremely tiny chunks (less than 10 words)
                if working_chunk.strip():
                    print(f"Saving tiny chunk directly: {working_words:,} words")
                    all_chunks.append({
                        "heading": "Content",
                        "content": working_chunk.strip(),
                        "keywords": [],
                        "summary": "Very small content section"
                    })
                break
            
            print(f"Sending {working_words:,} words to AI for semantic chunking...")
            
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
                    print(f"ERROR: AI chunking failed: {parsed_result['error']}")
                    all_chunks.append({
                        "heading": "Processing Error",
                        "content": working_chunk,
                        "keywords": [],
                        "summary": f"Error during processing: {parsed_result['error']}"
                    })
                    break
                
                if "chunks" not in parsed_result or not parsed_result["chunks"]:
                    # If no chunks came back, save the whole piece
                    print("WARNING: No semantic chunks returned from AI")
                    all_chunks.append({
                        "heading": "Unprocessed Content",
                        "content": working_chunk,
                        "keywords": [],
                        "summary": "Content that could not be broken into chunks"
                    })
                    break
                
                # Handle chunks based on working chunk size - now optimized for larger chunks
                chunks_received = parsed_result["chunks"]
                
                # Always take ALL chunks from response for better efficiency
                for chunk in chunks_received:
                    all_chunks.append(chunk)
                chunk_extraction_count += len(chunks_received)
                
                print(f"Extracted {len(chunks_received)} chunks from AI response:")
                for i, chunk in enumerate(chunks_received, 1):
                    chunk_words = count_words(chunk['content'])
                    heading_preview = chunk['heading'][:40] + "..." if len(chunk['heading']) > 40 else chunk['heading']
                    print(f"  #{chunk_extraction_count - len(chunks_received) + i}: '{heading_preview}' ({chunk_words:,} words)")
                
                # Calculate total words processed in this batch
                total_words_processed = sum(count_words(chunk['content']) for chunk in chunks_received)
                print(f"Total words processed in this batch: {total_words_processed:,}")
                
                # Remove all processed chunks from working text
                remaining_working_text = working_chunk
                for chunk in chunks_received:
                    remaining_working_text = _remove_chunk_from_text(remaining_working_text, chunk["content"])
                
                working_chunk = remaining_working_text
                
                # Check word count after extracting chunk(s)
                remaining_working_words = count_words(working_chunk) if working_chunk.strip() else 0
                print(f"Working chunk after extraction: {remaining_working_words:,} words")
                
                # If remainder is very small, save it directly instead of processing again
                if 0 < remaining_working_words < 10:
                    print(f"Saving tiny remainder directly: {remaining_working_words:,} words")
                    all_chunks.append({
                        "heading": "Additional Content",
                        "content": working_chunk.strip(),
                        "keywords": [],
                        "summary": "Remaining content from processing"
                    })
                    working_chunk = ""  # Clear to exit loop
                
            except Exception as e:
                print(f"ERROR: Exception during AI processing: {e}")
                # Save the whole chunk if something went wrong
                all_chunks.append({
                    "heading": "Processing Error",
                    "content": working_chunk,
                    "keywords": [],
                    "summary": f"Error during processing: {str(e)}"
                })
                break
        
        print(f"Iteration {iteration} complete: extracted {chunk_extraction_count} chunks")
        iteration += 1
    
    print(f"\n=== CHUNKING COMPLETE ===")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Total iterations: {iteration - 1}")
    
    return {"chunks": all_chunks}


def _remove_chunk_from_text(current_chunk, chunk_content):
    """Efficiently remove processed chunk content from working text"""
    # Quick word-based estimation for better performance
    chunk_words = count_words(chunk_content)
    current_words = current_chunk.split()
    
    # Remove approximately the same number of words from the beginning
    if len(current_words) > chunk_words:
        remaining_words = current_words[chunk_words:]
        return ' '.join(remaining_words)
    else:
        return ""





def clean_text_for_processing(text):
    """Clean and prepare text for processing"""
    # Remove excessive whitespace but keep paragraph breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces and tabs
    text = text.strip()
    return text
