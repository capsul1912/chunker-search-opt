import time
from text_tools import process_large_text, count_words

def test_chunking_performance():
    """Test chunking performance across different document sizes"""
    
    # Base text sample
    base_text = '''
    This is a comprehensive test document designed to evaluate the performance of our chunking system across various document sizes.

    Section 1: Introduction to Text Processing
    Text processing is a fundamental task in natural language processing that involves the manipulation and analysis of textual data. Modern applications require efficient methods to handle large volumes of text while maintaining semantic coherence and meaning.

    Section 2: Chunking Strategies and Methodologies  
    Document chunking involves breaking down large texts into smaller, manageable pieces. The key is to maintain semantic boundaries while optimizing for processing efficiency. Various strategies exist, from simple character-based splitting to sophisticated semantic-aware approaches.

    Section 3: Performance Optimization Techniques
    When dealing with large-scale text processing, performance becomes critical. Key optimization areas include API call frequency, chunk size optimization, embedding generation efficiency, and database storage strategies.

    Section 4: Semantic Understanding and Context Preservation
    Advanced chunking systems must preserve the semantic relationships between different parts of the text. This requires understanding context, maintaining thematic coherence, and ensuring that related concepts remain grouped together.

    Section 5: Implementation Best Practices
    Production-ready chunking systems should implement robust error handling, efficient resource utilization, scalable architecture patterns, and monitoring capabilities to ensure reliable operation at scale.
    '''

    test_cases = [
        (1, "Small document"),
        (5, "Medium document"), 
        (10, "Large document"),
        (20, "Very large document")
    ]
    
    print("=== CHUNKING PERFORMANCE ANALYSIS ===\n")
    
    for multiplier, description in test_cases:
        test_text = base_text * multiplier
        word_count = count_words(test_text)
        
        print(f"Testing {description}: {word_count:,} words")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            result = process_large_text(test_text)
            end_time = time.time()
            
            processing_time = end_time - start_time
            chunk_count = len(result.get("chunks", []))
            
            # Calculate efficiency metrics
            words_per_second = word_count / processing_time if processing_time > 0 else 0
            seconds_per_chunk = processing_time / chunk_count if chunk_count > 0 else 0
            
            print(f"✓ Processing time: {processing_time:.2f} seconds")
            print(f"✓ Chunks created: {chunk_count}")
            print(f"✓ Efficiency: {words_per_second:.0f} words/second")
            print(f"✓ Time per chunk: {seconds_per_chunk:.2f} seconds")
            
            # Show chunk size distribution
            if chunk_count > 0:
                chunk_sizes = [count_words(chunk['content']) for chunk in result['chunks']]
                avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
                min_chunk = min(chunk_sizes)
                max_chunk = max(chunk_sizes)
                
                print(f"✓ Average chunk size: {avg_chunk_size:.0f} words")
                print(f"✓ Chunk size range: {min_chunk} - {max_chunk} words")
        
        except Exception as e:
            print(f"✗ Error: {e}")
        
        print("\n")

if __name__ == "__main__":
    test_chunking_performance() 