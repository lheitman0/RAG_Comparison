"""
Simple test script to verify our retrieval implementation works correctly.
This helps ensure all the components are properly connected before running the full evaluation.
"""

import os
import json
import time
import sys
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.retrieval.base import retrieve, RetrievalRecipe, RetrievalApproach
from src.utils.chunk_merge import ContextHit

# Sample queries to test retrieval with
TEST_QUERIES = [
    {
        "query": "How do I configure my WiFi access point?",
        "language": "english",
        "expected_manual": "wifi_manual"
    },
    {
        "query": "Come configurare una macchina virtuale?",
        "language": "italian",
        "expected_manual": "VM_manual"
    }
]

def generate_simple_answer(query, context_hits, figure_paths=None, language="english"):
    """
    Generate a simple answer using OpenAI's API for demonstration purposes.
    
    Args:
        query: The user query
        context_hits: List of retrieved context hits
        figure_paths: List of figure paths (optional)
        language: Query language
        
    Returns:
        Generated answer
    """
    from openai import OpenAI
    import base64
    
    # Prepare context
    context_text = ""
    for i, hit in enumerate(context_hits):
        if hasattr(hit, 'content') and hit.content:
            content = hit.content
            source = hit.metadata.get('document', 'unknown')
            section = hit.metadata.get('section_id', '')
            context_text += f"\n--- Document {i+1} [Source: {source}, Section: {section}] ---\n{content}\n"
    
    # Language instruction
    language_instruction = f"Please respond in {language}." if language != "english" else ""
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Prepare images if available
    has_figures = figure_paths and len(figure_paths) > 0
    
    if has_figures:
        # Use GPT-4o Vision with figures
        messages = [
            {"role": "system", "content": f"You are a helpful technical documentation assistant. {language_instruction}"},
            {"role": "user", "content": [
                {"type": "text", "text": f"""Based on the following technical documentation and images, please answer the question:
                
Question: {query}

Relevant Documentation:
{context_text}

Please analyze both the text and the provided images to give a complete answer."""}
            ]}
        ]
        
        # Add images to the message (max 3 images)
        for figure_path in figure_paths[:3]:
            if os.path.exists(figure_path):
                # Read image and encode to base64
                with open(figure_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
        
        # Generate answer with vision model
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=300  # Keep it short for testing
        )
    else:
        # Text-only generation
        messages = [
            {"role": "system", "content": f"You are a helpful technical documentation assistant. {language_instruction}"},
            {"role": "user", "content": f"""Based on the following technical documentation, please answer the question:
            
Question: {query}

Relevant Documentation:
{context_text}"""}
        ]
        
        # Generate answer
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=300  # Keep it short for testing
        )
    
    return response.choices[0].message.content

def test_approach(approach_name, recipe):
    """
    Test a specific retrieval approach.
    
    Args:
        approach_name: Name of the approach being tested
        recipe: Retrieval recipe to use
    """
    print(f"\n===== Testing {approach_name} =====")
    
    for test_case in TEST_QUERIES:
        query = test_case["query"]
        language = test_case["language"]
        expected_manual = test_case["expected_manual"]
        
        print(f"\nQuery: {query}")
        print(f"Language: {language}")
        print(f"Expected manual: {expected_manual}")
        
        # Time the retrieval
        start_time = time.time()
        
        # Execute retrieval
        context_hits, metrics = retrieve(
            query=query,
            recipe=recipe,
            k=5  # Limit to 5 results for testing
        )
        
        # Calculate time
        retrieval_time = time.time() - start_time
        
        # Print retrieval metrics
        print(f"Retrieved {len(context_hits)} context hits in {retrieval_time:.2f} seconds")
        print(f"Text hits: {metrics.get('text_hits', 'N/A')}")
        print(f"Image hits: {metrics.get('image_hits', 0)}")
        
        # Check if we got results from the expected manual
        manual_hits = {}
        for hit in context_hits:
            if hasattr(hit, 'metadata') and 'document' in hit.metadata:
                doc = hit.metadata['document']
                manual_hits[doc] = manual_hits.get(doc, 0) + 1
            elif isinstance(hit, dict) and 'metadata' in hit and 'document' in hit['metadata']:
                doc = hit['metadata']['document']
                manual_hits[doc] = manual_hits.get(doc, 0) + 1
        
        print("Manual distribution:")
        for manual, count in manual_hits.items():
            print(f"  - {manual}: {count} hits")
            
        # Check if expected manual is in the results
        expected_found = any(expected_manual in doc for doc in manual_hits.keys())
        print(f"Found expected manual: {'Yes' if expected_found else 'No'}")
        
        # Extract figure paths
        figure_paths = []
        for hit in context_hits:
            if hasattr(hit, 'metadata') and 'figure_path' in hit.metadata:
                path = hit.metadata['figure_path']
                if os.path.exists(path):
                    figure_paths.append(path)
            elif isinstance(hit, dict) and 'metadata' in hit and 'figure_path' in hit['metadata']:
                path = hit['metadata']['figure_path']
                if os.path.exists(path):
                    figure_paths.append(path)
        
        print(f"Found {len(figure_paths)} figure paths")
        
        # Generate a quick answer (optional)
        if context_hits:
            print("\nGenerating quick answer...")
            try:
                answer = generate_simple_answer(
                    query=query,
                    context_hits=context_hits,
                    figure_paths=figure_paths,
                    language=language
                )
                print(f"\nAnswer: {answer[:200]}...")  # Print first 200 chars
            except Exception as e:
                print(f"Error generating answer: {e}")

def main():
    """Main function to test all retrieval approaches."""
    print("Testing retrieval approaches...\n")
    
    # Test each approach
    test_approach(
        "OpenSource RAG",
        RetrievalRecipe.open_source()
    )
    
    test_approach(
        "OpenAI CLIP RAG",
        RetrievalRecipe.openai_clip()
    )
    
    test_approach(
        "OpenAI Vision RAG",
        RetrievalRecipe.openai_vision()
    )
    
    test_approach(
        "Hybrid RAG",
        RetrievalRecipe.hybrid()
    )
    
    print("\nRetrieval testing complete!")

if __name__ == "__main__":
    main() 