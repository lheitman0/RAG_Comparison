"""
Debug script to inspect the vector store structure and contents.
"""

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.embeddings.vector_store import TechnicalManualVectorStore
import numpy as np

def debug_vector_store(persist_directory: str = "./vector_db/combined"):
    """Debug the vector store structure and contents."""
    print("\n=== Vector Store Debug Information ===\n")
    
    # Initialize vector store
    vector_store = TechnicalManualVectorStore(persist_directory=persist_directory)
    
    # Debug text collection
    print("=== Text Collection ===")
    text_collection = vector_store.text_collection
    text_data = text_collection.get()
    print(f"Number of documents: {len(text_data['ids'])}")
    
    # Get first document
    print("\nFirst Document:")
    print(f"ID: {text_data['ids'][0]}")
    print("Metadata:")
    for key, value in text_data['metadatas'][0].items():
        print(f"  {key}: {value}")
    print("\nContent:")
    print(text_data['documents'][0][:500] + "..." if len(text_data['documents'][0]) > 500 else text_data['documents'][0])
    
    # Debug image collection
    print("\n=== Image Collection ===")
    image_collection = vector_store.image_collection
    image_data = image_collection.get()
    print(f"Number of images: {len(image_data['ids'])}")
    
    # Get first image
    print("\nFirst Image:")
    print(f"ID: {image_data['ids'][0]}")
    print("Metadata:")
    for key, value in image_data['metadatas'][0].items():
        print(f"  {key}: {value}")
    print("\nCaption:")
    print(image_data['documents'][0])
    
    # Test a simple query
    print("\n=== Testing Query ===")
    test_query = "What are the steps to create a new VM in VMware?"
    
    # Create dummy embeddings for testing
    dummy_embedding = np.random.rand(1536).tolist()  # OpenAI embedding size
    
    # Text query
    print("\nText Query Results:")
    text_results = vector_store.query_text(dummy_embedding, n_results=3)
    print(f"Number of results: {len(text_results)}")
    if text_results:
        print("\nFirst Text Result:")
        for key, value in text_results[0].items():
            print(f"  {key}: {value}")
    
    # Image query
    print("\nImage Query Results:")
    image_results = vector_store.query_images(dummy_embedding, n_results=3)
    print(f"Number of results: {len(image_results)}")
    if image_results:
        print("\nFirst Image Result:")
        for key, value in image_results[0].items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    debug_vector_store() 