"""
Query distillation utility using GPT-4o
"""

import os
from typing import Optional
from openai import OpenAI


def distill_query(query: str, language: Optional[str] = None) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    lang_instruction = ""
    if language and language != "en":
        lang_instruction = f" Your response must be in {language} language only."
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": 
                 f"You are a query distillation assistant. Your job is to extract the core information need from a user query and reformulate it as a clear, concise search query. Remove conversational elements, pleasantries, and redundant context. Keep only what's needed for retrieval.{lang_instruction} Respond with ONLY the distilled query, no explanations or additional text."},
                {"role": "user", "content": query}
            ],
            temperature=0.0,
            max_tokens=100
        )
        
        distilled_query = response.choices[0].message.content.strip()
        return distilled_query
        
    except Exception as e:
        print(f"Error in query distillation: {e}")
        return query 