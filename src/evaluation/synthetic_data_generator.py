"""
Synthetic Data Generator for RAG Evaluation
This script generates synthetic questions and ground truth data for evaluating the 4 approaches
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import openai

QUESTION_CATEGORIES = [
    "factual",         
    "procedural",     
    "conceptual",      
    "comparative",     
    "troubleshooting", 
    "visual",          
    "technical",       
    "configuration",   
    "navigation",      
    "security",        
    "integration",     
    "optimization",    
]

QUERY_TYPES = [
    "direct",          
    "conversational",  
    "technical",       
    "simple",          
    "specific",        
    "broad",           
    "detailed",        
    "multilingual",    
]

COMPLEXITY_LEVELS = ["simple", "medium", "complex"]

def load_document_chunks(manual_type: str) -> List[Dict[str, Any]]:
    base_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    chunks_path = base_path / "data" / manual_type / f"cleaned_{manual_type.split('_')[0]}_chunks.json"
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_figure_metadata(manual_type: str) -> Dict[str, Any]:
    base_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    metadata_path = base_path / "data" / manual_type / "figure_metadata.json"
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        chunks = load_document_chunks(manual_type)
        figure_metadata = {}
        
        for chunk in chunks:
            if "figures" in chunk and chunk["figures"]:
                for figure in chunk["figures"]:
                    figure_id = figure["figure_id"]
                    figure_metadata[figure_id] = {
                        "filename": figure.get("filename", ""),
                        "caption": figure.get("caption", ""),
                        "page": figure.get("page", "unknown"),
                        "section_id": chunk.get("section_id", "unknown"),
                        "section_title": chunk.get("section_title", "unknown")
                    }
        
        return figure_metadata

def extract_toc(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    toc = []
    
    sorted_chunks = sorted(chunks, key=lambda x: x.get("section_id", ""))
    
    for chunk in sorted_chunks:
        section_id = chunk.get("section_id", "")
        section_title = chunk.get("section_title", "")
        page_range = chunk.get("page_range", "")
        
        if section_id and section_title:
            toc.append({
                "section_id": section_id,
                "section_title": section_title,
                "page_range": page_range
            })
    
    return toc

def select_representative_chunks(chunks: List[Dict[str, Any]], count: int = 5) -> List[Dict[str, Any]]:
    scored_chunks = []
    
    for chunk in chunks:
        content_length = len(chunk.get("content", ""))
        has_figures = 1 if chunk.get("figures") else 0
        
        score = content_length + has_figures * 1000
        
        scored_chunks.append((score, chunk))
    
    sorted_chunks = [chunk for _, chunk in sorted(scored_chunks, key=lambda x: x[0], reverse=True)]
    
    selected = []
    section_ids = set()
    
    for chunk in sorted_chunks:
        section_id = chunk.get("section_id", "")
        
        if any(section_id.startswith(s) or s.startswith(section_id) for s in section_ids):
            continue
        
        selected.append(chunk)
        section_ids.add(section_id)
        
        if len(selected) >= count:
            break
    
    return selected

def format_chunk_for_prompt(chunk: Dict[str, Any]) -> str:
    section_id = chunk.get("section_id", "Unknown")
    section_title = chunk.get("section_title", "Unknown")
    page_range = chunk.get("page_range", "Unknown")
    content = chunk.get("content", "")
    formatted = f"CHUNK (Section {section_id}: {section_title}, Pages {page_range}):\n"
    formatted += content + "\n\n"
    
    if "figures" in chunk and chunk["figures"]:
        formatted += "FIGURES IN THIS SECTION:\n"
        
        for figure in chunk["figures"]:
            figure_id = figure.get("figure_id", "Unknown")
            caption = figure.get("caption", "Unknown")
            page = figure.get("page", "Unknown")
            
            formatted += f"- {figure_id} (Page {page}): {caption}\n"
    
    return formatted

def generate_gpt4_prompt(manual_type: str, toc: List[Dict[str, Any]], 
                        figures: Dict[str, Any], chunks: List[Dict[str, Any]],
                        category: str) -> str:
    subject = "Virtual Machine creation and management" if manual_type == "VM_manual" else "WiFi configuration"
    
    toc_formatted = "TABLE OF CONTENTS:\n"
    for item in toc[:20]: 
        toc_formatted += f"- Section {item['section_id']}: {item['section_title']} (Pages {item['page_range']})\n"
    
    figures_formatted = "KEY FIGURES:\n"
    figure_sample = list(figures.items())[:10]  
    for figure_id, metadata in figure_sample:
        caption = metadata.get("caption", "Unknown")
        page = metadata.get("page", "Unknown")
        figures_formatted += f"- {figure_id} (Page {page}): {caption}\n"
    
    chunks_formatted = "\nRELEVANT MANUAL CHUNKS:\n"
    for i, chunk in enumerate(chunks):
        chunks_formatted += f"\n{format_chunk_for_prompt(chunk)}"
        
    content_formatted = f"{toc_formatted}\n\n{figures_formatted}\n\n{chunks_formatted}"
    
    category_instructions = {
        "factual": "Create questions asking for specific facts or information found in the manual.",
        "procedural": "Create step-by-step 'how-to' questions about completing specific tasks.",
        "conceptual": "Create questions about understanding key concepts and terminology.",
        "comparative": "Create questions comparing different features, options, or approaches.",
        "troubleshooting": "Create questions about solving common problems or errors.",
        "visual": "Create questions specifically about the visual elements (screenshots, diagrams) in the manual.",
        "technical": "Create questions about specific technical details, specifications, or requirements.",
        "configuration": "Create questions about how to configure or set up specific features or systems.",
        "navigation": "Create questions about navigating through interfaces or menus.",
        "security": "Create questions about security features, permissions, or best practices.",
        "integration": "Create questions about integrating with other systems or components.",
        "optimization": "Create questions about optimizing performance or efficiency."
    }
    
    query_type = random.choice(QUERY_TYPES)
    query_type_instructions = {
        "direct": "Format questions in a direct, straightforward style.",
        "conversational": "Format questions in a conversational, casual style.",
        "technical": "Use technical language and terminology in the questions.",
        "simple": "Use simple, non-technical language in the questions.",
        "specific": "Make questions very specific and detailed.",
        "broad": "Make questions broad and general in scope.",
        "detailed": "Include requests for detailed information in the questions.",
        "multilingual": "Include some technical terms in both languages in the questions."
    }
    
    prompt = f"""You are an expert in technical documentation and IT systems. I need you to create realistic test questions about {subject} based on a technical manual.

I'll provide:
1. A table of contents from the manual
2. Information about key figures/images
3. Several specific content chunks from the manual

TASK:
Generate 5 {category} questions that users might ask about the content in these chunks.
{category_instructions.get(category, "")}
{query_type_instructions.get(query_type, "")}

For each question:
1. Provide the question text in both English and Italian
2. Indicate which sections and figures contain the relevant information
3. Provide a brief "ground truth" answer (about 2-3 sentences) based ONLY on the information provided
4. Rate the question's complexity (simple, medium, complex)
5. Add a "query_type" field with value "{query_type}"

IMPORTANT:
- Only reference information that is actually in the manual chunks provided
- Make questions realistic and practical for IT professionals using this documentation
- Questions should vary in complexity and focus on different aspects of the content
- The ground truth answer should cite specific sections or figures
- Make sure questions cover a diverse range of topics within the provided chunks

{content_formatted}

FORMAT REQUIREMENTS:
Return your response as a valid JSON object with the structure:
{{
  "questions": [
    {{
      "question_en": "English question text",
      "question_it": "Italian question text",
      "relevant_sections": ["section_id1", "section_id2"],
      "relevant_figures": ["figure_id1", "figure_id2"],
      "ground_truth_en": "English answer",
      "ground_truth_it": "Italian answer",
      "complexity": "simple|medium|complex",
      "query_type": "{query_type}"
    }},
    {{
      // additional questions
    }}
  ]
}}

Do not include any explanations outside the JSON. ONLY return a valid, properly formatted JSON object.
"""
    return prompt

def generate_questions_with_gpt4(prompt: str) -> List[Dict[str, Any]]:
    client = openai.OpenAI()
    
    json_prompt = prompt + "\n\nIMPORTANT: Format your response as a valid JSON object with a 'questions' array."
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates synthetic test data for RAG system evaluation in JSON format."},
                {"role": "user", "content": json_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
    except Exception as e:
        print(f"Note: Using standard response format due to: {e}")
        response = client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates synthetic test data for RAG system evaluation in JSON format."},
                {"role": "user", "content": json_prompt}
            ],
            temperature=0.7
        )
    
    try:
        content = response.choices[0].message.content.strip()
        
        json_start = content.find('{')
        json_end = content.rfind('}')
        
        if json_start >= 0 and json_end > json_start:
            json_str = content[json_start:json_end+1]
            data = json.loads(json_str)
        else:
            data = json.loads(content)
        
        if "questions" in data:
            return data["questions"]
        elif isinstance(data, list):
            return data
        else:
            questions = []
            for key, value in data.items():
                if isinstance(value, dict) and "question_en" in value:
                    questions.append(value)
            
            if not questions:
                for key, value in data.items():
                    if isinstance(value, dict) and any(field in value for field in ["question", "question_en", "query"]):
                        questions.append(value)
            
            return questions
            
    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        print(f"Error parsing GPT-4 response: {e}")
        print(f"Raw response:\n{response.choices[0].message.content}")
        
        content = response.choices[0].message.content
        try:
            import re
            json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
            matches = re.findall(json_pattern, content)
            
            if matches:
                questions = []
                for match in matches:
                    try:
                        question_data = json.loads(match)
                        if "question_en" in question_data or "question" in question_data:
                            questions.append(question_data)
                    except:
                        pass
                
                if questions:
                    print(f"Successfully extracted {len(questions)} questions using pattern matching.")
                    return questions
        except Exception as e2:
            print(f"Pattern matching also failed: {e2}")
        
        return []

def generate_synthetic_data(manual_type: str, output_path: str, 
                          questions_per_category: int = 3,
                          chunk_count: int = 5):
    print(f"Loading chunks and metadata for {manual_type}...")
    chunks = load_document_chunks(manual_type)
    figures = load_figure_metadata(manual_type)
    
    toc = extract_toc(chunks)
    
    all_questions = []
    
    for category in QUESTION_CATEGORIES:
        print(f"Generating {questions_per_category} {category} questions...")
        
        for i in range(questions_per_category):
            selected_chunks = select_representative_chunks(chunks, count=chunk_count)
            
            prompt = generate_gpt4_prompt(
                manual_type=manual_type,
                toc=toc,
                figures=figures,
                chunks=selected_chunks,
                category=category
            )
            
            questions = generate_questions_with_gpt4(prompt)
            
            for j, question in enumerate(questions):
                question["id"] = f"{manual_type}_{category}_{i}_{j}"
                question["category"] = category
                question["manual_type"] = manual_type
                
                if "relevant_sections" not in question:
                    question["relevant_sections"] = []
                if "relevant_figures" not in question:
                    question["relevant_figures"] = []
                    
                if "question_en" not in question:
                    question["question_en"] = question.get("question", "")
                if "question_it" not in question:
                    question["question_it"] = question.get("question", "")
                if "ground_truth_en" not in question:
                    question["ground_truth_en"] = question.get("ground_truth", "")
                if "ground_truth_it" not in question:
                    question["ground_truth_it"] = question.get("ground_truth", "")
                    
                all_questions.append(question)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"questions": all_questions}, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(all_questions)} questions saved to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation data for RAG systems")
    parser.add_argument(
        "--manual-type",
        type=str,
        choices=["VM_manual", "wifi_manual", "both"],
        default="both",
        help="Manual type to generate questions for"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_data",
        help="Directory to store generated data"
    )
    parser.add_argument(
        "--questions-per-category",
        type=int,
        default=3,
        help="Number of question sets to generate per category"
    )
    parser.add_argument(
        "--chunk-count",
        type=int,
        default=5,
        help="Number of chunks to include in each prompt"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.manual_type == "both" or args.manual_type == "VM_manual":
        generate_synthetic_data(
            manual_type="VM_manual",
            output_path=os.path.join(args.output_dir, "vm_synthetic_questions.json"),
            questions_per_category=args.questions_per_category,
            chunk_count=args.chunk_count
        )
    
    if args.manual_type == "both" or args.manual_type == "wifi_manual":
        generate_synthetic_data(
            manual_type="wifi_manual",
            output_path=os.path.join(args.output_dir, "wifi_synthetic_questions.json"),
            questions_per_category=args.questions_per_category,
            chunk_count=args.chunk_count
        )