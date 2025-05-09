"""
Chunk merging utility
"""

from typing import List, Dict, Any, Optional
import re


class ContextHit:
    def __init__(
        self,
        content: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.score = score
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        doc = self.metadata.get('document', '')
        section = self.metadata.get('section_id', '')
        return f"ContextHit(doc={doc}, section={section}, score={self.score:.3f})"


def merge_adjacent_chunks(
    hits: List[Dict[str, Any]],
    max_tokens: int = 400,
    token_ratio: int = 4
) -> List[ContextHit]:
    if not hits:
        return []
    
    max_chars = max_tokens * token_ratio
    
    def get_sort_key(hit):
        meta = hit.get('metadata', {})
        doc = meta.get('document', '')
        
        section = meta.get('section_id', '')
        if section:
            parts = [int(p) if p.isdigit() else p for p in re.split(r'\.|\s', section)]
            return (doc, parts)
        
        page = meta.get('page', 0)
        return (doc, [page])
    
    try:
        sorted_hits = sorted(hits, key=get_sort_key)
    except:
        sorted_hits = hits
    
    merged_hits = []
    current_merged = None
    
    for hit in sorted_hits:
        content = hit.get('content', '')
        metadata = hit.get('metadata', {})
        score = hit.get('score', 0.0)
        
        if not current_merged:
            current_merged = ContextHit(content, score, metadata.copy())
        else:
            curr_meta = current_merged.metadata
            
            same_doc = metadata.get('document') == curr_meta.get('document')
            
            adjacent = False
            if same_doc:
                curr_section = curr_meta.get('section_id', '')
                next_section = metadata.get('section_id', '')
                
                if curr_section and next_section:
                    curr_parts = curr_section.split('.')
                    next_parts = next_section.split('.')
                    
                    if (len(curr_parts) < len(next_parts) and
                        all(c == n for c, n in zip(curr_parts, next_parts))):
                        adjacent = True
                    elif (len(curr_parts) == len(next_parts) and
                          all(c == n for c, n in zip(curr_parts[:-1], next_parts[:-1])) and
                          (int(next_parts[-1]) - int(curr_parts[-1]) == 1 if curr_parts[-1].isdigit() and next_parts[-1].isdigit() else False)):
                        adjacent = True
                else:
                    curr_page = curr_meta.get('page', 0)
                    next_page = metadata.get('page', 0)
                    if isinstance(curr_page, (int, float)) and isinstance(next_page, (int, float)):
                        adjacent = (next_page - curr_page <= 1)
            
            if same_doc and adjacent and len(current_merged.content) + len(content) <= max_chars:
                current_merged.content += f"\n\n{content}"
                current_merged.score = max(current_merged.score, score)
                
                if 'figure_path' in metadata:
                    if 'figure_path' not in current_merged.metadata:
                        current_merged.metadata['figure_path'] = metadata['figure_path']
                    else:
                        existing = current_merged.metadata['figure_path']
                        if isinstance(existing, list):
                            if metadata['figure_path'] not in existing:
                                existing.append(metadata['figure_path'])
                        else:
                            if metadata['figure_path'] != existing:
                                current_merged.metadata['figure_path'] = [existing, metadata['figure_path']]
                
                if 'section_id' in curr_meta and 'section_id' in metadata:
                    current_merged.metadata['section_id'] = f"{curr_meta['section_id']}-{metadata['section_id']}"
                if 'page' in curr_meta and 'page' in metadata:
                    if curr_meta['page'] != metadata['page']:
                        current_merged.metadata['page'] = f"{curr_meta['page']}-{metadata['page']}"
            else:
                merged_hits.append(current_merged)
                current_merged = ContextHit(content, score, metadata.copy())
    
    if current_merged:
        merged_hits.append(current_merged)
    
    merged_hits.sort(key=lambda x: x.score, reverse=True)
    
    return merged_hits 