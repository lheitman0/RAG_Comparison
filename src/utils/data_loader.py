import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

def load_json_file(file_path: str) -> Any:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_document_chunks(manual_type: str) -> List[Dict[str, Any]]:
    base_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    if manual_type == "VM_manual":
        chunks_path = base_path / "data" / "VM_manual" / "cleaned_VM_chunks.json"
    elif manual_type == "wifi_manual":
        chunks_path = base_path / "data" / "wifi_manual" / "cleaned_wifi_chunks.json"
    else:
        raise ValueError(f"Unknown manual type: {manual_type}")
    
    return load_json_file(chunks_path)

def load_specific_document_chunks(manual_type: str, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    if file_path and os.path.exists(file_path):
        return load_json_file(file_path)
    else:
        return load_document_chunks(manual_type)

def load_figure_metadata(manual_type: str) -> Dict[str, Any]:
    base_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    metadata_path = base_path / "data" / manual_type / "figure_metadata.json"
    
    return load_json_file(metadata_path)

def get_image_path(manual_type: str, figure_filename: str) -> Path:
    base_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    true_manual = None
    if "VM_manual" in figure_filename or "vm_manual" in figure_filename.lower():
        true_manual = "VM_manual"
    elif "wifi_manual" in figure_filename or "wifi_manual" in figure_filename.lower():
        true_manual = "wifi_manual"
    
    if true_manual:
        return base_path / "data" / true_manual / "figures" / figure_filename
    
    if manual_type == "ISTRUZIONE OPERATIVA CREAZIONE VM CLOUD INSIEL REV 00" or "vm" in manual_type.lower():
        folder_name = "VM_manual"
    elif manual_type == "ISTRUZIONE OPERATIVA CONFIGURAZIONE WIFI ARUBA REV 01" or "wifi" in manual_type.lower():
        folder_name = "wifi_manual"
    elif manual_type == "combined":
        if figure_filename.lower().startswith(("vm_", "vm-")):
            folder_name = "VM_manual"
        else:
            folder_name = "wifi_manual"
    else:
        folder_name = manual_type
    
    return base_path / "data" / folder_name / "figures" / figure_filename