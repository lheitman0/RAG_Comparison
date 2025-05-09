"""
Language detection utility using fastText's LID-176 model
"""

import os
from typing import Dict, Optional
from pathlib import Path


def detect_language(text: str) -> str:
    model_path = Path("./models/lid.176.ftz")

    try:
        import fasttext
        
        if not model_path.exists():
            model_path.parent.mkdir(exist_ok=True, parents=True)
            import urllib.request
            print("Downloading fastText language identification model…")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
                model_path
            )
        model = fasttext.load_model(str(model_path))
        predictions = model.predict(text.replace('\n', ' '))
        lang_code = predictions[0][0].replace('__label__', '')
        return lang_code
    except Exception as e:
        print(f"fastText model unavailable ({e}); falling back to heuristic language detection")
        return _fallback_language_detection(text)
    except ImportError:
        print("fastText python package not installed; using heuristic language detection")
        return _fallback_language_detection(text)


def _fallback_language_detection(text: str) -> str:
    language_markers: Dict[str, str] = {
        'en': ['the', 'and', 'is', 'in', 'to', 'it', 'of', 'that'],
        'it': ['il', 'la', 'e', 'che', 'di', 'un', 'per', 'sono'],
        'fr': ['le', 'la', 'et', 'est', 'en', 'que', 'une', 'pour'],
        'de': ['der', 'die', 'und', 'ist', 'das', 'in', 'ein', 'zu'],
        'es': ['el', 'la', 'en', 'y', 'es', 'que', 'de', 'un']
    }
    
    words = text.lower().split()
    
    language_scores = {lang: 0 for lang in language_markers}
    
    for word in words:
        for lang, markers in language_markers.items():
            if word in markers:
                language_scores[lang] += 1
    
    if any(language_scores.values()):
        return max(language_scores.items(), key=lambda x: x[1])[0]
    
    return 'en' 