# src/utils.py
import re
import nltk
from typing import List

nltk.download('punkt', quiet=True)

def normalize_whitespace(text: str) -> str:

    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_sentences(text: str) -> List[str]:

    text = normalize_whitespace(text)
    if not text:
        return []
    return nltk.sent_tokenize(text)

def remove_news_prefix(text: str) -> str:

    patterns = [
        r'^\s*\([A-Za-z]+\)\s*[-–—]+\s*',
        r'^\s*[A-Z][A-Za-z\.\'\- ]+(?:,\s*[A-Z][A-Za-z\.\'\- ]+){0,2}\s*\([A-Za-z]+\)\s*[-–—]+\s*',
        r'^\s*[A-Z][A-Za-z\.\'\- ]+\s*-\s*'
    ]
    for pattern in patterns:
        new_text = re.sub(pattern, '', text, count=1)
        if new_text != text:
            return new_text
    return text