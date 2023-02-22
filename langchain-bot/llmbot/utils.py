import re
import json
from typing import Dict


def parse_json_from_text_multiline(text: str) -> Dict:
    """Parse JSON from text, even if it spans multiple lines."""
    json_match = re.search(r"({.*?})", text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    return {}


def remove_json_from_text_multiline(text: str) -> str:
    """Remove JSON from text, even if it spans multiple lines."""
    json_match = re.search(r"({.*?})", text, re.DOTALL)
    if json_match:
        return text.replace(json_match.group(1), '')
    return text