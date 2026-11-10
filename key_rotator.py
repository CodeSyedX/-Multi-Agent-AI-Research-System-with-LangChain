
import itertools
import json
import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


class GeminiKeyRotator:
    def __init__(self, model: str = "gemini-3.5-flash", **kwargs):
        env_keys = [
            os.getenv("GEMINI_API_KEY_1"),
            os.getenv("GEMINI_API_KEY_2"),
            os.getenv("GEMINI_API_KEY_3"),
            os.getenv("GOOGLE_API_KEY"),
        ]
        self.api_keys = list(dict.fromkeys([k for k in env_keys if k]))

        if not self.api_keys:
            raise ValueError("[❌] No valid Gemini API keys found in environment variables.")

        self.key_cycle = itertools.cycle(self.api_keys)
        self.model = model
        self.kwargs = kwargs

    def get_llm(self):
        primary_key = next(self.key_cycle)
        masked = f"...{primary_key[-4:]}" if len(primary_key) >= 4 else "masked"
        print(f"🔑 [KeyRotator] Active key suffix: {masked}")

        backup_keys = [k for k in self.api_keys if k != primary_key]
        primary_llm = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=primary_key,
            **self.kwargs
        )

        if not backup_keys:
            return primary_llm

        fallback_llms = [
            ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=k,
                **self.kwargs
            )
            for k in backup_keys
        ]
        return primary_llm.with_fallbacks(fallback_llms)


def extract_text_content(raw) -> str:
    """
    THE ROOT CAUSE FIX:
    New langchain-google-genai SDK returns res.content as a list of
    content block dicts: [{'type': 'text', 'text': '...', 'extras': {...}}]
    instead of a plain string.

    This function safely extracts the text string regardless of whether
    res.content is:
      - a plain str (old SDK behavior)
      - a list of content block dicts (new SDK behavior)
      - a single dict
    """
    if isinstance(raw, str):
        return raw

    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, dict):
                # Standard content block: {'type': 'text', 'text': '...'}
                text = item.get("text", "")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()

    if isinstance(raw, dict):
        return str(raw.get("text", raw.get("content", str(raw)))).strip()

    return str(raw).strip()


def _fix_json_string(text: str) -> str:
    """Fix common LLM JSON issues: unescaped newlines, trailing commas."""
    fixed = re.sub(
        r'("(?:[^"\\]|\\.)*")',
        lambda m: m.group(0)
            .replace('\n', '\\n')
            .replace('\r', '\\r')
            .replace('\t', '\\t'),
        text
    )
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    if '"' not in fixed and "'" in fixed:
        fixed = fixed.replace("'", '"')
    return fixed


def safe_parse_json(text, default=None):
    """
    Robustly extract and parse JSON from raw LLM output.
    Accepts str, list of content blocks, or dict.
    """
    if default is None:
        default = {}

    # Step 1: normalize to plain string regardless of SDK version
    raw = extract_text_content(text)

    # Step 2: strip BOM and zero-width chars
    raw = raw.lstrip('\ufeff\u200b\u200c\u200d').strip()

    # Step 3: strip ALL markdown code fence variations
    raw = re.sub(r'`{3,}(?:json|JSON|python|text|js)?\s*', '', raw)
    raw = re.sub(r'~{3,}(?:json|JSON)?\s*', '', raw)
    raw = raw.strip()

    # Step 4: direct parse
    try:
        result = json.loads(raw, strict=False)
        return result
    except json.JSONDecodeError:
        pass

    # Step 5: extract JSON object { ... }
    obj_match = re.search(r'\{[\s\S]*\}', raw)
    if obj_match:
        candidate = obj_match.group(0).strip()
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_fix_json_string(candidate), strict=False)
        except json.JSONDecodeError:
            pass

    # Step 6: extract JSON array [ ... ]
    arr_match = re.search(r'\[[\s\S]*\]', raw)
    if arr_match:
        candidate = arr_match.group(0).strip()
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_fix_json_string(candidate), strict=False)
        except json.JSONDecodeError:
            pass

    print(f"⚠️ [safe_parse_json] All parse attempts failed. Returning default.")
    return default


# Singleton
try:
    rotator = GeminiKeyRotator(temperature=0.2, max_retries=2)
except ValueError as e:
    print(f"⚠️ [KeyRotator] Init warning: {e}")
    rotator = None


def get_llm():
    if rotator is None:
        raise RuntimeError("[❌ KeyRotator] No API keys configured.")
    return rotator.get_llm()