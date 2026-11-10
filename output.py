# test_llm_output.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from key_rotator import extract_text_content, safe_parse_json

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",   # Fixed model name
    google_api_key=os.getenv("GEMINI_API_KEY_1") or os.getenv("GOOGLE_API_KEY"),
    temperature=0.2
)

test_prompt = (
    "Divide the topic 'Python Programming' into exactly 5 sequential sub-topics.\n"
    "Return ONLY a raw JSON array of 5 objects. Each object must have keys: "
    "'step' (int), 'title' (str), 'focus' (str).\n"
    "Start your response with [ and end with ]."
)

res = llm.invoke(test_prompt)
print("=== TYPE ===", type(res.content))
print("=== EXTRACTED TEXT ===")
extracted = extract_text_content(res.content)
print(extracted[:500])
print("=== PARSED ===")
parsed = safe_parse_json(res.content, default=[])
print(parsed)