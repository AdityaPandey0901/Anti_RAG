import requests
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# "gemini" = auto-answer via Gemini 2.5 Flash, "human" = manual terminal input
SELECTOR = "gemini"

BASE_URL = "https://workwithus.lucioai.com"

HEADERS = {
    "User-Agent": "hari_seldon",
    "Authorization": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYW1lIjoiQWRpdHlhIFBhbmRleSIsImVtYWlsIjoiYXA1MzEyQG55dS5lZHUiLCJkYXRlIjoiMjAyNi0wMy0yMCAxMDo1MTowMyJ9._RQOxrhdJHdpy9eguY42xgx13hiLDogS1J78vI1CaZQ",
    "Accept": "*/*",
    "Cache-Control": "no-cache",
}

COOKIES = {
    "auth_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYW1lIjoiQWRpdHlhIFBhbmRleSIsImVtYWlsIjoiYXA1MzEyQG55dS5lZHUiLCJkYXRlIjoiMjAyNi0wMy0yMCAxMDo1MTowMyJ9._RQOxrhdJHdpy9eguY42xgx13hiLDogS1J78vI1CaZQ",
}

# Step 1: GET /logic-it-out
print("=" * 60)
print("Step 1: GET /logic-it-out")
print("=" * 60)
resp = requests.get(f"{BASE_URL}/logic-it-out", headers=HEADERS, cookies=COOKIES)
print(f"Status: {resp.status_code}")
data = resp.json()
print(json.dumps(data, indent=2))

# Step 2: Parse questions and token
token = data["token"]
questions = data["questions"]

# Step 3: Answer questions
print("\n" + "=" * 60)
print(f"ANSWERING QUESTIONS  [mode={SELECTOR}]")
print("=" * 60)

if SELECTOR == "gemini":
    # Build a single prompt with all questions
    prompt_lines = [
        "Answer each of the following trivia questions.",
        "Return ONLY a JSON array with the answers in order.",
        "Use the exact types specified (int → number, str → string).\n",
    ]
    for i, q in enumerate(questions):
        prompt_lines.append(f"{i+1}. [{q['answer_type']}] {q['question']}")

    prompt = "\n".join(prompt_lines)
    print(f"\nPrompt sent to Gemini:\n{prompt}\n")

    gemini_resp = model.generate_content(prompt)
    raw_text = gemini_resp.text.strip()
    print(f"Gemini raw response: {raw_text}")

    # Strip markdown code fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]  # drop opening ```json
        raw_text = raw_text.rsplit("```", 1)[0]  # drop closing ```
        raw_text = raw_text.strip()

    answers = json.loads(raw_text)

    # Enforce expected types
    for i, q in enumerate(questions):
        if q["answer_type"] == "int":
            answers[i] = int(answers[i])
        else:
            answers[i] = str(answers[i])

    print(f"Parsed answers: {answers}")

else:
    # Human mode — manual terminal input
    answers = []
    for i, q in enumerate(questions):
        print(f"\nQ{i+1} [{q['answer_type']}]: {q['question']}")
        raw = input(">>> ")
        if q["answer_type"] == "int":
            answers.append(int(raw))
        else:
            answers.append(raw)

# Step 4: Assemble payload
payload = {
    "token": token,
    "answers": answers,
}
print(f"\nPayload: {json.dumps(payload)}")

# Step 5: POST /fastest-fingers-first
print("\n" + "=" * 60)
print("Step 5: POST /fastest-fingers-first")
print("=" * 60)
resp2 = requests.post(
    f"{BASE_URL}/fastest-fingers-first",
    json=payload,
    headers=HEADERS,
    cookies=COOKIES,
)
print(f"Status: {resp2.status_code}")
print(f"Response: {resp2.text}")
