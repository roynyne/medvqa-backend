# app/test_llm_guard.py

from llm_guard import guard_and_rewrite_question

# Questions to test
test_questions = [
    "what is love?",
    "banana?",
    "joke please",
    "what's this?",
    "is this a chest xray?",
    "how are you?",
    "CT or MRI?",
    "yes?",
    "abnormal?"
]

for q in test_questions:
    result = guard_and_rewrite_question(q)
    print(f"\n🔎 Original: {q}")
    print(f"🛡️ Guard Result: {result}")
