import sys
import os
import dotenv
from groq import Groq

print(sys.executable)
dotenv.load_dotenv()


# 1. Chat Models
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You're an assistant knowledgeable about healthcare. Only answer healthcare-related questions.",
        },
        {
            "role": "user",
            "content": "How do I care for a wound?",
        }
    ],
    model="llama3-8b-8192",
    temperature=0.5,
    max_tokens=50
)
# print(chat_completion.choices)
# print("\n")
# print(chat_completion.choices[0])
# print("\n")
# print(chat_completion.choices[1])
# print("\n")
print(chat_completion.choices[0].message.content)
