# OPENAI API call (send a prompt to OpenAI and returns the generated answer)

from openai import OpenAI # library to interact with OpenAI REST API, send request & receive response from GPT models
import os

client = OpenAI(api_key = os.getenv("OPENAI_API_KEY")) 

def call_llm(system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {'role': "system", "content": system_prompt},
            {'role': "user", "content": user_prompt}
        ],
        temperature = 0.7,
        max_tokens = 150,
    )
    return resp.choices[0].message.content.strip()