# Google Gemini API (send a prompt to API and returns the generated answer)

import os
from google import genai
from google.genai import types

# config
try: 
    # get API Key from env
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)

except ValueError as e:
    print(e) 

except Exception as e:
    print(f"An unexpected error occurred during configuration: {e}")

# define settings
MODEL_NAME = "gemini-2.5-flash" 
GENERATION_CONFIG = types.GenerateContentConfig(
    temperature=0.7,
    max_output_tokens=10000
)

# safety settings 
SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]

def call_llm(system_prompt: str, user_prompt: str) -> str:
    '''
    Sends a prompt to Google Gemini API and returns the generated answer
    system_prompt: Instruction from the bot
    user_prompt: The user's query, including RAG context
    Returns the text response from LLM
    '''
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=10000,  # Use the higher limit
                system_instruction=system_prompt,
                safety_settings=SAFETY_SETTINGS
            )
        )
        
        if not response.candidates:
            return "I'm sorry, the model did not provide a response. Please try again."
        
        finish_reason = response.candidates[0].finish_reason

        # Check if we have text content first
        if not response.text:
            print(f"--- WARNING: No text in response. Finish reason: {finish_reason} ---")
            return "I'm sorry, I couldn't generate a response. Please try again."

        if finish_reason == types.FinishReason.STOP:
            return response.text.strip()
        elif finish_reason == types.FinishReason.SAFETY:
            print("--- WARNING: Response blocked by safety filter. ---")
            return "I'm sorry, I can't respond to that query. Please rephrase your request."
        elif finish_reason == types.FinishReason.MAX_TOKENS:
            print("--- WARNING: Response truncated due to max tokens. ---")
            # Return the partial response
            return response.text.strip() + "..."
        else:
            print(f"--- WARNING: Model finished with reason: {finish_reason} ---")
            # Still try to return the text if available
            return response.text.strip() if response.text else "I'm sorry, I ran into an issue. Please try again."
    
    except Exception as e:
        print(f"\n--- ERROR during Gemini API call ---")
        print(f"Error: {e}")
        return "I'm sorry, I ran into a system error. Please try again."