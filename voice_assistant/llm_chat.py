# llm_chat.py
import openai
from config import OPENAI_API_KEY
from utils.token_utils import count_tokens

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_chatgpt_response(user_prompt):
    messages = [
        {"role": "system", "content": "You are a helpful voice assistant."},
        {"role": "user", "content": user_prompt}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Use gpt-3.5-turbo for cheaper
        messages=messages
    )
    
    ai_response = response.choices[0].message.content
    usage = response.usage

    return {
        "response": ai_response,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens
    }