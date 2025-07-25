# main.py
# Terminal-based ChatGPT assistant using OpenAI API (openai>=1.0.0)
from token_utils import count_tokens    

import openai
try:
    from config import OPENAI_API_KEY
except ImportError:
    print("Error: Could not import OPENAI_API_KEY from config.py.")
    print("Please create a config.py file with your OpenAI API key.")
    exit(1)

# Set up the OpenAI client with your API key
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def chat_with_gpt():
    print("Welcome to the terminal ChatGPT assistant!")
    print("Type your message and press Enter. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input}
                ]
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Or "gpt-4" if you have access
                messages=messages
            )
            ai_message = response.choices[0].message.content
            print(f"ChatGPT: {ai_message}\n\n")
            # Print the token usage from the API response
            print("Prompt tokens (input):", response.usage.prompt_tokens)
            print("Completion tokens (output):", response.usage.completion_tokens)
            print("Total tokens:", response.usage.total_tokens)
        except Exception as e:
            print(f"Error communicating with OpenAI: {e}")

if __name__ == "__main__":
    chat_with_gpt() 