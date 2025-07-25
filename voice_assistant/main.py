# main_voice.py
from mic.mic_input import get_audio_input
from llm_chat import get_chatgpt_response

def run_voice_chat():
    try:
        while True:
            user_prompt = get_audio_input()
            if not user_prompt:
                print("Didn't catch that. Try again.")
                continue
            
            print(f"\nYou said: {user_prompt}")
            result = get_chatgpt_response(user_prompt)

            print(f"\n🤖 ChatGPT: {result['response']}\n\n")
            print(f"📊 Token Usage → Prompt: {result['prompt_tokens']}, Completion: {result['completion_tokens']}, Total: {result['total_tokens']}\n\n")

    except KeyboardInterrupt:
        print("\n👋 Exiting. Goodbye!")

if __name__ == "__main__":
    run_voice_chat()