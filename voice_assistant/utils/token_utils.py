import tiktoken

def count_tokens(messages, model):
    """
    Counts tokens used in a list of messages for OpenAI Chat API.
    Supports gpt-3.5-turbo and gpt-4 models.

    This is only an estimate of the number of tokens need for your prompt before sending it to the API. 
    The response already includes the tokens used for the response (input and output).
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Model not recognized. Defaulting to cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    token_count = 0
    for message in messages:
        # Every message follows this format: {"role": ..., "content": ...}
        token_count += 4  # Every message metadata takes 4 tokens (system/user/assistant + delimiters)
        for key, value in message.items():
            token_count += len(encoding.encode(value))
    token_count += 2  # For the reply priming
    return token_count