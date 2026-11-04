import os

from openai import OpenAI

# Configure the API and key (using DeepSeek as an example)
def generate(message, gpt="deepseek", temperature=0):

    if gpt == "deepseek":
        MODEL = "deepseek-chat"
        OPENAI_API_BASE = "https://api.deepseek.com"
        OPENAI_API_KEY = os.environ["DEEPSEEK_API_KEY"]
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

    elif gpt == "openai":
        MODEL = "gpt-4o"
        OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
        client = OpenAI(api_key=OPENAI_API_KEY)

    else:
        raise ValueError(f"Unsupported API provider: {gpt}")

    print('start generating')
    response = client.chat.completions.create(
        model=MODEL,
        messages=message,
        stream=False,
        temperature=temperature,
    )
    print('end generating')

    return response.choices[0].message.content

