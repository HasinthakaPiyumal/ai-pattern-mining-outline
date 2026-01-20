# Cluster 70

def call2():
    from openai import OpenAI
    openai_api_key = 'EMPTY'
    openai_api_base = 'http://10.140.24.142:29500/v1'
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    chat_response = client.chat.completions.create(model='../models/Qwen1.5-14B-Chat/', messages=[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'Tell me a joke.'}])
    print('Chat response:', chat_response)

