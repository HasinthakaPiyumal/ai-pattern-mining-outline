# Cluster 55

def get_user_input():
    print('Welcome to EvaDB!')
    print("Enter your image prompts one by one; type 'exit' to stop entering prompts.")
    print('========================================')
    prompts = []
    prompt = None
    while True:
        prompt = input('Enter prompt: ').strip()
        if prompt in ['Exit', 'exit', 'EXIT']:
            break
        prompts.append(prompt)
        print(prompt)
    return prompts

def set_replicate_token() -> None:
    key = input('Enter your Replicate API Token: ').strip()
    try:
        os.environ['REPLICATE_API_TOKEN'] = key
        print('Environment variable set successfully.')
    except Exception as e:
        print('❗️ Session ended with an error.')
        print(e)
        print('===========================================')

