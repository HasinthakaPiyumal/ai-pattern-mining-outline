# Cluster 11

class NMAP_AI_MODEL:

    @staticmethod
    def BardAI(key: str, data: Any) -> str:
        prompt = f'\n            Do a NMAP scan analysis on the provided NMAP scan information\n            The NMAP output must return in a JSON format accorging to the provided\n            output format. The data must be accurate in regards towards a pentest report.\n            The data must follow the following rules:\n            1) The NMAP scans must be done from a pentester point of view\n            2) The final output must be minimal according to the format given.\n            3) The final output must be kept to a minimal.\n            4) If a value not found in the scan just mention an empty string.\n            5) Analyze everything even the smallest of data.\n            6) Completely analyze the data provided and give a confirm answer using the output format.\n\n            The output format:\n            {{\n                "critical score": [""],\n                "os information": [""],\n                "open ports": [""],\n                "open services": [""],\n                "vulnerable service": [""],\n                "found cve": [""]\n            }}\n\n            NMAP Data to be analyzed: {data}\n            '
        url = 'https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText?key=' + key
        headers = {'Content-Type': 'application/json'}
        data = {'prompt': {'text': prompt}}
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            generated_text = response.json()
            return nmap_ai_data_regex(str(generated_text))
        else:
            print('Error: Unable to generate text. Status Code:', response.status_code)
            return 'None'

    @staticmethod
    def Llama_AI(data: str, mode: str, lkey: str, lendpoint: str) -> Any:
        api_url = 'http://localhost:5000/api/chatbot'
        user_instruction = '\n        Do a NMAP scan analysis on the provided NMAP scan information. The NMAP output must return in a asked format accorging to the provided output format. The data must be accurate in regards towards a pentest report.\n        The data must follow the following rules:\n        1) The NMAP scans must be done from a pentester point of view\n        2) The final output must be minimal according to the format given.\n        3) The final output must be kept to a minimal.\n        4) If a value not found in the scan just mention an empty string.\n        5) Analyze everything even the smallest of data.\n        6) Completely analyze the data provided and give a confirm answer using the output format.\n        7) mention all the data you found in the output format provided so that regex can be used on it.\n        8) avoid unnecessary explaination.\n        9) the critical score must be calculated based on the CVE if present or by the nature of the services open\n        10) the os information must contain the OS used my the target.\n        11) the open ports must include all the open ports listed in the data[tcp] and varifying if it by checking its states value.  you should not negect even one open port.\n        12) the vulnerable services can be determined via speculation of the service nature or by analyzing the CVE\'s found.\n        The output format:\n            critical score:\n            - Give info on the criticality\n            "os information":\n            - List out the OS information\n            "open ports and services":\n            - List open ports\n            - List open ports services\n            "vulnerable service":\n            - Based on CVEs or nature of the ports opened list the vulnerable services\n            "found cve":\n            - List the CVE\'s found and list the main issues.\n        '
        user_message = f'\n            NMAP Data to be analyzed: {data}\n        '
        model_name = 'TheBloke/Llama-2-7B-Chat-GGML'
        file_name = 'llama-2-7b-chat.ggmlv3.q4_K_M.bin'
        if mode == 'local':
            bot_response = chat_with_api(api_url, user_message, user_instruction, model_name, file_name)
        elif mode == 'runpod':
            prompt = f'[INST] <<SYS>> {user_instruction}<</SYS>> NMAP Data to be analyzed: {user_message} [/INST]'
            bot_response = llama_runpod_api(prompt, lkey, lendpoint)
        if bot_response:
            return bot_response

    @staticmethod
    def GPT_AI(key: str, data: Any) -> str:
        openai.api_key = key
        try:
            prompt = f'\n            Do a NMAP scan analysis on the provided NMAP scan information\n            The NMAP output must return in a JSON format accorging to the provided\n            output format. The data must be accurate in regards towards a pentest report.\n            The data must follow the following rules:\n            1) The NMAP scans must be done from a pentester point of view\n            2) The final output must be minimal according to the format given.\n            3) The final output must be kept to a minimal.\n            4) If a value not found in the scan just mention an empty string.\n            5) Analyze everything even the smallest of data.\n            6) Completely analyze the data provided and give a confirm answer using the output format.\n\n            The output format:\n            {{\n                "critical score": [""],\n                "os information": [""],\n                "open ports": [""],\n                "open services": [""],\n                "vulnerable service": [""],\n                "found cve": [""]\n            }}\n\n            NMAP Data to be analyzed: {data}\n            '
            messages = [{'content': prompt, 'role': 'assistant'}]
            response = openai.ChatCompletion.create(model=model_engine, messages=messages, max_tokens=2500, n=1, stop=None)
            response = response['choices'][0]['message']['content']
            rsp = str(response)
            return str(nmap_ai_data_regex(rsp))
        except KeyboardInterrupt:
            print('Bye')
            quit()

def chat_with_api(api_url: str, user_message: str, user_instruction: str, model_name: str, file_name: str=None) -> Any:
    data = {'user_message': user_message, 'model_name': model_name, 'file_name': file_name, 'user_instruction': user_instruction}
    response = requests.post(api_url, json=data)
    if response.status_code == 200:
        return response.json()['bot_response']
    else:
        print(f'Error: {response.status_code} - {response.text}')
        return None

def llama_runpod_api(prompt: str, lkey: str, lendpoint: str) -> Any:
    url = f'https://api.runpod.ai/v2/{lendpoint}/runsync'
    payload = json.dumps({'input': {'prompt': prompt, 'max_new_tokens': 4500, 'temperature': 0.9, 'top_k': 50, 'top_p': 0.7, 'repetition_penalty': 1.2, 'batch_size': 8, 'stop': ['</s>']}})
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {lkey}'}
    response = requests.request('POST', url, headers=headers, data=payload)
    response_t = json.loads(response.text)
    return response_t['output']

class NMAP_AI_MODEL:

    @staticmethod
    def BardAI(key: str, data: Any) -> str:
        prompt = f'\n            Do a NMAP scan analysis on the provided NMAP scan information\n            The NMAP output must return in a JSON format accorging to the provided\n            output format. The data must be accurate in regards towards a pentest report.\n            The data must follow the following rules:\n            1) The NMAP scans must be done from a pentester point of view\n            2) The final output must be minimal according to the format given.\n            3) The final output must be kept to a minimal.\n            4) If a value not found in the scan just mention an empty string.\n            5) Analyze everything even the smallest of data.\n            6) Completely analyze the data provided and give a confirm answer using the output format.\n\n            The output format:\n            {{\n                "critical score": [""],\n                "os information": [""],\n                "open ports": [""],\n                "open services": [""],\n                "vulnerable service": [""],\n                "found cve": [""]\n            }}\n\n            NMAP Data to be analyzed: {data}\n            '
        url = 'https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText?key=' + key
        headers = {'Content-Type': 'application/json'}
        data = {'prompt': {'text': prompt}}
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            generated_text = response.json()
            return nmap_ai_data_regex(str(generated_text))
        else:
            print('Error: Unable to generate text. Status Code:', response.status_code)
            return 'None'

    @staticmethod
    def Llama_AI(data: str, mode: str, lkey: str, lendpoint: str) -> Any:
        api_url = 'http://localhost:5000/api/chatbot'
        user_instruction = '\n        Do a NMAP scan analysis on the provided NMAP scan information. The NMAP output must return in a asked format accorging to the provided output format. The data must be accurate in regards towards a pentest report.\n        The data must follow the following rules:\n        1) The NMAP scans must be done from a pentester point of view\n        2) The final output must be minimal according to the format given.\n        3) The final output must be kept to a minimal.\n        4) If a value not found in the scan just mention an empty string.\n        5) Analyze everything even the smallest of data.\n        6) Completely analyze the data provided and give a confirm answer using the output format.\n        7) mention all the data you found in the output format provided so that regex can be used on it.\n        8) avoid unnecessary explaination.\n        9) the critical score must be calculated based on the CVE if present or by the nature of the services open\n        10) the os information must contain the OS used my the target.\n        11) the open ports must include all the open ports listed in the data[tcp] and varifying if it by checking its states value.  you should not negect even one open port.\n        12) the vulnerable services can be determined via speculation of the service nature or by analyzing the CVE\'s found.\n        The output format:\n            critical score:\n            - Give info on the criticality\n            "os information":\n            - List out the OS information\n            "open ports and services":\n            - List open ports\n            - List open ports services\n            "vulnerable service":\n            - Based on CVEs or nature of the ports opened list the vulnerable services\n            "found cve":\n            - List the CVE\'s found and list the main issues.\n        '
        user_message = f'\n            NMAP Data to be analyzed: {data}\n        '
        model_name = 'TheBloke/Llama-2-7B-Chat-GGML'
        file_name = 'llama-2-7b-chat.ggmlv3.q4_K_M.bin'
        if mode == 'local':
            bot_response = chat_with_api(api_url, user_message, user_instruction, model_name, file_name)
        elif mode == 'runpod':
            prompt = f'[INST] <<SYS>> {user_instruction}<</SYS>> NMAP Data to be analyzed: {user_message} [/INST]'
            bot_response = llama_runpod_api(prompt, lkey, lendpoint)
        if bot_response:
            return bot_response

    @staticmethod
    def GPT_AI(key: str, data: Any) -> str:
        openai.api_key = key
        try:
            prompt = f'\n            Do a NMAP scan analysis on the provided NMAP scan information\n            The NMAP output must return in a JSON format accorging to the provided\n            output format. The data must be accurate in regards towards a pentest report.\n            The data must follow the following rules:\n            1) The NMAP scans must be done from a pentester point of view\n            2) The final output must be minimal according to the format given.\n            3) The final output must be kept to a minimal.\n            4) If a value not found in the scan just mention an empty string.\n            5) Analyze everything even the smallest of data.\n            6) Completely analyze the data provided and give a confirm answer using the output format.\n\n            The output format:\n            {{\n                "critical score": [""],\n                "os information": [""],\n                "open ports": [""],\n                "open services": [""],\n                "vulnerable service": [""],\n                "found cve": [""]\n            }}\n\n            NMAP Data to be analyzed: {data}\n            '
            messages = [{'content': prompt, 'role': 'assistant'}]
            response = openai.ChatCompletion.create(model=model_engine, messages=messages, max_tokens=2500, n=1, stop=None)
            response = response['choices'][0]['message']['content']
            rsp = str(response)
            return str(nmap_ai_data_regex(rsp))
        except KeyboardInterrupt:
            print('Bye')
            quit()

