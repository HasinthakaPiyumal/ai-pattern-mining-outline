# Cluster 2

class JWT_AI_MODEL:

    @staticmethod
    def BardAI(key: str, jwt_data: Any) -> str:
        prompt = f"""\n        Perform a comprehensive analysis on the provided JWT token. The analysis output must be in a JSON format according to the provided output structure. Ensure accuracy for inclusion in a penetration testing report.\n        Follow these guidelines:\n        1) Analyze the JWT token from a pentester's perspective\n        2) Keep the final output minimal while adhering to the given format\n        3) Highlight JWT-specific details and enumerate possible attacks and vulnerabilities\n        5) For the output "Algorithm Used" value use the Algorithm value from the JWT data.\n        6) For the output "Header" value use the Header value from the JWT data.\n        7) For the "Payload" Use the decoded payloads as a reference and then analyze any attack endpoints.\n        8) For "Signature" mention the signatures discovered.\n        9) List a few endpoints you feel are vulnerable for "VulnerableEndpoints"\n\n        The output format:\n        {{\n            "Algorithm Used": "",\n            "Header": "",\n            "Payload": "",\n            "Signature": "",\n            "PossibleAttacks": "",\n            "VulnerableEndpoints": ""\n        }}\n\n        JWT Token Data to be analyzed: {jwt_data}\n        """
        url = 'https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText?key=' + key
        headers = {'Content-Type': 'application/json'}
        data = {'prompt': {'text': prompt}}
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            generated_text = response.json()
            jwt_analysis_data = jwt_ai_data_regex(str(generated_text))
            print(jwt_analysis_data)
            return jwt_analysis_data
        else:
            print('Error: Unable to generate text. Status Code:', response.status_code)
            return 'None'

    @staticmethod
    def llama_AI(self, jwt_data: str, mode: str, lkey, lendpoint):
        api_url = 'http://localhost:5000/api/chatbot'
        user_instruction = '\n            Perform a comprehensive analysis on the provided JWT token. The JWT analysis output must be in a asked format according to the provided output structure. Ensure accuracy for inclusion in a penetration testing report.\n            Follow these guidelines:\n            1) Analyze the JWT token from a pentester\'s perspective\n            2) Keep the final output minimal while adhering to the given format\n            3) Highlight JWT-specific details and enumerate possible attacks\n\n            The output format:\n            "Header":\n            - List the JWT header details and security views on them\n            "Payload":\n            - List the JWT payload details and security views on them\n            "Signature":\n            - Provide insights on the JWT signature\n            "PossibleAttacks":\n            - List possible JWT exploits and attacks\n        '
        user_message = f'\n            JWT Token Data to be analyzed: {jwt_data}\n        '
        model_name = 'TheBloke/Llama-2-7B-Chat-GGML'
        file_name = 'llama-2-7b-chat.ggmlv3.q4_K_M.bin'
        if mode == 'local':
            bot_response = self.chat_with_api(api_url, user_message, user_instruction, model_name, file_name)
        elif mode == 'runpod':
            prompt = f'[INST] <<SYS>> {user_instruction}<</SYS>> JWT Token Data to be analyzed: {user_message} [/INST]'
            bot_response = self.llama_runpod_api(prompt, lkey, lendpoint)
        bot_response = self.chat_with_api(api_url, user_message, user_instruction, model_name, file_name)
        print('test')
        if bot_response:
            return bot_response

    @staticmethod
    def gpt_ai(analyze: str, api_key: Optional[str]) -> str:
        openai.api_key = api_key
        prompt = f"""\n        Perform a comprehensive analysis on the provided JWT token. The analysis output must be in a JSON format according to the provided output structure. Ensure accuracy for inclusion in a penetration testing report.\n        Follow these guidelines:\n        1) Analyze the JWT token from a pentester's perspective\n        2) Keep the final output minimal while adhering to the given format\n        3) Highlight JWT-specific details and enumerate possible attacks and vulnerabilities\n        5) For the output "Algorithm Used" value use the Algorithm value from the JWT data.\n        6) For the output "Header" value use the Header value from the JWT data.\n        7) For the "Payload" Use the decoded payloads as a reference and then analyze any attack endpoints.\n        8) For "Signature" mention the signatures discovered.\n        9) List a few endpoints you feel are vulnerable for "VulnerableEndpoints"\n\n        The output format:\n        {{\n            "Algorithm Used": "",\n            "Header": "",\n            "Payload": "",\n            "Signature": "",\n            "PossibleAttacks": "",\n            "VulnerableEndpoints": ""\n        }}\n\n        JWT Token Data to be analyzed: {analyze}\n        """
        try:
            messages = [{'content': prompt, 'role': 'user'}]
            response = openai.ChatCompletion.create(model=model_engine, messages=messages, max_tokens=1024, n=1, stop=None)
            response = response['choices'][0]['message']['content']
            rsp = str(response)
            return rsp
        except KeyboardInterrupt:
            print('Bye')
            quit()

def jwt_ai_data_regex(json_string: str) -> Any:
    header_pattern = '"Header": \\{\\s*"alg": "(.*?)",\\s*"typ": "(.*?)"\\s*\\}'
    payload_pattern = '"Payload": \\{\\s*"iss": "(.*?)",\\s*"sub": "(.*?)",\\s*"aud": "(.*?)",\\s*"exp": "(.*?)",\\s*"nbf": "(.*?)",\\s*"iat": "(.*?)"\\s*\\}'
    signature_pattern = '"Signature": "(.*?)"'
    possible_attacks_pattern = '"PossibleAttacks": "(.*?)"'
    vulnerable_endpoints_pattern = '"VulnerableEndpoints": "(.*?)"'
    header = {}
    payload = {}
    signature = ''
    possible_attacks = ''
    vulnerable_endpoints = ''
    match_header = re.search(header_pattern, json_string)
    if match_header:
        header = {'alg': match_header.group(1), 'typ': match_header.group(2)}
    match_payload = re.search(payload_pattern, json_string)
    if match_payload:
        payload = {'iss': match_payload.group(1), 'sub': match_payload.group(2), 'aud': match_payload.group(3), 'exp': match_payload.group(4), 'nbf': match_payload.group(5), 'iat': match_payload.group(6)}
    match_signature = re.search(signature_pattern, json_string)
    if match_signature:
        signature = match_signature.group(1)
    match_attacks = re.search(possible_attacks_pattern, json_string)
    if match_attacks:
        possible_attacks = match_attacks.group(1)
    match_endpoints = re.search(vulnerable_endpoints_pattern, json_string)
    if match_endpoints:
        vulnerable_endpoints = match_endpoints.group(1)
    data = {'Header': header, 'Payload': payload, 'Signature': signature, 'PossibleAttacks': possible_attacks, 'VulnerableEndpoints': vulnerable_endpoints}
    json_output = json.dumps(data)
    return json_output

class JWT_AI_MODEL:

    @staticmethod
    def BardAI(key: str, jwt_data: Any) -> str:
        prompt = f"""\n        Perform a comprehensive analysis on the provided JWT token. The analysis output must be in a JSON format according to the provided output structure. Ensure accuracy for inclusion in a penetration testing report.\n        Follow these guidelines:\n        1) Analyze the JWT token from a pentester's perspective\n        2) Keep the final output minimal while adhering to the given format\n        3) Highlight JWT-specific details and enumerate possible attacks and vulnerabilities\n        5) For the output "Algorithm Used" value use the Algorithm value from the JWT data.\n        6) For the output "Header" value use the Header value from the JWT data.\n        7) For the "Payload" Use the decoded payloads as a reference and then analyze any attack endpoints.\n        8) For "Signature" mention the signatures discovered.\n        9) List a few endpoints you feel are vulnerable for "VulnerableEndpoints"\n\n        The output format:\n        {{\n            "Algorithm Used": "",\n            "Header": "",\n            "Payload": "",\n            "Signature": "",\n            "PossibleAttacks": "",\n            "VulnerableEndpoints": ""\n        }}\n\n        JWT Token Data to be analyzed: {jwt_data}\n        """
        url = 'https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText?key=' + key
        headers = {'Content-Type': 'application/json'}
        data = {'prompt': {'text': prompt}}
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            generated_text = response.json()
            jwt_analysis_data = jwt_ai_data_regex(str(generated_text))
            print(jwt_analysis_data)
            return jwt_analysis_data
        else:
            print('Error: Unable to generate text. Status Code:', response.status_code)
            return 'None'

    @staticmethod
    def llama_AI(self, jwt_data: str, mode: str, lkey, lendpoint):
        api_url = 'http://localhost:5000/api/chatbot'
        user_instruction = '\n            Perform a comprehensive analysis on the provided JWT token. The JWT analysis output must be in a asked format according to the provided output structure. Ensure accuracy for inclusion in a penetration testing report.\n            Follow these guidelines:\n            1) Analyze the JWT token from a pentester\'s perspective\n            2) Keep the final output minimal while adhering to the given format\n            3) Highlight JWT-specific details and enumerate possible attacks\n\n            The output format:\n            "Header":\n            - List the JWT header details and security views on them\n            "Payload":\n            - List the JWT payload details and security views on them\n            "Signature":\n            - Provide insights on the JWT signature\n            "PossibleAttacks":\n            - List possible JWT exploits and attacks\n        '
        user_message = f'\n            JWT Token Data to be analyzed: {jwt_data}\n        '
        model_name = 'TheBloke/Llama-2-7B-Chat-GGML'
        file_name = 'llama-2-7b-chat.ggmlv3.q4_K_M.bin'
        if mode == 'local':
            bot_response = self.chat_with_api(api_url, user_message, user_instruction, model_name, file_name)
        elif mode == 'runpod':
            prompt = f'[INST] <<SYS>> {user_instruction}<</SYS>> JWT Token Data to be analyzed: {user_message} [/INST]'
            bot_response = self.llama_runpod_api(prompt, lkey, lendpoint)
        bot_response = self.chat_with_api(api_url, user_message, user_instruction, model_name, file_name)
        print('test')
        if bot_response:
            return bot_response

    @staticmethod
    def gpt_ai(analyze: str, api_key: Optional[str]) -> str:
        openai.api_key = api_key
        prompt = f"""\n        Perform a comprehensive analysis on the provided JWT token. The analysis output must be in a JSON format according to the provided output structure. Ensure accuracy for inclusion in a penetration testing report.\n        Follow these guidelines:\n        1) Analyze the JWT token from a pentester's perspective\n        2) Keep the final output minimal while adhering to the given format\n        3) Highlight JWT-specific details and enumerate possible attacks and vulnerabilities\n        5) For the output "Algorithm Used" value use the Algorithm value from the JWT data.\n        6) For the output "Header" value use the Header value from the JWT data.\n        7) For the "Payload" Use the decoded payloads as a reference and then analyze any attack endpoints.\n        8) For "Signature" mention the signatures discovered.\n        9) List a few endpoints you feel are vulnerable for "VulnerableEndpoints"\n\n        The output format:\n        {{\n            "Algorithm Used": "",\n            "Header": "",\n            "Payload": "",\n            "Signature": "",\n            "PossibleAttacks": "",\n            "VulnerableEndpoints": ""\n        }}\n\n        JWT Token Data to be analyzed: {analyze}\n        """
        try:
            messages = [{'content': prompt, 'role': 'user'}]
            response = openai.ChatCompletion.create(model=model_engine, messages=messages, max_tokens=1024, n=1, stop=None)
            response = response['choices'][0]['message']['content']
            rsp = str(response)
            return rsp
        except KeyboardInterrupt:
            print('Bye')
            quit()

