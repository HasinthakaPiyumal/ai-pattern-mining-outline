# Cluster 13

class DNS_AI_MODEL:

    @staticmethod
    def BardAI(key: str, data: Any) -> str:
        prompt = f'\n            Do a DNS analysis on the provided DNS scan information\n            The DNS output must return in a JSON format accorging to the provided\n            output format. The data must be accurate in regards towards a pentest report.\n            The data must follow the following rules:\n            1) The DNS scans must be done from a pentester point of view\n            2) The final output must be minimal according to the format given\n            3) The final output must be kept to a minimal\n\n            The output format:\n            {{\n                "A": [""],\n                "AAA": [""],\n                "NS": [""],\n                "MX": [""],\n                "PTR": [""],\n                "SOA": [""],\n                "TXT": [""]\n            }}\n            DNS Data to be analyzed: {data}\n            '
        url = 'https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText?key=' + key
        headers = {'Content-Type': 'application/json'}
        data = {'prompt': {'text': prompt}}
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            generated_text = response.json()
            data = dns_ai_data_regex(str(generated_text))
            print(data)
            return dns_ai_data_regex(str(generated_text))
        else:
            print('Error: Unable to generate text. Status Code:', response.status_code)
            return 'None'

    @staticmethod
    def llama_AI(self, data: str, mode: str, lkey, lendpoint):
        api_url = 'http://localhost:5000/api/chatbot'
        user_instruction = '\n            Do a DNS scan analysis on the provided DNS scan information. The DNS output must return in a asked format accorging to the provided output format. The data must be accurate in regards towards a pentest report.\n            The data must follow the following rules:\n            1) The DNS scans must be done from a pentester point of view\n            2) The final output must be minimal according to the format given\n            3) The final output must be kept to a minimal\n            4) So the analysis and provide your view according to the given format\n            5) Remember to provide views as a security engineer or an security analyst.\n            The output format:\n            "A":\n            - List the A records and security views on them\n            "AAA":\n            - List the AAA records and security views on them\n            "NS":\n            - List the NS records and security views on them\n            "MX":\n            - List the MX records and security views on them\n            "PTR":\n            - List the PTR records and security views on them\n            "SOA":\n            - List the SOA records and security views on them\n            "TXT":\n            - List the TXT records and security views on them\n        '
        user_message = f'\n            DNS Data to be analyzed: {data}\n        '
        model_name = 'TheBloke/Llama-2-7B-Chat-GGML'
        file_name = 'llama-2-7b-chat.ggmlv3.q4_K_M.bin'
        if mode == 'local':
            bot_response = self.chat_with_api(api_url, user_message, user_instruction, model_name, file_name)
        elif mode == 'runpod':
            prompt = f'[INST] <<SYS>> {user_instruction}<</SYS>> NMAP Data to be analyzed: {user_message} [/INST]'
            bot_response = self.llama_runpod_api(prompt, lkey, lendpoint)
        bot_response = self.chat_with_api(api_url, user_message, user_instruction, model_name, file_name)
        print('test')
        if bot_response:
            return bot_response

    @staticmethod
    def gpt_ai(analyze: str, key: Optional[str]) -> str:
        openai.api_key = key
        prompt = f'\n        Do a DNS analysis on the provided DNS scan information\n        The DNS output must return in a JSON format accorging to the provided\n        output format. The data must be accurate in regards towards a pentest report.\n        The data must follow the following rules:\n        1) The DNS scans must be done from a pentester point of view\n        2) The final output must be minimal according to the format given\n        3) The final output must be kept to a minimal\n\n        The output format:\n        {{\n            "A": [""],\n            "AAA": [""],\n            "NS": [""],\n            "MX": [""],\n            "PTR": [""],\n            "SOA": [""],\n            "TXT": [""]\n        }}\n\n        DNS Data to be analyzed: {analyze}\n        '
        try:
            messages = [{'content': prompt, 'role': 'user'}]
            response = openai.ChatCompletion.create(model=model_engine, messages=messages, max_tokens=1024, n=1, stop=None)
            response = response['choices'][0]['message']['content']
            return dns_ai_data_regex(str(response))
        except KeyboardInterrupt:
            print('Bye')
            quit()

def dns_ai_data_regex(json_string: str) -> Any:
    A_pattern = '"A": \\["(.*?)"\\]'
    AAA_pattern = '"AAAA": \\["(.*?)"\\]'
    NS_pattern = '"NS": \\["(.*?)"\\]'
    MX_pattern = '"MX": \\["(.*?)"\\]'
    PTR_pattern = '"PTR": \\["(.*?)"\\]'
    SOA_pattern = '"SOA": \\["(.*?)"\\]'
    TXT_pattern = '"TXT": \\["(.*?)"\\]'
    Reverse_DNS_pattern = '"Reverse_DNS": \\{ "IP_Address": "(.*?)", "Domain": "(.*?)" \\}'
    Zone_Transfer_Scan_pattern = '"Zone_Transfer_Scan": \\{ "Allowed": (.*?), "Name_Servers": \\["(.*?)"\\] \\}'
    A = None
    AAA = None
    NS = None
    MX = None
    PTR = None
    SOA = None
    TXT = None
    Reverse_DNS_IP = None
    Reverse_DNS_Domain = None
    Zone_Transfer_Allowed = None
    Zone_Transfer_Name_Servers = None
    match = re.search(A_pattern, json_string)
    if match:
        A = match.group(1)
    match = re.search(AAA_pattern, json_string)
    if match:
        AAA = match.group(1)
    match = re.search(NS_pattern, json_string)
    if match:
        NS = match.group(1)
    match = re.search(MX_pattern, json_string)
    if match:
        MX = match.group(1)
    match = re.search(PTR_pattern, json_string)
    if match:
        PTR = match.group(1)
    match = re.search(SOA_pattern, json_string)
    if match:
        SOA = match.group(1)
    match = re.search(TXT_pattern, json_string)
    if match:
        TXT = match.group(1)
    match = re.search(Reverse_DNS_pattern, json_string)
    if match:
        Reverse_DNS_IP = match.group(1)
        Reverse_DNS_Domain = match.group(2)
    match = re.search(Zone_Transfer_Scan_pattern, json_string)
    if match:
        Zone_Transfer_Allowed = bool(match.group(1))
        Zone_Transfer_Name_Servers = match.group(2)
    data = {'DNS_Records': {'A': A, 'AAAA': AAA, 'NS': NS, 'MX': MX, 'PTR': PTR, 'SOA': SOA, 'TXT': TXT}, 'Reverse_DNS': {'IP_Address': Reverse_DNS_IP, 'Domain': Reverse_DNS_Domain}, 'Zone_Transfer_Scan': {'Allowed': Zone_Transfer_Allowed, 'Name_Servers': [Zone_Transfer_Name_Servers] if Zone_Transfer_Name_Servers else []}}
    json_output = json.dumps(data)
    return json_output

class DNS_AI_MODEL:

    @staticmethod
    def BardAI(key: str, data: Any) -> str:
        prompt = f'\n            Do a DNS analysis on the provided DNS scan information\n            The DNS output must return in a JSON format accorging to the provided\n            output format. The data must be accurate in regards towards a pentest report.\n            The data must follow the following rules:\n            1) The DNS scans must be done from a pentester point of view\n            2) The final output must be minimal according to the format given\n            3) The final output must be kept to a minimal\n\n            The output format:\n            {{\n                "A": [""],\n                "AAA": [""],\n                "NS": [""],\n                "MX": [""],\n                "PTR": [""],\n                "SOA": [""],\n                "TXT": [""]\n            }}\n            DNS Data to be analyzed: {data}\n            '
        url = 'https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText?key=' + key
        headers = {'Content-Type': 'application/json'}
        data = {'prompt': {'text': prompt}}
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            generated_text = response.json()
            data = dns_ai_data_regex(str(generated_text))
            print(data)
            return dns_ai_data_regex(str(generated_text))
        else:
            print('Error: Unable to generate text. Status Code:', response.status_code)
            return 'None'

    @staticmethod
    def llama_AI(self, data: str, mode: str, lkey, lendpoint):
        api_url = 'http://localhost:5000/api/chatbot'
        user_instruction = '\n            Do a DNS scan analysis on the provided DNS scan information. The DNS output must return in a asked format accorging to the provided output format. The data must be accurate in regards towards a pentest report.\n            The data must follow the following rules:\n            1) The DNS scans must be done from a pentester point of view\n            2) The final output must be minimal according to the format given\n            3) The final output must be kept to a minimal\n            4) So the analysis and provide your view according to the given format\n            5) Remember to provide views as a security engineer or an security analyst.\n            The output format:\n            "A":\n            - List the A records and security views on them\n            "AAA":\n            - List the AAA records and security views on them\n            "NS":\n            - List the NS records and security views on them\n            "MX":\n            - List the MX records and security views on them\n            "PTR":\n            - List the PTR records and security views on them\n            "SOA":\n            - List the SOA records and security views on them\n            "TXT":\n            - List the TXT records and security views on them\n        '
        user_message = f'\n            DNS Data to be analyzed: {data}\n        '
        model_name = 'TheBloke/Llama-2-7B-Chat-GGML'
        file_name = 'llama-2-7b-chat.ggmlv3.q4_K_M.bin'
        if mode == 'local':
            bot_response = self.chat_with_api(api_url, user_message, user_instruction, model_name, file_name)
        elif mode == 'runpod':
            prompt = f'[INST] <<SYS>> {user_instruction}<</SYS>> NMAP Data to be analyzed: {user_message} [/INST]'
            bot_response = self.llama_runpod_api(prompt, lkey, lendpoint)
        bot_response = self.chat_with_api(api_url, user_message, user_instruction, model_name, file_name)
        print('test')
        if bot_response:
            return bot_response

    @staticmethod
    def gpt_ai(analyze: str, key: Optional[str]) -> str:
        openai.api_key = key
        prompt = f'\n        Do a DNS analysis on the provided DNS scan information\n        The DNS output must return in a JSON format accorging to the provided\n        output format. The data must be accurate in regards towards a pentest report.\n        The data must follow the following rules:\n        1) The DNS scans must be done from a pentester point of view\n        2) The final output must be minimal according to the format given\n        3) The final output must be kept to a minimal\n\n        The output format:\n        {{\n            "A": [""],\n            "AAA": [""],\n            "NS": [""],\n            "MX": [""],\n            "PTR": [""],\n            "SOA": [""],\n            "TXT": [""]\n        }}\n\n        DNS Data to be analyzed: {analyze}\n        '
        try:
            messages = [{'content': prompt, 'role': 'user'}]
            response = openai.ChatCompletion.create(model=model_engine, messages=messages, max_tokens=1024, n=1, stop=None)
            response = response['choices'][0]['message']['content']
            return dns_ai_data_regex(str(response))
        except KeyboardInterrupt:
            print('Bye')
            quit()

