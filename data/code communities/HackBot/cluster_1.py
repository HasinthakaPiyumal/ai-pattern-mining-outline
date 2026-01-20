# Cluster 1

def llama_api(prompt):
    payload = json.dumps({'input': {'prompt': prompt, 'max_new_tokens': 4500, 'temperature': 0.9, 'top_k': 50, 'top_p': 0.7, 'repetition_penalty': 1.2, 'batch_size': 8, 'stop': ['</s>']}})
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {RUNPOD_API_KEY}'}
    response = requests.request('POST', url, headers=headers, data=payload)
    response_t = json.loads(response.text)
    return response_t['output']

def vuln_analysis(scan_type, file_path, ai_option) -> Panel:
    global chat_history
    f = open(file_path, 'r')
    file_data = f.read()
    f.close
    instructions = '\n    You are a Universal Vulnerability Analyzer powered by the Llama2 model. Your main objective is to analyze any provided scan data or log data to identify potential vulnerabilities in the target system or network. You can use the scan type or the scanner type to prepare better report.\n        1. Data Analysis: Thoroughly analyze the given scan data or log data to uncover vulnerabilities and security issues in the target environment.\n        2. Format Flexibility: Be adaptable to handle various data formats, such as NMAP scans, vulnerability assessment reports, security logs, or any other relevant data.\n        3. Vulnerability Identification: Identify different types of vulnerabilities, including but not limited to software vulnerabilities, misconfigurations, exposed sensitive information, potential security risks, and more.\n        4. Accuracy and Precision: Ensure the analysis results are accurate and precise to provide reliable information for further actions.\n        5. Comprehensive Report: Generate a detailed vulnerability report that includes the following sections:\n            - Vulnerability Summary: A brief overview of the detected vulnerabilities.\n            - Software Vulnerabilities: List of identified software vulnerabilities with their respective severity levels.\n            - Misconfigurations: Highlight any misconfigurations found during the analysis.\n            - Exposed Sensitive Information: Identify any exposed sensitive data, such as passwords, API keys, or usernames.\n            - Security Risks: Flag potential security risks and their implications.\n            - Recommendations: Provide actionable recommendations to mitigate the detected vulnerabilities.\n        6. Threat Severity: Prioritize vulnerabilities based on their severity level to help users focus on critical issues first.\n        7. Context Awareness: Consider the context of the target system or network when analyzing vulnerabilities. Take into account factors like system architecture, user permissions, and network topology.\n        8. Handling Unsupported Data: If the provided data format is unsupported or unclear, politely ask for clarifications or indicate the limitations.\n        9. Language and Style: Use clear and concise language to present the analysis results. Avoid jargon and unnecessary technicalities.\n        10. Provide output in Markdown. \n    '
    data = f'\n        Provide the scan type: {scan_type} \n        Provide the scan data or log data that needs to be analyzed: {file_data}\n    '
    prompt = f'[INST] <<SYS>> {instructions}<</SYS>> Data to be analyzed: {data} [/INST]'
    if ai_option == 'RUNPOD':
        out = llama_api(prompt)
    else:
        out = llm(prompt)
    ai_out = Markdown(out)
    message_panel = Panel(Align.center(Group('\n', Align.center(ai_out)), vertical='middle'), box=box.ROUNDED, padding=(1, 2), title='[b red]The HackBot AI output', border_style='blue')
    save_data = {'Query': str(prompt), 'AI Answer': str(out)}
    chat_history.append(save_data)
    return message_panel

