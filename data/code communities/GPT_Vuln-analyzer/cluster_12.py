# Cluster 12

def PortAI(key: str, data: Any) -> str:
    openai.api_key = key
    try:
        prompt = f'\n        Do a NMAP scan analysis on the provided NMAP scan information\n        The NMAP output must return in a JSON format accorging to the provided\n        output format. The data must be accurate in regards towards a pentest report.\n        The data must follow the following rules:\n        1) The NMAP scans must be done from a pentester point of view\n        2) The final output must be minimal according to the format given.\n        3) The final output must be kept to a minimal.\n        4) If a value not found in the scan just mention an empty string.\n        5) Analyze everything even the smallest of data.\n\n        The output format:\n        {{\n            "critical score": [""],\n            "os information": [""],\n            "open ports": [""],\n            "open services": [""],\n            "vulnerable service": [""],\n            "found cve": [""]\n        }}\n\n        NMAP Data to be analyzed: {data}\n        '
        completion = openai.Completion.create(engine=model_engine, prompt=prompt, max_tokens=1024, n=1, stop=None)
        response = completion.choices[0].text
        return port_extract_data(str(response))
    except KeyboardInterrupt:
        print('Bye')
        quit()

def port_extract_data(json_string: str) -> Any:
    critical_score_pattern = '"critical score": \\["(.*?)"\\]'
    os_information_pattern = '"os information": \\["(.*?)"\\]'
    open_ports_pattern = '"open ports": \\["(.*?)"\\]'
    open_services_pattern = '"open services": \\["(.*?)"\\]'
    vulnerable_service_pattern = '"vulnerable service": \\["(.*?)"\\]'
    found_cve_pattern = '"found cve": \\["(.*?)"\\]'
    critical_score = None
    os_information = None
    open_ports = None
    open_services = None
    vulnerable_service = None
    found_cve = None
    match = re.search(critical_score_pattern, json_string)
    if match:
        critical_score = match.group(1)
    match = re.search(os_information_pattern, json_string)
    if match:
        os_information = match.group(1)
    match = re.search(open_ports_pattern, json_string)
    if match:
        open_ports = match.group(1)
    match = re.search(open_services_pattern, json_string)
    if match:
        open_services = match.group(1)
    match = re.search(vulnerable_service_pattern, json_string)
    if match:
        vulnerable_service = match.group(1)
    match = re.search(found_cve_pattern, json_string)
    if match:
        found_cve = match.group(1)
    data = {'critical score': critical_score, 'os information': os_information, 'open ports': open_ports, 'open services': open_services, 'vulnerable service': vulnerable_service, 'found cve': found_cve}
    json_output = json.dumps(data)
    return json_output

def scanner(ip: Optional[str], profile: int, key: str) -> str:
    if key is not None:
        pass
    else:
        raise ValueError('KeyNotFound: Key Not Provided')
    profile_argument = ''
    if profile == 1:
        profile_argument = '-Pn -sV -T4 -O -F'
    elif profile == 2:
        profile_argument = '-Pn -T4 -A -v'
    elif profile == 3:
        profile_argument = '-Pn -sS -sU -T4 -A -v'
    elif profile == 4:
        profile_argument = '-Pn -p- -T4 -A -v'
    elif profile == 5:
        profile_argument = '-Pn -sS -sU -T4 -A -PE -PP -PS80,443 -PA3389 -PU40125 -PY -g 53 --script=vuln'
    else:
        raise ValueError(f'Invalid Argument: {profile}')
    nm.scan('{}'.format(ip), arguments='{}'.format(profile_argument))
    json_data = nm.analyse_nmap_xml_scan()
    analyze = json_data['scan']
    try:
        response = PortAI(key, analyze)
    except KeyboardInterrupt:
        print('Bye')
        quit()
    return str(response)

def PortAI(key: str, data: Any) -> str:
    openai.api_key = key
    try:
        prompt = f'\n        Do a NMAP scan analysis on the provided NMAP scan information\n        The NMAP output must return in a JSON format accorging to the provided\n        output format. The data must be accurate in regards towards a pentest report.\n        The data must follow the following rules:\n        1) The NMAP scans must be done from a pentester point of view\n        2) The final output must be minimal according to the format given.\n        3) The final output must be kept to a minimal.\n        4) If a value not found in the scan just mention an empty string.\n        5) Analyze everything even the smallest of data.\n\n        The output format:\n        {{\n            "critical score": [""],\n            "os information": [""],\n            "open ports": [""],\n            "open services": [""],\n            "vulnerable service": [""],\n            "found cve": [""]\n        }}\n\n        NMAP Data to be analyzed: {data}\n        '
        completion = openai.Completion.create(engine=model_engine, prompt=prompt, max_tokens=1024, n=1, stop=None)
        response = completion.choices[0].text
        return port_extract_data(str(response))
    except KeyboardInterrupt:
        print('Bye')
        quit()

def scanner(ip: Optional[str], profile: int, key: str) -> str:
    if key is not None:
        pass
    else:
        raise ValueError('KeyNotFound: Key Not Provided')
    profile_argument = ''
    if profile == 1:
        profile_argument = '-Pn -sV -T4 -O -F'
    elif profile == 2:
        profile_argument = '-Pn -T4 -A -v'
    elif profile == 3:
        profile_argument = '-Pn -sS -sU -T4 -A -v'
    elif profile == 4:
        profile_argument = '-Pn -p- -T4 -A -v'
    elif profile == 5:
        profile_argument = '-Pn -sS -sU -T4 -A -PE -PP -PS80,443 -PA3389 -PU40125 -PY -g 53 --script=vuln'
    else:
        raise ValueError(f'Invalid Argument: {profile}')
    nm.scan('{}'.format(ip), arguments='{}'.format(profile_argument))
    json_data = nm.analyse_nmap_xml_scan()
    analyze = json_data['scan']
    try:
        response = PortAI(key, analyze)
    except KeyboardInterrupt:
        print('Bye')
        quit()
    return str(response)

