# Cluster 8

def application() -> None:
    try:
        apikey = entry1.get()
        openai.api_key = apikey
        target = entry2.get()
        attack = entry5.get()
        outputf = str(entry4.get())
        match attack:
            case 'geo':
                val = geoip(apikey, target)
                print(val)
                output_save(val, outputf)
            case 'nmap':
                p = int(entry3.get())
                match p:
                    case 1:
                        val = scanner(target, 1, apikey)
                        print(val)
                        output_save(val, outputf)
                    case 2:
                        val = scanner(target, 2, apikey)
                        print(val)
                        output_save(val, outputf)
                    case 3:
                        val = scanner(target, 3, apikey)
                        print(val)
                        output_save(val, outputf)
                    case 4:
                        val = scanner(target, 4, apikey)
                        print(val)
                        output_save(val, outputf)
                    case 5:
                        val = scanner(target, 5, apikey)
                        print(val)
                        output_save(val, outputf)
            case 'dns':
                val = dns_recon(target, apikey)
                output_save(val, outputf)
            case 'subd':
                val = sub(target)
                output_save(val, outputf)
    except KeyboardInterrupt:
        print('Keyboard Interrupt detected ...')

def output_save(output: str) -> None:
    if output == 'Done':
        output_data = 'Status: Successful'
        output_textbox.insert('1.0', output_data)
    else:
        output_textbox.delete('1.0', 'end')
        json_data = json.loads(output)
        formatted_json = json.dumps(json_data, indent=2)
        output_textbox.insert('1.0', formatted_json)

def application() -> None:
    try:
        apikey = entry1.get()
        openai.api_key = apikey
        target = entry2.get()
        attack = entry5.get()
        outputf = str(entry4.get())
        match attack:
            case 'geo':
                val = geoip(apikey, target)
                print(val)
                output_save(val, outputf)
            case 'nmap':
                p = int(entry3.get())
                match p:
                    case 1:
                        val = scanner(target, 1, apikey)
                        print(val)
                        output_save(val, outputf)
                    case 2:
                        val = scanner(target, 2, apikey)
                        print(val)
                        output_save(val, outputf)
                    case 3:
                        val = scanner(target, 3, apikey)
                        print(val)
                        output_save(val, outputf)
                    case 4:
                        val = scanner(target, 4, apikey)
                        print(val)
                        output_save(val, outputf)
                    case 5:
                        val = scanner(target, 5, apikey)
                        print(val)
                        output_save(val, outputf)
            case 'dns':
                val = dns_recon(target, apikey)
                output_save(val, outputf)
            case 'subd':
                val = sub(target)
                output_save(val, outputf)
    except KeyboardInterrupt:
        print('Keyboard Interrupt detected ...')

def geoip(key: Optional[str], target: str) -> Any:
    if key is None:
        raise ValueError('KeyNotFound: Key Not Provided')
    assert key is not None
    if target is None:
        raise ValueError('InvalidTarget: Target Not Provided')
    url = f'https://api.ipgeolocation.io/ipgeo?apiKey={key}&ip={target}'
    response = requests.get(url)
    content = response.text
    return content

def sub(target: str) -> Any:
    s_array = ['www', 'mail', 'ftp', 'localhost', 'webmail', 'smtp', 'hod', 'butterfly', 'ckp', 'tele2', 'receiver', 'reality', 'panopto', 't7', 'thot', 'wien', 'uat-online', 'Footer']
    ss = []
    out = ''
    for subd in s_array:
        try:
            ip_value = dns.resolver.resolve(f'{subd}.{target}', 'A')
            if ip_value:
                ss.append(f'{subd}.{target}')
                if f'{subd}.{target}' in ss:
                    print(f'{subd}.{target} | Found')
                    out += f'{subd}.{target}'
                    out += '\n'
                    out += ''
                else:
                    pass
        except dns.resolver.NXDOMAIN:
            pass
        except dns.resolver.NoAnswer:
            pass
        except KeyboardInterrupt:
            print('Ended')
            quit()
    return out

