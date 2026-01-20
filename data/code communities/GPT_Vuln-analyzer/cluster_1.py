# Cluster 1

def handle_attack(attack_type, target, ai, api_keys, additional_params=None):
    additional_params = additional_params or {}
    if attack_type == 'geo':
        output = geo_ip.geoip(api_keys['geoip_api_key'], target)
        asset_codes.print_output(attack_type.capitalize(), str(output), ai)
    elif attack_type == 'nmap':
        output = port_scanner.scanner(ip=target, profile=additional_params.get('profile'), akey=api_keys['openai_api_key'], bkey=api_keys['bard_api_key'], lkey=api_keys['runpod_api_key'], lendpoint=api_keys['runpod_endpoint_id'], AI=ai)
        asset_codes.print_output(attack_type.capitalize(), str(output), ai)
    elif attack_type == 'dns':
        output = dns_enum.dns_resolver(target=target, akey=api_keys['openai_api_key'], bkey=api_keys['bard_api_key'], lkey=api_keys['runpod_api_key'], lendpoint=api_keys['runpod_endpoint_id'], AI=ai)
        asset_codes.print_output(attack_type.capitalize(), str(output), ai)
    elif attack_type == 'sub':
        output = sub_recon.sub_enumerator(target, additional_params.get('list_loc'))
        console.print(output, style='bold underline')
        asset_codes.print_output(attack_type.capitalize(), str(output), ai)
    elif attack_type == 'jwt':
        output = jwt_analyzer.analyze(token=target, openai_api_token=api_keys['openai_api_key'], bard_api_token=api_keys['bard_api_key'], llama_api_token=api_keys['runpod_api_key'], llama_endpoint=api_keys['runpod_endpoint_id'], AI=ai)
        asset_codes.print_output('JWT', output, ai)
    elif attack_type == 'pcap':
        packet_analysis.perform_full_analysis(pcap_path=target, json_path=additional_params.get('output_loc'))
        return 'Done'
    elif attack_type == 'passcracker':
        hash = additional_params.get('password_hash')
        wordlist = additional_params.get('wordlist_file')
        salt = additional_params.get('salt')
        parallel = additional_params.get('parallel')
        complexity = additional_params.get('complexity')
        min_length = additional_params.get('min_length')
        max_length = additional_params.get('max_length')
        character_set = additional_params.get('charecter_set')
        brute_force = additional_params.get('brute_force')
        algorithm = additional_params.get('algorithm')
        Cracker = PasswordCracker(password_hash=hash, wordlist_file=wordlist, algorithm=algorithm, salt=salt, parallel=parallel, complexity_check=complexity)
        if brute_force:
            Cracker.crack_passwords_with_brute_force(min_length, max_length, character_set)
        else:
            Cracker.crack_passwords_with_wordlist()
        Cracker.print_statistics()

def main() -> None:
    asset_codes.run_docker_container()
    args = parse_arguments()
    api_keys = get_api_keys()
    asset_codes.clearscr()
    cowsay.cow('GVA Usage in progress...')
    target = args.target or '127.0.0.1'
    try:
        if args.rich_menu == 'help':
            asset_codes.help_menu()
        elif args.menu is True:
            Menus(lkey='', threads=4, output_loc='', lendpoint='', keyset='', t='', profile_num='', ai_set='', akey_set='', bkey_set='', ai_set_args='', llamakey='', llamaendpoint='', password_hash='', salt='', wordlist_loc='', algorithm='', parallel_proc=True, complexity=True, min_length=1, max_length=6, char_set='abcdefghijklmnopqrstuvwxyz0123456789', bforce=True)
        else:
            additional_params = {'profile': args.profile, 'list_loc': args.sub_list, 'output_loc': args.output, 'password_hash': args.password_hash, 'salt': args.salt, 'parallel': args.parallel, 'complexity': args.complexity, 'brute_force': args.brute_force, 'min_length': args.min_length, 'max_lenght': args.max_length, 'character_set': args.character_set, 'algorithm': args.algorithm, 'wordlist_file': args.wordlist_file}
            handle_attack(args.attack, target, args.ai, api_keys, additional_params)
    except KeyboardInterrupt:
        console.print_exception('Bye')
        quit()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Python-Nmap and chatGPT integrated Vulnerability scanner')
    parser.add_argument('--target', type=str, help='Target IP, hostname, JWT token or pcap file location')
    parser.add_argument('--profile', type=int, default=1, help='Enter Profile of scan 1-13 (Default: 1)')
    parser.add_argument('--attack', type=str, help='Attack type: nmap, dns, sub, jwt, pcap, passcracker')
    parser.add_argument('--sub_list', type=str, default=DEFAULT_LIST_LOC, help='Path to the subdomain list file (txt)')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_LOC, help='Pcap analysis output file')
    parser.add_argument('--rich_menu', type=str, help='Shows a clean help menu using rich')
    parser.add_argument('--menu', action='store_true', default=False, help='Terminal Interactive Menu')
    parser.add_argument('--ai', type=str, default='openai', help='AI options: openai, bard, llama, llama-api')
    parser.add_argument('--password_hash', help='Password hash')
    parser.add_argument('--wordlist_file', help='Wordlist File')
    parser.add_argument('--algorithm', choices=hashlib.algorithms_guaranteed, help='Hash algorithm')
    parser.add_argument('--salt', help='Salt Value')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--complexity', action='store_true', help='Check for password complexity')
    parser.add_argument('--brute_force', action='store_true', help='Perform a brute force attack')
    parser.add_argument('--min_length', type=int, default=1, help='Minimum password length for brute force attack')
    parser.add_argument('--max_length', type=int, default=6, help='Minimum password length for brute force attack')
    parser.add_argument('--character_set', default='abcdefghijklmnopqrstuvwxyz0123456789', help='Character set for brute force attack')
    return parser.parse_args()

def get_api_keys():
    return {'geoip_api_key': os.getenv('GEOIP_API_KEY'), 'openai_api_key': os.getenv('OPENAI_API_KEY'), 'bard_api_key': os.getenv('BARD_API_KEY'), 'runpod_api_key': os.getenv('RUNPOD_API_KEY'), 'runpod_endpoint_id': os.getenv('RUNPOD_ENDPOINT_ID')}

class Menus:

    def nmap_menu(self) -> None:
        try:
            table = Table()
            table.add_column('Options', style='cyan')
            table.add_column('Utility', style='green')
            table.add_row('1', 'AI Options')
            table.add_row('2', 'Set Target')
            table.add_row('3', 'Set Profile')
            table.add_row('4', 'Show options')
            table.add_row('5', 'Run Attack')
            table.add_row('q', 'Quit')
            console.print(table)
            self.option = input('Enter your choice: ')
            match self.option:
                case '1':
                    clearscr()
                    table0 = Table()
                    table0.add_column('Options', style='cyan')
                    table0.add_column('AI Available', style='green')
                    table0.add_row('1', 'OpenAI')
                    table0.add_row('2', 'Bard')
                    table0.add_row('3', 'LLama2')
                    print(Panel(table0))
                    self.ai_set_choice = input('Enter AI of Choice: ')
                    match self.ai_set_choice:
                        case '1':
                            self.ai_set_args, self.ai_set = ('openai', 'openai')
                            self.akey_set = input('Enter OpenAI API: ')
                            print(Panel(f'API-Key Set: {self.akey_set}'))
                        case '2':
                            self.ai_set_args, self.ai_set = ('bard', 'bard')
                            self.bkey_set = input('Enter Bard AI API: ')
                            print(Panel(f'API-Key Set: {self.bkey_set}'))
                        case '3':
                            clearscr()
                            tablel = Table()
                            tablel.add_column('Options', style='cyan')
                            tablel.add_column('Llama Options', style='cyan')
                            tablel.add_row('1', 'Llama Local')
                            tablel.add_row('2', 'Llama RunPod')
                            print(tablel)
                            self.ai_set_choice = input('Enter AI of Choice: ')
                            self.ai_set_args = 'llama'
                            self.ai_set = 'llama'
                            if self.ai_set_choice == '1':
                                self.ai_set = 'llama'
                                print(Panel('No Key needed'))
                                print(Panel('Selected LLama'))
                            elif self.ai_set_choice == '2':
                                self.ai_set = 'llama-api'
                                self.llamaendpoint = input('Enter Runpod Endpoint ID: ')
                                self.llamakey = input('Enter Runpod API Key: ')
                                print(Panel(f'API-Key Set: {self.llamakey}'))
                                print(Panel(f'Runpod Endpoint Set: {self.llamaendpoint}'))
                    self.nmap_menu()
                case '2':
                    clearscr()
                    print(Panel('Set Target Hostname or IP'))
                    self.t = input('Enter Target: ')
                    print(Panel(f'Target Set: {self.t}'))
                    self.nmap_menu()
                case '3':
                    clearscr()
                    table1 = Table()
                    table1.add_column('Options', style='cyan')
                    table1.add_column('Value', style='green')
                    table1.add_row('1', '-Pn -sV -T4 -O -F')
                    table1.add_row('2', '-Pn -T4 -A -v')
                    table1.add_row('3', '-Pn -sS -sU -T4 -A -v')
                    table1.add_row('4', '-Pn -p- -T4 -A -v')
                    table1.add_row('5', '-Pn -sS -sU -T4 -A -PE -PP  -PY -g 53 --script=vuln')
                    table1.add_row('6', '-Pn -sV -p- -A')
                    table1.add_row('7', '-Pn -sS -sV -O -T4 -A')
                    table1.add_row('8', '-Pn -sC')
                    table1.add_row('9', '-Pn -p 1-65535 -T4 -A -v')
                    table1.add_row('10', '-Pn -sU -T4')
                    table1.add_row('11', '-Pn -sV --top-ports 100')
                    table1.add_row('12', '-Pn -sS -sV -T4 --script=default,discovery,vuln')
                    table1.add_row('13', '-Pn -F')
                    print(Panel(table1))
                    self.profile_num = input('Enter your Profile: ')
                    print(Panel(f'Profile Set {self.profile_num}'))
                    self.nmap_menu()
                case '4':
                    clearscr()
                    table2 = Table()
                    table2.add_column('Options', style='cyan')
                    table2.add_column('Value', style='green')
                    table2.add_row('AI Set', str(self.ai_set_args))
                    table2.add_row('OpenAI API Key', str(self.akey_set))
                    table2.add_row('Bard AI API Key', str(self.bkey_set))
                    table2.add_row('Llama Runpod API Key', str(self.llamakey))
                    table2.add_row('Runpod Endpoint ID', str(self.llamaendpoint))
                    table2.add_row('Target', str(self.t))
                    table2.add_row('Profile', str(self.profile_num))
                    print(Panel(table2))
                    self.nmap_menu()
                case '5':
                    clearscr()
                    pout: str = port_scanner.scanner(AIModels=p_ai_models, ip=self.t, profile=int(self.profile_num), akey=self.akey_set, bkey=self.bkey_set, lkey=self.lkey, lendpoint=self.lendpoint, AI=self.ai_set)
                    assets.print_output('Nmap', pout, self.ai_set)
                case 'q':
                    quit()
        except KeyboardInterrupt:
            print(Panel('Exiting Program'))

    def dns_menu(self) -> None:
        try:
            table = Table()
            table.add_column('Options', style='cyan')
            table.add_column('Utility', style='green')
            table.add_row('1', 'AI Option')
            table.add_row('2', 'Set Target')
            table.add_row('3', 'Show options')
            table.add_row('4', 'Run Attack')
            table.add_row('q', 'Quit')
            console.print(table)
            option = input('Enter your choice: ')
            match option:
                case '1':
                    clearscr()
                    table0 = Table()
                    table0.add_column('Options', style='cyan')
                    table0.add_column('AI Available', style='green')
                    table0.add_row('1', 'OpenAI')
                    table0.add_row('2', 'Bard')
                    table0.add_row('3', 'LLama2')
                    print(Panel(table0))
                    self.ai_set_choice = input('Enter AI of Choice: ')
                    match self.ai_set_choice:
                        case '1':
                            self.ai_set_args, self.ai_set = ('openai', 'openai')
                            self.akey_set = input('Enter OpenAI API: ')
                            print(Panel(f'API-Key Set: {self.akey_set}'))
                        case '2':
                            self.ai_set_args, self.ai_set = ('bard', 'bard')
                            self.bkey_set = input('Enter Bard AI API: ')
                            print(Panel(f'API-Key Set: {self.bkey_set}'))
                        case '3':
                            clearscr()
                            tablel = Table()
                            tablel.add_column('Options', style='cyan')
                            tablel.add_column('Llama Options', style='cyan')
                            tablel.add_row('1', 'Llama Local')
                            tablel.add_row('2', 'Llama RunPod')
                            print(tablel)
                            self.ai_set_choice = input('Enter AI of Choice: ')
                            self.ai_set_args = 'llama'
                            self.ai_set = 'llama'
                            if self.ai_set_choice == '1':
                                self.ai_set = 'llama'
                                print(Panel('No Key needed'))
                                print(Panel('Selected LLama'))
                            elif self.ai_set_choice == '2':
                                self.ai_set = 'llama-api'
                                self.llamaendpoint = input('Enter Runpod Endpoint ID: ')
                                self.llamakey = input('Enter Runpod API Key: ')
                                print(Panel(f'API-Key Set: {self.llamakey}'))
                                print(Panel(f'Runpod Endpoint Set: {self.llamaendpoint}'))
                    self.dns_menu()
                case '2':
                    clearscr()
                    print(Panel('Set Target Hostname or IP'))
                    self.t = input('Enter Target: ')
                    print(Panel(f'Target Set:{self.t}'))
                    self.dns_menu()
                case '3':
                    clearscr()
                    table1 = Table()
                    table1.add_column('Options', style='cyan')
                    table1.add_column('Value', style='green')
                    table1.add_row('AI Set', str(self.ai_set_args))
                    table1.add_row('OpenAI API Key', str(self.akey_set))
                    table1.add_row('Bard AI API Key', str(self.bkey_set))
                    table1.add_row('Llama Runpod API Key', str(self.llamakey))
                    table1.add_row('Runpod Endpoint ID', str(self.llamaendpoint))
                    table1.add_row('Target', str(self.t))
                    print(Panel(table1))
                    self.dns_menu()
                case '4':
                    clearscr()
                    dns_output: str = dns_enum.dns_resolver(AIModels=dns_ai_models, target=self.t, akey=self.akey_set, bkey=self.bkey_set, lkey=self.lkey, lendpoint=self.lendpoint, AI=self.ai_set)
                    assets.print_output('DNS', dns_output, self.ai_set)
                case 'q':
                    quit()
        except KeyboardInterrupt:
            print(Panel('Exiting Program'))

    def jwt_menu(self) -> None:
        try:
            table = Table()
            table.add_column('Options', style='cyan')
            table.add_column('Utility', style='green')
            table.add_row('1', 'AI Option')
            table.add_row('2', 'Set Token')
            table.add_row('3', 'Show options')
            table.add_row('4', 'Run Attack')
            table.add_row('q', 'Quit')
            console.print(table)
            option = input('Enter your choice: ')
            match option:
                case '1':
                    clearscr()
                    table0 = Table()
                    table0.add_column('Options', style='cyan')
                    table0.add_column('AI Available', style='green')
                    table0.add_row('1', 'OpenAI')
                    table0.add_row('2', 'Bard')
                    table0.add_row('3', 'LLama2')
                    print(Panel(table0))
                    self.ai_set_choice = input('Enter AI of Choice: ')
                    match self.ai_set_choice:
                        case '1':
                            self.ai_set_args, self.ai_set = ('openai', 'openai')
                            self.akey_set = input('Enter OpenAI API: ')
                            print(Panel(f'API-Key Set: {self.akey_set}'))
                        case '2':
                            self.ai_set_args, self.ai_set = ('bard', 'bard')
                            self.bkey_set = input('Enter Bard AI API: ')
                            print(Panel(f'API-Key Set: {self.bkey_set}'))
                        case '3':
                            clearscr()
                            tablel = Table()
                            tablel.add_column('Options', style='cyan')
                            tablel.add_column('Llama Options', style='cyan')
                            tablel.add_row('1', 'Llama Local')
                            tablel.add_row('2', 'Llama RunPod')
                            print(tablel)
                            self.ai_set_choice = input('Enter AI of Choice: ')
                            self.ai_set_args = 'llama'
                            self.ai_set = 'llama'
                            if self.ai_set_choice == '1':
                                self.ai_set = 'llama'
                                print(Panel('No Key needed'))
                                print(Panel('Selected LLama'))
                            elif self.ai_set_choice == '2':
                                self.ai_set = 'llama-api'
                                self.llamaendpoint = input('Enter Runpod Endpoint ID: ')
                                self.llamakey = input('Enter Runpod API Key: ')
                                print(Panel(f'API-Key Set: {self.llamakey}'))
                                print(Panel(f'Runpod Endpoint Set: {self.llamaendpoint}'))
                    self.jwt_menu()
                case '2':
                    clearscr()
                    print(Panel('Set Token value'))
                    self.t = input('Enter TOKEN: ')
                    print(Panel(f'Token Set:{self.t}'))
                    self.jwt_menu()
                case '3':
                    clearscr()
                    table1 = Table()
                    table1.add_column('Options', style='cyan')
                    table1.add_column('Value', style='green')
                    table1.add_row('AI Set', str(self.ai_set_args))
                    table1.add_row('OpenAI API Key', str(self.akey_set))
                    table1.add_row('Bard AI API Key', str(self.bkey_set))
                    table1.add_row('Llama Runpod API Key', str(self.llamakey))
                    table1.add_row('Runpod Endpoint ID', str(self.llamaendpoint))
                    table1.add_row('JWT TOKEN', str(self.t))
                    print(Panel(table1))
                    self.jwt_menu()
                case '4':
                    clearscr()
                    JWT_output: str = jwt_analyzer.analyze(AIModels=jwt_ai_model, token=self.t, openai_api_token=self.akey_set, bard_api_token=self.bkey_set, llama_api_token=self.lkey, llama_endpoint=self.lendpoint, AI=self.ai_set)
                    assets.print_output('JWT', JWT_output, self.ai_set)
                case 'q':
                    quit()
        except KeyboardInterrupt:
            print(Panel('Exiting Program'))

    def pcap_menu(self) -> None:
        try:
            table = Table()
            table.add_column('Options', style='cyan')
            table.add_column('Utility', style='green')
            table.add_row('1', 'Set Target file location')
            table.add_row('2', 'Set Output file location')
            table.add_row('3', 'Show options')
            table.add_row('4', 'Run Attack')
            table.add_row('q', 'Quit')
            console.print(table)
            self.option = input('Enter your choice: ')
            match self.option:
                case '1':
                    clearscr()
                    print(Panel('Set Target PCAP file Location'))
                    self.t = input('Enter Target: ')
                    print(Panel(f'Target Set: {self.t}'))
                    self.pcap_menu()
                case '2':
                    clearscr()
                    print(Panel('Set Output file Location'))
                    self.t = input('Enter Location: ')
                    print(Panel(f'Output Set: {self.output_loc}'))
                    self.pcap_menu()
                case '3':
                    clearscr()
                    print(Panel('Set Number of threads'))
                    self.t = input('Enter Threads: ')
                    print(Panel(f'Threads Set: {self.threads}'))
                    self.pcap_menu()
                case '4':
                    clearscr()
                    packetanalysis.perform_full_analysis(pcap_path=self.t, json_path=self.output_loc)
                case 'q':
                    quit()
        except KeyboardInterrupt:
            print(Panel('Exiting Program'))

    def geo_menu(self) -> None:
        try:
            table = Table()
            table.add_column('Options', style='cyan')
            table.add_column('Utility', style='green')
            table.add_row('1', 'ADD API Key')
            table.add_row('2', 'Set Target')
            table.add_row('3', 'Show options')
            table.add_row('4', 'Run Attack')
            table.add_row('q', 'Quit')
            console.print(table)
            self.option = input('Enter your choice: ')
            match self.option:
                case '1':
                    clearscr()
                    self.keyset = input('Enter GEO-IP API: ')
                    print(Panel(f'GEOIP API Key Set: {self.keyset}'))
                    self.geo_menu()
                case '2':
                    clearscr()
                    print(Panel('Set Target Hostname or IP'))
                    self.t = input('Enter Target: ')
                    print(Panel(f'Target Set: {self.t}'))
                    self.geo_menu()
                case '3':
                    clearscr()
                    table1 = Table()
                    table1.add_column('Options', style='cyan')
                    table1.add_column('Value', style='green')
                    table1.add_row('API Key', str(self.keyset))
                    table1.add_row('Target', str(self.t))
                    print(Panel(table1))
                    self.geo_menu()
                case '4':
                    clearscr()
                    geo_output: str = geo_ip.geoip(self.keyset, self.t)
                    assets.print_output('GeoIP', str(geo_output), ai='None')
                case 'q':
                    quit()
        except KeyboardInterrupt:
            print(Panel('Exiting Program'))

    def sub_menu(self) -> None:
        try:
            table = Table()
            table.add_column('Options', style='cyan')
            table.add_column('Utility', style='green')
            table.add_row('1', 'ADD Subdomain list')
            table.add_row('2', 'Set Target')
            table.add_row('3', 'Show options')
            table.add_row('4', 'Run Attack')
            table.add_row('q', 'Quit')
            console.print(table)
            self.option = input('Enter your choice: ')
            match self.option:
                case '1':
                    clearscr()
                    print(Panel('Set TXT subdomain file location'))
                    self.list_loc = input('Enter List Location:  ')
                    print(Panel(f'Location Set: {self.list_loc}'))
                    self.sub_menu()
                case '2':
                    clearscr()
                    print(Panel('Set Target Hostname or IP'))
                    self.t = input('Enter Target: ')
                    print(Panel(f'Target Set: {self.t}'))
                    self.sub_menu()
                case '3':
                    clearscr()
                    table1 = Table()
                    table1.add_column('Options', style='cyan')
                    table1.add_column('Value', style='green')
                    table1.add_row('Location', str(self.list_loc))
                    table1.add_row('Target', str(self.t))
                    print(Panel(table1))
                    self.sub_menu()
                case '4':
                    clearscr()
                    sub_output: str = sub_recon.sub_enumerator(self.t, self.list_loc)
                    console.print(sub_output, style='bold underline')
                case 'q':
                    quit()
        except KeyboardInterrupt:
            print(Panel('Exiting Program'))

    def str_to_bool(self, input_str):
        return input_str.lower() in ('true', '1', 't', 'y', 'yes')

    def hash_menu(self) -> None:
        """
            Password Hash: str
            Salt: str
            Wordlist File: str:loc
            Algorithm: str
            Parallel Processing: bool: True
            Complexity: bool: True
            Min Length: int:1
            Max Length: int:6
            Charecter Set: str:abcdefghijklmnopqrstuvwxyz0123456789
            Bruteforce: bool:True
            Attack
            password_hash, salt, wordlist_loc, algorithm, parallel_proc, complexity, min_length, max_length, char_set, bforce
        """
        try:
            self.char_set = 'abcdefghijklmnopqrstuvwxyz0123456789'
            self.min_length = 1
            self.max_length = 6
            table = Table()
            table.add_column('Options', style='cyan')
            table.add_column('Utility', style='green')
            table.add_row('1', 'Set Password Hash')
            table.add_row('2', 'Set Salt')
            table.add_row('3', 'Set Algorithm')
            table.add_row('4', 'Set Wordlist Loc')
            table.add_row('5', 'Set Parallel Proc')
            table.add_row('6', 'Set Complexity')
            table.add_row('7', 'Set Min Gen Length')
            table.add_row('8', 'Set Max Gen Length')
            table.add_row('9', 'Set Charecter Set')
            table.add_row('10', 'Set Attack Type')
            table.add_row('11', 'Show Options')
            table.add_row('12', 'Run Attack')
            table.add_row('q', 'Quit')
            console.print(table)
            self.option = input('Enter your choice: ')
            match self.option:
                case '1':
                    clearscr()
                    print(Panel('Set Password Hash Value'))
                    self.password_hash = input('Enter Hash Value:  ')
                    print(Panel(f'Hash Set: {self.password_hash}'))
                    self.hash_menu()
                case '2':
                    clearscr()
                    print(Panel('Set Salt Value'))
                    self.salt = input('Enter Salt Value:  ')
                    print(Panel(f'Salt Set: {self.salt}'))
                    self.hash_menu()
                case '3':
                    clearscr()
                    print(Panel('\n                                Set Algorithm Value\n                                Select From: sha256,shake_128,sha3_224,sha1,sha224,sha512,blake2s,blake2b,md5,sha384,sha3_384,sha3_256,shake_256,sha3_512\n                                '))
                    self.algorithm = input('Enter Algorithm Value:  ')
                    print(Panel(f'Algorithm Set: {self.algorithm}'))
                    self.hash_menu()
                case '4':
                    clearscr()
                    print(Panel('Set Wordlist location'))
                    self.wordlist_loc = input('Enter Wordlist location:  ')
                    print(Panel(f'Wordlist Location Set: {self.wordlist_loc}'))
                    self.hash_menu()
                case '5':
                    clearscr()
                    print(Panel(f'Set Parallel Processing: Default value = {self.parallel_proc}'))
                    self.parallel_proc = self.str_to_bool(input('Enter True/False:  '))
                    print(Panel(f'Proccessing Option Set: {self.parallel_proc}'))
                    self.hash_menu()
                case '6':
                    clearscr()
                    print(Panel(f'Set Complexity: Default value = {self.complexity}'))
                    self.complexity = self.str_to_bool(input('Enter True/False:  '))
                    print(Panel(f'Complexity Set: {self.complexity}'))
                    self.hash_menu()
                case '7':
                    clearscr()
                    print(Panel(f'Set Min Password Gen value: Default value = {self.min_length}'))
                    self.min_length = input('Enter Number:  ')
                    print(Panel(f'Min value Set: {self.min_length}'))
                    self.hash_menu()
                case '8':
                    clearscr()
                    print(Panel(f'Set Max Password Gen value: Default value = {self.max_length}'))
                    self.max_length = input('Enter Number:  ')
                    print(Panel(f'Max Value Set: {self.max_length}'))
                    self.hash_menu()
                case '9':
                    clearscr()
                    print(Panel(f'Set Charecter Set value: Default value = {self.char_set}'))
                    self.max_length = input('Enter Number:  ')
                    print(Panel(f'Charecter Set: {self.max_length}'))
                    self.hash_menu()
                case '10':
                    clearscr()
                    print(Panel(f'Set Attack Type: Default value = {self.bforce}'))
                    self.bforce = self.str_to_bool(input('Enter True/False:  '))
                    print(Panel(f'Attack Type Set: {self.bforce}'))
                    self.hash_menu()
                case '11':
                    clearscr()
                    clearscr()
                    table1 = Table()
                    table1.add_column('Options', style='cyan')
                    table1.add_column('Value', style='green')
                    table1.add_row('Password hash', str(self.password_hash))
                    table1.add_row('Salt', str(self.salt))
                    table1.add_row('Algorithm', str(self.algorithm))
                    table1.add_row('Wordlist Loc', str(self.wordlist_loc))
                    table1.add_row('Parallel Proc', str(self.parallel_proc))
                    table1.add_row('Complexity', str(self.complexity))
                    table1.add_row('Min Gen Length', str(self.min_length))
                    table1.add_row('Max Gen Length', str(self.max_length))
                    table1.add_row('Charecter Set', str(self.char_set))
                    table1.add_row('Attack Type', str(self.bforce))
                    print(Panel(table1))
                    self.hash_menu()
                case '12':
                    clearscr()
                    self.parallel_proc = True if self.parallel_proc is None else self.parallel_proc
                    self.complexity = True if self.complexity is None else self.complexity
                    self.bforce = True if self.bforce is None else self.bforce
                    print(self.parallel_proc)
                    print(self.complexity)
                    print(self.bforce)
                    passcracker = PasswordCracker(password_hash=self.password_hash, wordlist_file=self.wordlist_loc, algorithm=self.algorithm, salt=self.salt, parallel=self.parallel_proc, complexity_check=self.complexity)
                    if self.bforce is True:
                        passcracker.crack_passwords_with_brute_force(self.min_length, self.max_length, self.char_set)
                    elif self.bforce is False:
                        passcracker.crack_passwords_with_wordlist()
                    passcracker.print_statistics()
                case 'q':
                    quit()
        except KeyboardInterrupt:
            print(Panel('Exiting Program'))

    def __init__(self, lkey, threads, output_loc, lendpoint, keyset, t, profile_num, ai_set, akey_set, bkey_set, ai_set_args, llamakey, llamaendpoint, password_hash, salt, wordlist_loc, algorithm, parallel_proc, complexity, min_length, max_length, char_set, bforce) -> None:
        try:
            self.lkey = lkey
            self.threads = threads
            self.output_loc = output_loc
            self.lendpoint = lendpoint
            self.keyset = keyset
            self.t = t
            self.profile_num = profile_num
            self.ai_set = ai_set
            self.akey_set = akey_set
            self.bkey_set = bkey_set
            self.ai_set_args = ai_set_args
            self.llamakey = llamakey
            self.llamaendpoint = llamaendpoint
            self.password_hash = password_hash
            self.salt = salt
            self.wordlist_loc = wordlist_loc
            self.algorithm = algorithm
            self.parallel_proc = parallel_proc
            self.complexity = complexity
            self.min_length = min_length
            self.max_length = max_length
            self.char_set = char_set
            self.bforce = bforce
            table = Table()
            table.add_column('Options', style='cyan')
            table.add_column('Utility', style='green')
            table.add_row('1', 'Nmap Enum')
            table.add_row('2', 'DNS Enum')
            table.add_row('3', 'Subdomain Enum')
            table.add_row('4', 'GEO-IP Enum')
            table.add_row('5', 'JWT Analysis')
            table.add_row('6', 'PCAP Analysis')
            table.add_row('7', 'Hash Cracker')
            table.add_row('q', 'Quit')
            console.print(table)
            option = input('Enter your choice: ')
            match option:
                case '1':
                    clearscr()
                    self.nmap_menu()
                case '2':
                    clearscr()
                    self.dns_menu()
                case '3':
                    clearscr()
                    self.sub_menu()
                case '4':
                    clearscr()
                    self.geo_menu()
                case '5':
                    clearscr()
                    self.jwt_menu()
                case '6':
                    clearscr()
                    self.pcap_menu()
                case '7':
                    clearscr()
                    self.hash_menu()
                case 'q':
                    quit()
        except KeyboardInterrupt:
            print(Panel('Exiting Program'))

