# Cluster 80

def parse_args():
    parser = argparse.ArgumentParser(description='LinkedIn Candidate CSV Exporter')
    parser.add_argument('--criteria', required=True, help='Search criteria string for LinkedIn candidates')
    parser.add_argument('--max-results', type=int, default=10, help='Maximum number of candidates to find')
    parser.add_argument('--output', default='candidates.csv', help='Output CSV file path')
    return parser.parse_args()

def print_welcome():
    print_banner()
    print(f'\n{BOLD}Welcome to Browser Console Agent{RESET}')
    print('Interact with websites using natural language in your terminal.\n')
    print(f'{SYSTEM_COLOR}You can type a {BOLD}number{RESET}{SYSTEM_COLOR} to select from suggested actions or type your own queries.{RESET}')
    print(f"{SYSTEM_COLOR}Type {BOLD}'exit'{RESET}{SYSTEM_COLOR} or {BOLD}'quit'{RESET}{SYSTEM_COLOR} to end the session.{RESET}\n")

def print_banner():
    banner = ['╔═══════════════════════════════════════════════════════════════╗', '║                                                               ║', '║                     BROWSER CONSOLE AGENT                     ║', '║                                                               ║', '╚═══════════════════════════════════════════════════════════════╝']
    for line in banner:
        print(f'{TITLE_COLOR}{line}{RESET}')

def format_agent_response(response):
    parts = re.split('(?i)possible actions:', response, 1)
    description = parts[0].strip()
    formatted_description = ''
    for paragraph in description.split('\n'):
        if paragraph.strip():
            wrapped = wrap(paragraph, width=80)
            formatted_description += '\n'.join(wrapped) + '\n\n'
    actions_text = ''
    action_items_list = []
    if len(parts) > 1:
        action_text = parts[1].strip()
        actions_text = f'\n{OPTION_COLOR}POSSIBLE ACTIONS:{RESET}\n'
        action_items = re.findall('(?:^|\\n)[•\\-\\d*)\\s]+(.+?)(?=$|\\n[•\\-\\d*)])', action_text, re.MULTILINE)
        if not action_items:
            actions_text += action_text
        else:
            action_items_list = [action.strip() for action in action_items]
            for i, action in enumerate(action_items_list, 1):
                actions_text += f'{OPTION_COLOR}{i}.{RESET} {action}\n'
    return (formatted_description, actions_text, action_items_list)

def update_session_info(response):
    global current_url, visited_urls
    urls = re.findall('https?://[^\\s<>"]+|www\\.[^\\s<>"]+', response)
    if urls:
        new_url = urls[0]
        if new_url != current_url:
            current_url = new_url
            visited_urls.add(current_url)
    return ''

