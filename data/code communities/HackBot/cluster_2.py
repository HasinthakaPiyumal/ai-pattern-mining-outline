# Cluster 2

def main() -> None:
    clearscr()
    banner = '\n     _   _            _    ____        _   \n    | | | | __ _  ___| | _| __ )  ___ | |_ \n    | |_| |/ _` |/ __| |/ /  _ \\ / _ \\| __| By: Morpheuslord\n    |  _  | (_| | (__|   <| |_) | (_) | |_  AI used: Meta-LLama2\n    |_| |_|\\__,_|\\___|_|\\_\\____/ \\___/ \\__|\n    '
    contact_dev = '\n    Email = morpheuslord@protonmail.com\n    Twitter = https://twitter.com/morpheuslord2\n    LinkedIn https://www.linkedin.com/in/chiranjeevi-g-naidu/\n    Github = https://github.com/morpheuslord\n    '
    help_menu = '\n    - clear_screen: Clears the console screen for better readability.\n    - quit_bot: This is used to quit the chat application\n    - bot_banner: Prints the default bots banner.\n    - contact_dev: Provides my contact information.\n    - save_chat: Saves the current sessions interactions.\n    - help_menu: Lists chatbot commands.\n    - vuln_analysis: Does a Vuln analysis using the scan data or log file.\n    - static_code_analysis: Does a Static code analysis using the scan data or log file.\n    '
    console.print(Panel(Markdown(banner)), style='bold green')
    while True:
        try:
            prompt_in = Prompt.ask('> ')
            if prompt_in == 'quit_bot':
                quit()
            elif prompt_in == 'clear_screen':
                clearscr()
                pass
            elif prompt_in == 'bot_banner':
                console.print(Panel(Markdown(banner)), style='bold green')
                pass
            elif prompt_in == 'save_chat':
                save_chat(chat_history)
                pass
            elif prompt_in == 'static_code_analysis':
                print(Markdown('----------'))
                language_used = Prompt.ask('Language Used> ')
                file_path = Prompt.ask('File Path> ')
                print(Markdown('----------'))
                print(static_analysis(language_used, file_path, AI_OPTION))
                pass
            elif prompt_in == 'vuln_analysis':
                print(Markdown('----------'))
                language_used = Prompt.ask('Scan Type > ')
                file_path = Prompt.ask('File Path > ')
                print(Markdown('----------'))
                print(static_analysis(language_used, file_path, AI_OPTION))
                pass
            elif prompt_in == 'contact_dev':
                console.print(Panel(Align.center(Group(Align.center(Markdown(contact_dev))), vertical='middle'), title='Dev Contact', border_style='red'), style='bold green')
                pass
            elif prompt_in == 'help_menu':
                console.print(Panel(Align.center(Group(Align.center(Markdown(help_menu))), vertical='middle'), title='Help Menu', border_style='red'), style='bold green')
                pass
            else:
                instructions = '\n                You are an helpful cybersecurity assistant and I want you to answer my query and provide output in Markdown: \n                '
                prompt = f'[INST] <<SYS>> {instructions}<</SYS>> Cybersecurity Query: {prompt_in} [/INST]'
                print(Print_AI_out(prompt, AI_OPTION))
                pass
        except KeyboardInterrupt:
            pass

def clearscr() -> None:
    try:
        osp = platform.system()
        match osp:
            case 'Darwin':
                os.system('clear')
            case 'Linux':
                os.system('clear')
            case 'Windows':
                os.system('cls')
    except Exception:
        pass

def save_chat(chat_history: list[Any, Any]) -> None:
    f = open('chat_history.json', 'w+')
    f.write(json.dumps(chat_history))
    f.close

