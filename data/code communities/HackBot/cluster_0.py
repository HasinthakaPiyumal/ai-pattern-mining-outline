# Cluster 0

def Print_AI_out(prompt, ai_option) -> Panel:
    global chat_history
    if ai_option == 'RUNPOD':
        out = llama_api(prompt)
    else:
        out = llm(prompt)
    ai_out = Markdown(out)
    message_panel = Panel(Align.center(Group('\n', Align.center(ai_out)), vertical='middle'), box=box.ROUNDED, padding=(1, 2), title='[b red]The HackBot AI output', border_style='blue')
    save_data = {'Query': str(prompt), 'AI Answer': str(out)}
    chat_history.append(save_data)
    return message_panel

