# Cluster 3

def application(attack, entry2, entry3, entry_ai, entry5):
    try:
        target = entry2.get()
        profile = entry3.get() if entry3 else None
        save_loc = entry5.get() if entry5 else None
        ai_choices = entry_ai.get() if entry_ai else None
        if attack == 'geo':
            geo_output: str = geo_ip_recon.geoip(gkey, target)
            output_save(str(geo_output))
        elif attack == 'nmap':
            p1_out = port_scanner.scanner(ip=target, profile=int(profile) if profile else None, akey=akey, bkey=bkey, lkey=lkey, lendpoint=lendpoint, AI=ai_choices)
            output_save(p1_out)
        elif attack == 'dns':
            dns_output: str = dns_enum.dns_resolver(target=target, akey=akey, bkey=bkey, lkey=lkey, lendpoint=lendpoint, AI=ai_choices)
            output_save(dns_output)
        elif attack == 'sub':
            sub_output: str = sub_recon.sub_enumerator(target, list_loc)
            output_save(sub_output)
        elif attack == 'jwt':
            output: str = jwt_analyzer.analyze(token=target, openai_api_token=akey, bard_api_token=bkey, llama_api_token=lkey, llama_endpoint=lendpoint, AI=ai_choices)
            output_save(output)
        elif attack == 'pcap':
            packet_analysis.perform_full_analysis(pcap_path=target, json_path=save_loc)
            output_save('Done')
    except KeyboardInterrupt:
        print('Keyboard Interrupt detected ...')

def select_frame_by_name(name):
    global frame
    frame.destroy()
    frame = customtkinter.CTkFrame(master=input_frame)
    frame.pack(pady=20, padx=20, fill='both', expand=True)
    label_text = f'GVA System - {name.capitalize()}'
    label = customtkinter.CTkLabel(master=frame, text=label_text)
    label.pack(pady=12, padx=10)
    entry2 = customtkinter.CTkEntry(master=frame, placeholder_text='Target/capfile/token')
    entry2.pack(pady=12, padx=10)
    if name in ['nmap', 'dns', 'jwt']:
        ai_choices_val = ['openai', 'bard', 'llama-api']
        entry_ai = customtkinter.CTkComboBox(master=frame, values=ai_choices_val, state='readonly')
        entry_ai.set('Select AI Input')
        entry_ai.pack(pady=12, padx=10)
    else:
        entry_ai = None
    entry3 = None
    entry5 = None
    if name == 'nmap':
        entry3 = customtkinter.CTkEntry(master=frame, placeholder_text='Profile')
        entry3.pack(pady=12, padx=10)
    elif name == 'sub':
        entry3 = customtkinter.CTkEntry(master=frame, placeholder_text='File Location')
        entry3.pack(pady=12, padx=10)
    elif name == 'pcap':
        entry5 = customtkinter.CTkEntry(master=frame, placeholder_text='Save Location')
        entry5.pack(pady=12, padx=10)
    button = customtkinter.CTkButton(master=frame, text='Run', command=lambda: application(attack=name, entry2=entry2, entry3=entry3, entry_ai=entry_ai, entry5=entry5))
    button.pack(pady=12, padx=10)

