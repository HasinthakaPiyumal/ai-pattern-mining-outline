# Cluster 39

def wa_skill_processor_tarbot(msg):
    confidence = 0.0
    data = None
    if msg.startswith('tar'):
        return (None, 0.0)
    response, success = call_wa_skill(msg, __self)
    if not success:
        return ({'text': response}, 0.0)
    try:
        intent = response['intents'][0]['intent']
        confidence = response['intents'][0]['confidence']
    except IndexError or KeyError:
        pass
    entities = {}
    for item in response['entities']:
        if item['entity'] in entities:
            entities[item['entity']].append(item['value'])
        else:
            entities[item['entity']] = [item['value']]
    filename = '<archive-file>'
    dirname = '<directory>'
    extensions = {'images': ['.png, .bmp, .jpg'], 'documents': ['.doc', '.docx', '.txt', '.pdf', '.md']}
    flags = ''
    if 'verbose' in entities:
        flags += 'v'
    if 'tar-type' in entities:
        if 'gz' in entities['tar-type']:
            flags += 'z'
        elif 'bz2' in entities['tar-type']:
            flags += 'j'
    intents = {'check-size': {'text': 'Try >> tar -czf - {} | wc -c'.format(filename)}, 'check-validity': {'text': 'Try >> tar tvfW {}'.format(filename)}, 'tar-usage': {'text': 'Tar Usage and Options: \nc - create a archive file. \nx - extract a archive file. \nv - show the progress of archive file. \nf - filename of archive file. \nt - viewing content of archive file.  \nj - filter archive through bzip2.  \nz - filter archive through gzip. \nr - append or update files or directories to existing archive file.  \nW - Verify a archive file.  wildcards - Specify patterns in unix tar command.'}}
    if intent in intents:
        data = intents[intent]
    elif intent == 'tar-directory-to-file':
        flags = 'c{}f'.format(flags)
        data = {'text': 'Try >> tar -{} {} {}'.format(flags, filename, dirname)}
    elif intent == 'untar-file-to-directory':
        flags = 'x{}f'.format(flags)
        if 'directory-name' in entities:
            data = {'text': 'Try >> tar -{} {} -C {}'.format(flags, filename, dirname)}
        else:
            data = {'text': 'Try >> tar -{} {}'.format(flags, filename)}
    elif intent == 'untar-group-of-files':
        flags = 'x{}f'.format(flags)
        ext = []
        if 'file-extension' in entities:
            ext += entities['file-extension']
        elif 'file-type' in entities:
            ext += extensions[entities['file-type'][0]]
        if not ext:
            ext = ['.ext']
        if 'directory-name' in entities:
            data = {'text': 'Try >> tar -{} {} --wildcards {} -C {}'.format(flags, filename, ' '.join(['*' + e for e in ext]), dirname)}
        else:
            data = {'text': 'Try >> tar -{} {} --wildcards {}'.format(flags, filename, ' '.join(['*' + e for e in ext]))}
    elif intent == 'untar-single-file':
        flags = 'x{}f'.format(flags)
        if 'directory-name' in entities:
            data = {'text': 'Try >> tar -{} {} -C {}'.format(flags, filename, dirname)}
        else:
            data = {'text': 'Try >> tar -{} {}'.format(flags, filename)}
    elif intent == 'add-to-file':
        flags = '{}rf'.format(flags)
        if 'tar-type' in entities:
            if 'gz' in entities['tar-type'] or 'bz2' in entities['tar-type']:
                data = {'text': "The tar command don't have a option to add files or directories to an existing compressed tar.gz and tar.bz2 archive file."}
            else:
                data = {'text': 'Unrecognized option.'}
        else:
            data = {'text': 'Try >> tar -{} {} <file>'.format(flags, filename)}
    elif intent == 'list-contents':
        flags = '{}tf'.format(flags)
        ext = []
        if 'file-extension' in entities:
            ext += entities['file-extension']
        elif 'file-type' in entities:
            ext += extensions[entities['file-type'][0]]
        if not ext:
            data = {'text': 'Try >> tar -{} {}'.format(flags, filename)}
        else:
            data = {'text': "Try >> tar -{} {} '{}'".format(flags, filename, ' '.join(['*' + e for e in ext]))}
    else:
        pass
    return (data, confidence)

def call_wa_skill(msg: str, name: str) -> List:
    wa_endpoint = services[name]
    try:
        response = requests.put(wa_endpoint, json={'text': msg}).json()
        if response['result'] == 'success':
            return (response['response']['output'], SUCCESS)
        else:
            return (response['result'], FAILURE)
    except Exception as ex:
        return ('Method failed with status ' + str(ex), FAILURE)

def wa_skill_processor_grepbot(msg):
    confidence = 0.0
    data = {}
    try:
        match_string = re.findall('\\"(.+?)\\"', msg)[0]
        msg = msg.replace('"{}"'.format(match_string), '')
    except:
        match_string = '<string>'
    response, success = call_wa_skill(msg, __self)
    if not success:
        return ({'text': response}, 0.0)
    if msg.startswith('grep for'):
        confidence = 1.0
    else:
        confidence = max([item['confidence'] for item in response['intents']])
    filename = '<filename>'
    dirname = None
    for item in response['entities']:
        if item['entity'] == 'directory':
            dirname = '<directory>'
        if item['entity'] == 'starts-with':
            match_string = '^' + match_string
        if item['entity'] == 'ends-with':
            match_string = match_string + '$'
    flags = '-'
    if dirname:
        flags += 'r'
    for intent in response['intents']:
        if intent['confidence'] > 0.25 and intent['intent'] != 'find':
            flags += __flags[intent['intent']]
    if flags == '-':
        flags = ''
    command = 'grep {} "{}"'.format(flags, match_string)
    if dirname:
        command += ' {}'.format(dirname)
    else:
        command += ' {}'.format(filename)
    data = {'text': 'Try >> ' + command}
    return (data, confidence)

def wa_skill_processor_cloudbot(msg):
    if msg.startswith('ibmcloud'):
        return (None, 0.0)
    response, success = call_wa_skill(msg, __self)
    if not success:
        return ({'text': response}, 0.0)
    try:
        intent = response['intents'][0]['intent']
        confidence = response['intents'][0]['confidence']
    except IndexError or KeyError:
        intent = 'generic'
        confidence = 0.0
    data = {'text': 'Try >> ibmcloud ' + __intents[intent]}
    return (data, confidence)

def wa_skill_processor_zosbot(msg):
    confidence = 0.0
    data = None
    response, success = call_wa_skill(msg, __self)
    if not success:
        return ({'text': response}, 0.0)
    try:
        intent = response['intents'][0]['intent']
        confidence = response['intents'][0]['confidence']
    except IndexError or KeyError:
        pass
    entities = {}
    for item in response['entities']:
        if item['entity'] in entities:
            entities[item['entity']].append(item['value'])
        else:
            entities[item['entity']] = [item['value']]
    if intent == 'bpxmtext':
        data = {'text': 'Try >> bpxmtext [reasoncode]'}
    elif intent == 'compile-c-code':
        data = {'text': 'Try >> xlc'}
    elif intent == 'extattr':
        data = {'text': 'Try >> extattr [+alps] [-alps] [-Fformat] [file] ...'}
    elif intent == 'obrowse':
        data = {'text': 'Try >> obrowse -r xx [file]'}
    elif intent == 'oedit':
        data = {'text': 'Try >> oedit -r xx [file]'}
    elif intent == 'oget':
        data = {'text': 'Try >> OGET [pathname] mvs_data_set_name(member_name)'}
    elif intent == 'oput':
        data = {'text': 'Try >> OPUT mvs_data_set_name(member_name) [pathname]'}
    elif intent == 'oeconsol':
        if 'iplinfo' in entities:
            data = {'text': 'Try >> oeconsol [d iplinfo]'}
        elif 'command' in entities:
            data = {'text': 'Try >> oeconsol [command]'}
        else:
            data = {'text': 'Try >> oeconsol [d parmlib]'}
    elif intent == 'tso':
        data = {'text': 'Try >> tso [-o] [-t] TSO_command'}
    else:
        pass
    return (data, confidence)

