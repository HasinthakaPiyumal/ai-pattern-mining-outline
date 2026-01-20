# Cluster 2

def StartServices():
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if process.info['cmdline'] and 'ailice.modules' in ' '.join(process.info['cmdline']):
                print(f'killing proc with PID {process.info['pid']}')
                process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    for serviceName, cfg in config.services.items():
        if 'speech' == serviceName and (not config.speechOn):
            continue
        if 'cmd' not in cfg or '' == cfg['cmd'].strip():
            print(f"{serviceName}'s cmd is not configured and will attempt to connect {cfg['addr']} directly.")
            continue
        p = subprocess.Popen(cfg['cmd'], shell=True, cwd=None, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
        print(serviceName, ' started.')
    signal.signal(signal.SIGINT, TerminateSubprocess)
    signal.signal(signal.SIGTERM, TerminateSubprocess)

def main():
    config.Initialize(configFile=os.path.join(appdirs.user_config_dir('ailice', 'Steven Lu'), 'config.json'))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelID', type=str, default=config.modelID, help='modelID specifies the model. There are two modes for model configuration. In the first mode, the model is uniformly specified by modelID. In the second mode, different types of agents will run on different models. When this parameter is an empty string (unspecified), the second mode will be used automatically, i.e., the models configured individually for different agents under the agentModelConfig field in config.json will be used. The currently supported models can be seen in config.json. Default: %(default)s')
    parser.add_argument('--quantization', type=str, default=config.quantization, help='quantization is the quantization option, you can choose 4bit or 8bit. Default: %(default)s')
    parser.add_argument('--maxMemory', type=dict, default=config.maxMemory, help='maxMemory is the memory video memory capacity constraint, the format when set is like "{0:"23GiB", 1:"24GiB", "cpu": "64GiB"}". Default: %(default)s')
    parser.add_argument('--prompt', type=str, default=config.prompt, help='prompt specifies the prompt to be executed, which is the type of agent. Default: %(default)s')
    parser.add_argument('--temperature', type=float, default=config.temperature, help='temperature sets the temperature parameter of LLM reasoning. Default: %(default)s')
    parser.add_argument('--flashAttention2', type=bool, default=config.flashAttention2, help='flashAttention2 is the switch to enable flash attention 2 to speed up inference. It may have a certain impact on output quality. Default: %(default)s')
    parser.add_argument('--contextWindowRatio', type=float, default=config.contextWindowRatio, help='contextWindowRatio is a user-specified proportion coefficient, which determines the proportion of the upper limit of the prompt length constructed during inference to the LLM context window in some cases. Default: %(default)s')
    parser.add_argument('--speechOn', type=bool, default=config.speechOn, help='speechOn is the switch to enable voice conversation. Please note that the voice dialogue is currently not smooth yet. Default: %(default)s')
    parser.add_argument('--ttsDevice', type=str, default=config.ttsDevice, help='ttsDevice specifies the computing device used by the text-to-speech model. You can set it to "cuda" if there is enough video memory. Default: %(default)s')
    parser.add_argument('--sttDevice', type=str, default=config.sttDevice, help='sttDevice specifies the computing device used by the speech-to-text model. You can set it to "cuda" if there is enough video memory. Default: %(default)s')
    parser.add_argument('--resetApiKey', action='store_true', help="Whether to reset the model's API key after startup.")
    parser.add_argument('--chatHistoryPath', type=str, default=config.chatHistoryPath, help='chatHistoryPath is used to specify the chat history storage path. Default: %(default)s')
    parser.add_argument('--session', type=str, default='', help='session is used to specify the session storage path, if the directory is not empty, the conversation history stored in that directory will be loaded and updated. Default: %(default)s')
    kwargs = vars(parser.parse_args())
    config.Check4Update(kwargs['modelID'], kwargs['resetApiKey'])
    config.Update(kwargs)
    try:
        mainLoop(session=kwargs['session'])
    except Exception as e:
        print(f'Encountered an exception, AIlice is exiting: {str(e)}')
        print(e.tb) if hasattr(e, 'tb') else traceback.print_tb(e.__traceback__)
        TerminateSubprocess()
        raise

def TerminateSubprocess(signum=None, frame=None):
    for p in processes:
        if p.poll() is None:
            p.terminate()
            p.wait()
    sys.exit(0)

def Init():
    StartServices()
    logger.info('Services started')
    context.Create()
    InitServer()
    return

def InitServer():
    os.makedirs(f'{config.chatHistoryPath}/logs', exist_ok=True)
    handler = RotatingFileHandler(f'{config.chatHistoryPath}/logs/app.log', maxBytes=1000000, backupCount=5)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)
    logger.info(f'Flask app logger configured to write to {config.chatHistoryPath}/logs/app.log')
    return

def main():
    config.Initialize(configFile=AILICE_CONFIG)
    logger.info('Configuration initialized')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelID', type=str, default=config.modelID, help='modelID specifies the model. There are two modes for model configuration. In the first mode, the model is uniformly specified by modelID. In the second mode, different types of agents will run on different models. When this parameter is an empty string (unspecified), the second mode will be used automatically, i.e., the models configured individually for different agents under the agentModelConfig field in config.json will be used. The currently supported models can be seen in config.json. Default: %(default)s')
    parser.add_argument('--quantization', type=str, default=config.quantization, help='quantization is the quantization option, you can choose 4bit or 8bit. Default: %(default)s')
    parser.add_argument('--maxMemory', type=dict, default=config.maxMemory, help='maxMemory is the memory video memory capacity constraint, the format when set is like "{0:"23GiB", 1:"24GiB", "cpu": "64GiB"}". Default: %(default)s')
    parser.add_argument('--prompt', type=str, default=config.prompt, help='prompt specifies the prompt to be executed, which is the type of agent. Default: %(default)s')
    parser.add_argument('--temperature', type=float, default=config.temperature, help='temperature sets the temperature parameter of LLM reasoning. Default: %(default)s')
    parser.add_argument('--flashAttention2', type=bool, default=config.flashAttention2, help='flashAttention2 is the switch to enable flash attention 2 to speed up inference. It may have a certain impact on output quality. Default: %(default)s')
    parser.add_argument('--contextWindowRatio', type=float, default=config.contextWindowRatio, help='contextWindowRatio is a user-specified proportion coefficient, which determines the proportion of the upper limit of the prompt length constructed during inference to the LLM context window in some cases. Default: %(default)s')
    parser.add_argument('--speechOn', type=bool, default=config.speechOn, help='speechOn is the switch to enable voice conversation. Please note that the voice dialogue is currently not smooth yet. Default: %(default)s')
    parser.add_argument('--ttsDevice', type=str, default=config.ttsDevice, help='ttsDevice specifies the computing device used by the text-to-speech model. You can set it to "cuda" if there is enough video memory. Default: %(default)s')
    parser.add_argument('--sttDevice', type=str, default=config.sttDevice, help='sttDevice specifies the computing device used by the speech-to-text model. You can set it to "cuda" if there is enough video memory. Default: %(default)s')
    parser.add_argument('--resetApiKey', action='store_true', help="Whether to reset the model's API key after startup.")
    parser.add_argument('--chatHistoryPath', type=str, default=config.chatHistoryPath, help='chatHistoryPath is used to specify the chat history storage path. Default: %(default)s')
    parser.add_argument('--certificate', type=str, default=config.certificate, help='Certificate settings for the web interface. The simplest option is an empty string, which will use the HTTP protocol for the UI web page. Setting it to \'adhoc\' will use a self-generated certificate, providing encryption for the data flow between the UI and server, but it requires dismissing browser security warnings. The most secure method is to apply for a certificate and set this parameter to \'{"cert": "your_cert.pem", "key": "your_key.pem")\'. Default: %(default)s')
    parser.add_argument('--expose', type=bool, default=False, help='Whether to provide public access. Default: %(default)s')
    parser.add_argument('--logLevel', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: %(default)s')
    kwargs = vars(parser.parse_args())
    log_level = getattr(logging, kwargs['logLevel'].upper(), logging.INFO)
    logger.setLevel(log_level)
    logger.info(f'Log level set to {kwargs['logLevel']}')
    config.Check4Update(kwargs['modelID'], kwargs['resetApiKey'])
    config.Update(kwargs)
    logger.info(f'Configuration updated with command line arguments')
    try:
        Init()
        if kwargs['certificate'] == '':
            ssl_context = None
            logger.info('Starting server with HTTP (no SSL)')
        elif kwargs['certificate'] == 'adhoc':
            ssl_context = 'adhoc'
            logger.info('Starting server with self-signed certificate')
        else:
            try:
                certCfg = json.loads(kwargs['certificate'])
                ssl_context = (certCfg['cert'], certCfg['key'])
                logger.info(f'Starting server with certificate: {certCfg['cert']} and key: {certCfg['key']}')
            except Exception as e:
                error_msg = 'The certificate configuration you entered could not be recognized. Please set it according to the following format: {"cert": "your_cert.pem", "key": "your_key.pem")'
                logger.error(f'{error_msg}. Error: {str(e)}')
                print(error_msg)
                sys.exit(0)
        host = '0.0.0.0' if kwargs['expose'] else '127.0.0.1'
        port = 5000
        logger.info(f'Starting Flask server on {host}:{port}')
        app.run(debug=False, ssl_context=ssl_context, host=host, port=port)
    except Exception as e:
        error_msg = f'Encountered an exception, AIlice is exiting: {str(e)}'
        logger.critical(error_msg, exc_info=True)
        print(error_msg)
        if hasattr(e, 'tb'):
            print(e.tb)
        else:
            traceback.print_tb(e.__traceback__)
        TerminateSubprocess()
        raise

def Init():
    print(colored('In order to simplify installation and usage, we have set local execution as the default behavior, which means AI has complete control over the local environment. To prevent irreversible losses due to potential AI errors, you may consider one of the following two methods: the first one, run AIlice in a virtual machine; the second one, install Docker, use the provided Dockerfile to build an image and container, and modify the relevant configurations in config.json. For detailed instructions, please refer to the documentation.', 'red'))
    print(colored('If you find that ailice is running slowly or experiencing high CPU usage, please run `ailice_turbo` to install GPU acceleration support.', 'green'))
    StartServices()
    InitServer()
    return

def main():
    config.Initialize(configFile=os.path.join(appdirs.user_config_dir('ailice', 'Steven Lu'), 'config.json'))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelID', type=str, default=config.modelID, help='modelID specifies the model. There are two modes for model configuration. In the first mode, the model is uniformly specified by modelID. In the second mode, different types of agents will run on different models. When this parameter is an empty string (unspecified), the second mode will be used automatically, i.e., the models configured individually for different agents under the agentModelConfig field in config.json will be used. The currently supported models can be seen in config.json. Default: %(default)s')
    parser.add_argument('--quantization', type=str, default=config.quantization, help='quantization is the quantization option, you can choose 4bit or 8bit. Default: %(default)s')
    parser.add_argument('--maxMemory', type=dict, default=config.maxMemory, help='maxMemory is the memory video memory capacity constraint, the format when set is like "{0:"23GiB", 1:"24GiB", "cpu": "64GiB"}". Default: %(default)s')
    parser.add_argument('--prompt', type=str, default=config.prompt, help='prompt specifies the prompt to be executed, which is the type of agent. Default: %(default)s')
    parser.add_argument('--temperature', type=float, default=config.temperature, help='temperature sets the temperature parameter of LLM reasoning. Default: %(default)s')
    parser.add_argument('--flashAttention2', type=bool, default=config.flashAttention2, help='flashAttention2 is the switch to enable flash attention 2 to speed up inference. It may have a certain impact on output quality. Default: %(default)s')
    parser.add_argument('--contextWindowRatio', type=float, default=config.contextWindowRatio, help='contextWindowRatio is a user-specified proportion coefficient, which determines the proportion of the upper limit of the prompt length constructed during inference to the LLM context window in some cases. Default: %(default)s')
    parser.add_argument('--speechOn', type=bool, default=config.speechOn, help='speechOn is the switch to enable voice conversation. Please note that the voice dialogue is currently not smooth yet. Default: %(default)s')
    parser.add_argument('--ttsDevice', type=str, default=config.ttsDevice, help='ttsDevice specifies the computing device used by the text-to-speech model. You can set it to "cuda" if there is enough video memory. Default: %(default)s')
    parser.add_argument('--sttDevice', type=str, default=config.sttDevice, help='sttDevice specifies the computing device used by the speech-to-text model. You can set it to "cuda" if there is enough video memory. Default: %(default)s')
    parser.add_argument('--resetApiKey', action='store_true', help="Whether to reset the model's API key after startup.")
    parser.add_argument('--chatHistoryPath', type=str, default=config.chatHistoryPath, help='chatHistoryPath is used to specify the chat history storage path. Default: %(default)s')
    parser.add_argument('--certificate', type=str, default=config.certificate, help='Certificate settings for the web interface. The simplest option is an empty string, which will use the HTTP protocol for the UI web page. Setting it to \'adhoc\' will use a self-generated certificate, providing encryption for the data flow between the UI and server, but it requires dismissing browser security warnings. The most secure method is to apply for a certificate and set this parameter to \'{"cert": "your_cert.pem", "key": "your_key.pem")\'. Default: %(default)s')
    parser.add_argument('--expose', type=bool, default=False, help='Whether to provide public access. Default: %(default)s')
    kwargs = vars(parser.parse_args())
    config.Check4Update(kwargs['modelID'], kwargs['resetApiKey'])
    config.Update(kwargs)
    try:
        Init()
        if kwargs['certificate'] == '':
            ssl_context = None
        elif kwargs['certificate'] == 'adhoc':
            ssl_context = 'adhoc'
        else:
            try:
                certCfg = json.loads(kwargs['certificate'])
                ssl_context = (certCfg['cert'], certCfg['key'])
            except Exception as e:
                print('The certificate configuration you entered could not be recognized. Please set it according to the following format: {"cert": "your_cert.pem", "key": "your_key.pem")')
                sys.exit(0)
        app.run(debug=False, ssl_context=ssl_context, host='0.0.0.0' if kwargs['expose'] else '127.0.0.1', port=5000)
    except Exception as e:
        print(f'Encountered an exception, AIlice is exiting: {str(e)}')
        print(e.tb) if hasattr(e, 'tb') else traceback.print_tb(e.__traceback__)
        TerminateSubprocess()
        raise

def finetune(modelID, dataset: str, dataDir: str, epochs: int, maxWindow: int, outDir: str, logDir: str):
    config.Initialize(needOpenaiGPTKey=False)
    modelType, modelName = (modelID[:modelID.find(':')], modelID[modelID.find(':') + 1:])
    ds = load_dataset(dataset, maxWindow=maxWindow, data_dir=dataDir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(modelName, truncation=True, max_length=maxWindow, add_special_tokens=True, add_bos_token=False, add_eos_token=False, legacy=False)
    tokenizer.pad_token = tokenizer.unk_token
    quant_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
    model = transformers.AutoModelForCausalLM.from_pretrained(modelName, quantization_config=quant_config, device_map='auto')
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    def tokenizeOpenorca(batch):
        concatenatedSamples = [f'<|im_start|>system\n{system_prompt}\n<|im_end|>\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n{response}\n<|im_end|>' for system_prompt, question, response in zip(batch['system_prompt'], batch['question'], batch['response'])]
        tokenizedInputs = tokenizer(concatenatedSamples, padding=True, truncation=True, max_length=maxWindow, return_tensors='pt')
        return tokenizedInputs

    def tokenizeAIlice(batch):
        modelCfg = config.models[modelType]['modelList'][modelName]
        formatter = CreateFormatter(modelCfg['formatter'], tokenizer=None, systemAsUser=modelCfg['systemAsUser'])
        concatenatedSamples = [formatter(prompt0='', conversations=[{'role': role, 'msg': msg} for role, msg in zip(conv['role'], conv['msg'])], encode=False, assistTag=False)[0] for conv in batch['conversations']]
        tokenizedInputs = tokenizer(concatenatedSamples, padding=True, truncation=True, max_length=maxWindow, return_tensors='pt')
        return tokenizedInputs
    trainData = ds['train'].map(tokenizeAIlice, batched=True, num_proc=32, remove_columns=['conversations'])
    trainData = trainData.add_column('labels', trainData['input_ids'])
    trainData = trainData.with_format('torch')
    validData = ds['validation'].map(tokenizeAIlice, batched=True, num_proc=32, remove_columns=['conversations'])
    validData = validData.add_column('labels', validData['input_ids'])
    validData = validData.with_format('torch')
    loraConfig = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES, lora_dropout=LORA_DROPOUT, bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, loraConfig)
    model.print_trainable_parameters()
    trainingArguments = transformers.TrainingArguments(per_device_train_batch_size=MICRO_BATCH_SIZE, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, warmup_ratio=0.1, num_train_epochs=epochs, learning_rate=LEARNING_RATE, fp16=True, logging_steps=10, optim='paged_adamw_8bit', evaluation_strategy='no', save_strategy='steps', eval_steps=50, save_steps=50, output_dir=outDir, save_total_limit=3, load_best_model_at_end=False, report_to='tensorboard', logging_dir=logDir)
    collator = MyDataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt', padding=True)
    trainer = transformers.Trainer(model=model, train_dataset=trainData, eval_dataset=validData, args=trainingArguments, data_collator=collator)
    model.config.use_cache = False
    model = torch.compile(model)
    trainer.train()
    model.save_pretrained(outDir)

