# Cluster 23

def tokenizeAIlice(batch):
    modelCfg = config.models[modelType]['modelList'][modelName]
    formatter = CreateFormatter(modelCfg['formatter'], tokenizer=None, systemAsUser=modelCfg['systemAsUser'])
    concatenatedSamples = [formatter(prompt0='', conversations=[{'role': role, 'msg': msg} for role, msg in zip(conv['role'], conv['msg'])], encode=False, assistTag=False)[0] for conv in batch['conversations']]
    tokenizedInputs = tokenizer(concatenatedSamples, padding=True, truncation=True, max_length=maxWindow, return_tensors='pt')
    return tokenizedInputs

def CreateFormatter(formatterClsName: str, tokenizer, systemAsUser):
    formatterList = [obj for name, obj in inspect.getmembers(inspect.getmodule(inspect.currentframe())) if inspect.isclass(obj) and name.startswith('AFormatter')]
    for formatterCls in formatterList:
        if formatterClsName == formatterCls.__name__:
            return formatterCls(tokenizer=tokenizer, systemAsUser=systemAsUser)
    raise ValueError(f'{formatterClsName} is not a valid formatter class name.')

class AModelChatGPT:

    def __init__(self, modelType: str, modelName: str, config):
        self.tokenizer = None
        self.modelType = modelType
        self.modelName = modelName
        self.config = config
        self.modelCfg = self.config.models[modelType]['modelList'][modelName]
        self.formatter = CreateFormatter(self.modelCfg['formatter'], tokenizer=self.tokenizer, systemAsUser=self.modelCfg['systemAsUser'])
        self.contextWindow = self.modelCfg['contextWindow']
        return

    def Generate(self, prompt: tuple[list[dict[str, str]], int], proc: callable, endchecker: callable, temperature: float, gasTank) -> str:
        currentPosition = 0
        text = ''
        extras = {}
        extras.update({'max_tokens': 4096} if 'vision' in self.modelName else {})
        extras.update(self.modelCfg.get('args', {}))
        extras.update({'temperature': temperature} if None != temperature else {})
        try:
            gasTank.Consume(resourceType='ChatGPT/InputTokens', amount=prompt[1])
            with openai.OpenAI(api_key=self.config.models[self.modelType]['apikey'], base_url=self.config.models[self.modelType]['baseURL']) as client:
                for chunk in client.chat.completions.create(model=self.modelName, messages=prompt[0], stream=True, timeout=60, **extras):
                    text += chunk.choices[0].delta.content or ''
                    if endchecker(text):
                        break
                    sentences = [x for x in sentences_split(text[currentPosition:])]
                    if 2 <= len(sentences) and '' != sentences[0].strip():
                        gasTank.Consume(resourceType='ChatGPT/OutputTokens', amount=len(sentences[0]) // 4)
                        proc(txt=sentences[0])
                        currentPosition += len(sentences[0])
        except openai.AuthenticationError as e:
            msg = colored('The program encountered an authorization error. Please check your API key:', 'yellow') + colored(f'\n\n{self.modelType}: ', 'green') + colored(f"'{self.config.models[self.modelType]['apikey']}'\n\n", 'blue') + colored("If it's incorrect, append '--resetApiKey' to the command parameters you are using to restart ailice and reset the API key.", 'yellow')
            print('\n\n', msg)
            print('\n\nException:\n', str(e))
            os._exit(1)
        gasTank.Consume(resourceType='ChatGPT/OutputTokens', amount=len(text[currentPosition:]) // 4)
        proc(txt=text[currentPosition:])
        return text

class AModelCausalLM:

    def __init__(self, modelType: str, modelName: str, config):
        self.modelType = modelType
        self.config = config
        self.tokenizer = None
        self.model = None
        self.configMap = {'': None, None: None, '4bit': transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16), '8bit': transformers.BitsAndBytesConfig(load_in_8bit=True)}
        self.LoadModel(modelName)
        if modelType not in config.models or modelName not in config.models[modelType]['modelList']:
            print(f'LLM {modelType}:{modelName} not supported yet.')
            exit(-1)
        modelCfg = config.models[modelType]['modelList'][modelName]
        self.formatter = CreateFormatter(modelCfg['formatter'], tokenizer=self.tokenizer, systemAsUser=modelCfg['systemAsUser'])
        self.contextWindow = modelCfg['contextWindow']
        return

    def LoadModel(self, modelName: str):
        if 'peft' == self.modelType:
            self.LoadModel_PEFT(modelName=modelName)
        else:
            self.LoadModel_Default(modelName=modelName)
        return

    def LoadModel_Default(self, modelName: str):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(modelName, use_fast=False, legacy=False, force_download=False, resume_download=True)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(modelName, device_map='auto', low_cpu_mem_usage=True, quantization_config=self.configMap[self.config.quantization], attn_implementation='flash_attention_2' if self.config.flashAttention2 else None, max_memory=self.config.maxMemory, force_download=False, resume_download=True)
        return

    def LoadModel_PEFT(self, modelName: str):
        if not PEFT_INSTALLED:
            print('peft not installed. Please install it with the following command: pip install -e .[finetuning]')
            sys.exit()
        peftConfig = PeftConfig.from_pretrained(modelName)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(peftConfig.base_model_name_or_path, use_fast=False)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(peftConfig.base_model_name_or_path, device_map='auto', low_cpu_mem_usage=True, quantization_config=self.configMap[self.config.quantization], attn_implementation='flash_attention_2' if self.config.flashAttention2 else None, max_memory=self.config.maxMemory)
        self.model = PeftModel.from_pretrained(self.model, modelName)
        return

    def Generate(self, prompt: str, proc: callable, endchecker: callable, temperature: float, gasTank) -> str:
        predictedIDs = torch.tensor([prompt[0]]).cuda()
        generatedIDs = None
        pastKeyValues = None
        currentPosition = 0
        text = ''
        gasTank.Consume(resourceType='HFCausalLM/InputTokens', amount=predictedIDs.shape[1])
        for _ in range(4096):
            with torch.no_grad():
                outputs = self.model(input_ids=predictedIDs, past_key_values=pastKeyValues, use_cache=True)
            logits = outputs.logits
            pastKeyValues = outputs.past_key_values
            if temperature > 1e-09:
                scaledLogits = logits / temperature
                probs = torch.nn.functional.softmax(scaledLogits, dim=-1)
                predictedIDs = torch.multinomial(probs[:, -1, :], 1)
            else:
                predictedIDs = torch.argmax(logits[..., -1, :], dim=-1, keepdim=True)
            gasTank.Consume(resourceType='HFCausalLM/OutputTokens', amount=predictedIDs.shape[1])
            generatedIDs = predictedIDs if None == generatedIDs else torch.cat((generatedIDs, predictedIDs), dim=-1)
            text = self.tokenizer.decode(generatedIDs[0].cpu().numpy(), skip_special_tokens=True)
            if predictedIDs.item() == self.tokenizer.eos_token_id or endchecker(text):
                break
            sentences = [x for x in sentences_split(text[currentPosition:])]
            if 2 <= len(sentences) and '' != sentences[0].strip():
                proc(txt=sentences[0])
                currentPosition += len(sentences[0])
        proc(txt=text[currentPosition:])
        return text

class AModelAnthropic:

    def __init__(self, modelType: str, modelName: str, config):
        self.tokenizer = None
        self.modelType = modelType
        self.modelName = modelName
        self.config = config
        self.modelCfg = config.models[modelType]['modelList'][modelName]
        self.formatter = CreateFormatter(self.modelCfg['formatter'], tokenizer=self.tokenizer, systemAsUser=self.modelCfg['systemAsUser'])
        self.contextWindow = self.modelCfg['contextWindow']
        return

    def Generate(self, prompt: tuple[list[dict[str, str]], int], proc: callable, endchecker: callable, temperature: float, gasTank) -> str:
        currentPosition = 0
        text = ''
        extras = {}
        extras.update(self.modelCfg.get('args', {}))
        extras.update({'temperature': temperature} if None != temperature else {})
        try:
            gasTank.Consume(resourceType='Anthropic/InputTokens', amount=prompt[1])
            with anthropic.Anthropic(api_key=self.config.models[self.modelType]['apikey'], base_url=self.config.models[self.modelType]['baseURL']) as client:
                with client.messages.stream(model=self.modelName, max_tokens=4096, system=prompt[0][0]['content'], messages=prompt[0][1:], timeout=60, **extras) as stream:
                    for delta in stream.text_stream:
                        text += delta
                        if endchecker(text):
                            break
                        sentences = [x for x in sentences_split(text[currentPosition:])]
                        if 2 <= len(sentences) and '' != sentences[0].strip():
                            gasTank.Consume(resourceType='Anthropic/OutputTokens', amount=len(sentences[0]) // 4)
                            proc(txt=sentences[0])
                            currentPosition += len(sentences[0])
        except anthropic.AuthenticationError as e:
            msg = colored('The program encountered an authorization error. Please check your API key:', 'yellow') + colored(f'\n\n{self.modelType}: ', 'green') + colored(f"'{self.config.models[self.modelType]['apikey']}'\n\n", 'blue') + colored("If it's incorrect, append '--resetApiKey' to the command parameters you are using to restart ailice and reset the API key.", 'yellow')
            print('\n\n', msg)
            print('\n\nException:\n', str(e))
            os._exit(1)
        gasTank.Consume(resourceType='Anthropic/OutputTokens', amount=len(text[currentPosition:]) // 4)
        proc(txt=text[currentPosition:])
        return text

class AModelMistral:

    def __init__(self, modelType: str, modelName: str, config):
        self.tokenizer = None
        self.modelType = modelType
        self.modelName = modelName
        self.config = config
        self.modelCfg = config.models[modelType]['modelList'][modelName]
        self.formatter = CreateFormatter(self.modelCfg['formatter'], tokenizer=self.tokenizer, systemAsUser=self.modelCfg['systemAsUser'])
        self.contextWindow = self.modelCfg['contextWindow']
        return

    def Generate(self, prompt: tuple[list[dict[str, str]], int], proc: callable, endchecker: callable, temperature: float, gasTank) -> str:
        currentPosition = 0
        text = ''
        extras = {}
        extras.update(self.modelCfg.get('args', {}))
        extras.update({'temperature': temperature} if None != temperature else {})
        try:
            gasTank.Consume(resourceType='Mistral/InputTokens', amount=prompt[1])
            with Mistral(api_key=self.config.models[self.modelType]['apikey']) as client:
                for chunk in client.chat.stream(model=self.modelName, messages=prompt[0], timeout_ms=60000, **extras):
                    text += chunk.data.choices[0].delta.content or ''
                    if endchecker(text):
                        break
                    sentences = [x for x in sentences_split(text[currentPosition:])]
                    if 2 <= len(sentences) and '' != sentences[0].strip():
                        gasTank.Consume(resourceType='Mistral/OutputTokens', amount=len(sentences[0]) // 4)
                        proc(txt=sentences[0])
                        currentPosition += len(sentences[0])
        except models.sdkerror.SDKError as e:
            if 'Unauthorized' in e.body:
                msg = colored('The program encountered an authorization error. Please check your API key:', 'yellow') + colored(f'\n\n{self.modelType}: ', 'green') + colored(f"'{self.config.models[self.modelType]['apikey']}'\n\n", 'blue') + colored("If it's incorrect, append '--resetApiKey' to the command parameters you are using to restart ailice and reset the API key.", 'yellow')
                print('\n\n', msg)
                print('\n\nException:\n', str(e))
                os._exit(1)
            else:
                raise
        gasTank.Consume(resourceType='Mistral/OutputTokens', amount=len(text[currentPosition:]) // 4)
        proc(txt=text[currentPosition:])
        return text

