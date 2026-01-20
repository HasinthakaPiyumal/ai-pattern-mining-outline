# Cluster 4

@torch.no_grad()
def main():
    args = parse_args()
    print('\n=============== Argument ===============')
    for key in vars(args):
        print(f'{key}: {vars(args)[key]}')
    print('========================================')
    model_id = args.model_id
    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    check_params(args, hf_config)
    quantization_config = generate_quanto_config(hf_config, args.quant_type)
    device_map = {'model.embed_tokens': 'cuda:0', 'model.norm': f'cuda:{args.world_size - 1}', 'lm_head': f'cuda:{args.world_size - 1}'}
    layers_per_device = hf_config.num_hidden_layers // args.world_size
    for i in range(args.world_size):
        for j in range(layers_per_device):
            device_map[f'model.layers.{i * layers_per_device + j}'] = f'cuda:{i}'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    message = [{'role': 'system', 'content': [{'type': 'text', 'text': 'You are a helpful assistant.'}]}, {'role': 'user', 'content': [{'type': 'text', 'text': 'Hello, what is the weather today?'}]}]
    tools = [{'name': 'get_location', 'description': 'Get the location of the user.', 'parameters': {'type': 'object', 'properties': {}}}, {'name': 'get_weather', 'description': 'Get the weather of a city.', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string', 'description': 'The name of the city'}}}}, {'name': 'get_news', 'description': 'Get the news.', 'parameters': {'type': 'object', 'properties': {'domain': {'type': 'string', 'description': 'The domain of the news'}}}}]
    text = tokenizer.apply_chat_template(message, tools, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(text, return_tensors='pt').to('cuda')
    quantized_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='bfloat16', device_map=device_map, quantization_config=quantization_config, trust_remote_code=True, offload_buffers=True)
    generation_config = GenerationConfig(max_new_tokens=20, eos_token_id=200020, use_cache=True)
    generated_ids = quantized_model.generate(**model_inputs, generation_config=generation_config)
    print(f'generated_ids: {generated_ids}')
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quant_type', type=str, default='default', choices=['default', 'int8'])
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--world_size', type=int, required=True)
    return parser.parse_args()

def check_params(args, hf_config: AutoConfig):
    if args.quant_type == 'int8':
        assert args.world_size >= 8, 'int8 weight-only quantization requires at least 8 GPUs'
    assert hf_config.num_hidden_layers % args.world_size == 0, f'num_hidden_layers({hf_config.num_hidden_layers}) must be divisible by world_size({args.world_size})'

def generate_quanto_config(hf_config: AutoConfig, quant_type: str):
    QUANT_TYPE_MAP = {'default': None, 'int8': QuantoConfig(weights='int8', modules_to_not_convert=['lm_head', 'embed_tokens'] + [f'model.layers.{i}.coefficient' for i in range(hf_config.num_hidden_layers)] + [f'model.layers.{i}.block_sparse_moe.gate' for i in range(hf_config.num_hidden_layers)])}
    return QUANT_TYPE_MAP[quant_type]

