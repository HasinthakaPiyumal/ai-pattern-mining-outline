# Cluster 18

def generate_post_vector(model: AutoModel, tokenizer: AutoTokenizer, texts, batch_size):
    all_outputs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_outputs = process_batch(model, tokenizer, batch_texts)
        all_outputs.append(batch_outputs)
    all_outputs_tensor = torch.cat(all_outputs, dim=0)
    return all_outputs_tensor.cpu()

@torch.no_grad()
def process_batch(model: AutoModel, tokenizer: AutoTokenizer, batch_texts: List[str]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    return outputs.pooler_output

