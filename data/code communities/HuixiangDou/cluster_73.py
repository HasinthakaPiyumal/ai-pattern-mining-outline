# Cluster 73

def profiling():
    onnx_path = 'o4-opt-onnx'
    model = ORTModelForFeatureExtraction.from_pretrained(onnx_path, file_name='model.onnx')
    model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(onnx_path)
    vanilla_emb = SentenceEmbeddingPipeline(model=model, tokenizer=tokenizer)
    pred = vanilla_emb('Could you assist me in finding my lost card? vanilla_emb')
    print(pred[0][:5])
    onnx_path = 'onnx'
    model = ORTModelForFeatureExtraction.from_pretrained(onnx_path, file_name='model.onnx')
    model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(onnx_path)
    q8_emb = SentenceEmbeddingPipeline(model=model, tokenizer=tokenizer)
    pred = q8_emb('Could you assist me in finding my lost card? q8_emb')
    print(pred[0][:5])
    from time import perf_counter
    import numpy as np
    payload = 'Hello, my name is Philipp and I live in Nuremberg, Germany. Currently I am working as a Technical Lead at Hugging Face to democratize artificial intelligence through open source and open science. In the past I designed and implemented cloud-native machine learning architectures for fin-tech and insurance companies. I found my passion for cloud concepts and machine learning 5 years ago. Since then I never stopped learning. Currently, I am focusing myself in the area NLP and how to leverage models like BERT, Roberta, T5, ViT, and GPT2 to generate business value. I cannot wait to see what is next for me'
    print(f'Payload sequence length: {len(tokenizer(payload)['input_ids'])}')

    def measure_latency(pipe):
        latencies = []
        for _ in range(4):
            _ = pipe(payload)
        for _ in range(20):
            start_time = perf_counter()
            _ = pipe(payload)
            latency = perf_counter() - start_time
            latencies.append(latency)
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies)
        time_p95_ms = 1000 * np.percentile(latencies, 95)
        return (f'P95 latency (ms) - {time_p95_ms}; Average latency (ms) - {time_avg_ms:.2f} +\\- {time_std_ms:.2f};', time_p95_ms)
    vanilla_model = measure_latency(vanilla_emb)
    quantized_model = measure_latency(q8_emb)
    print(f'Vanilla model: {vanilla_model[0]}')
    print(f'Quantized model: {quantized_model[0]}')
    print(f'Improvement through quantization: {round(vanilla_model[1] / quantized_model[1], 2)}x')

def measure_latency(pipe):
    latencies = []
    for _ in range(4):
        _ = pipe(payload)
    for _ in range(20):
        start_time = perf_counter()
        _ = pipe(payload)
        latency = perf_counter() - start_time
        latencies.append(latency)
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies, 95)
    return (f'P95 latency (ms) - {time_p95_ms}; Average latency (ms) - {time_avg_ms:.2f} +\\- {time_std_ms:.2f};', time_p95_ms)

