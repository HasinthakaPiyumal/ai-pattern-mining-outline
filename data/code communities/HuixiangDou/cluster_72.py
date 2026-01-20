# Cluster 72

class SentenceEmbeddingPipeline(Pipeline):

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        return (preprocess_kwargs, {}, {})

    def preprocess(self, inputs):
        encoded_inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
        return encoded_inputs

    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        return {'outputs': outputs, 'attention_mask': model_inputs['attention_mask']}

    def postprocess(self, model_outputs):
        sentence_embeddings = mean_pooling(model_outputs['outputs'], model_outputs['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-09)

