# Cluster 40

class PretrainedEmbedding(nn.Module):

    def __init__(self, feature_name, feature_spec, pretrain_path, vocab_path, embedding_dim, pretrain_dim, pretrain_usage='init', embedding_initializer='partial(nn.init.normal_, std=1e-4)'):
        """
        Fusion pretrained embedding with ID embedding
        :param: fusion_type: init/sum/concat
        """
        super().__init__()
        assert pretrain_usage in ['init', 'sum', 'concat']
        self.pretrain_usage = pretrain_usage
        self.embedding_initializer = get_initializer(embedding_initializer)
        padding_idx = feature_spec.get('padding_idx', None)
        self.oov_idx = feature_spec['oov_idx']
        self.freeze_emb = feature_spec['freeze_emb']
        self.pretrain_embedding = self.load_pretrained_embedding(feature_spec['vocab_size'], pretrain_dim, pretrain_path, vocab_path, feature_name, freeze=self.freeze_emb, padding_idx=padding_idx)
        if pretrain_usage != 'init':
            self.id_embedding = nn.Embedding(feature_spec['vocab_size'], embedding_dim, padding_idx=padding_idx)
        self.proj = None
        if pretrain_usage in ['init', 'sum'] and embedding_dim != pretrain_dim:
            self.proj = nn.Linear(pretrain_dim, embedding_dim, bias=False)
        if pretrain_usage == 'concat':
            self.proj = nn.Linear(pretrain_dim + embedding_dim, embedding_dim, bias=False)

    def init_weights(self):
        if self.pretrain_usage in ['sum', 'concat']:
            nn.init.zeros_(self.id_embedding.weight)
            self.embedding_initializer(self.id_embedding.weight[1:self.oov_idx, :])

    def load_feature_vocab(self, vocab_path, feature_name):
        with io.open(vocab_path, 'r', encoding='utf-8') as fd:
            vocab = json.load(fd)
            vocab_type = type(list(vocab.items())[1][0])
        return (vocab[feature_name], vocab_type)

    def load_pretrained_embedding(self, vocab_size, pretrain_dim, pretrain_path, vocab_path, feature_name, freeze=False, padding_idx=None):
        embedding_layer = nn.Embedding(vocab_size, pretrain_dim, padding_idx=padding_idx)
        if freeze:
            embedding_matrix = np.zeros((vocab_size, pretrain_dim))
        else:
            embedding_matrix = np.random.normal(loc=0, scale=0.0001, size=(vocab_size, pretrain_dim))
            if padding_idx:
                embedding_matrix[padding_idx, :] = np.zeros(pretrain_dim)
        logging.info('Loading pretrained_emb: {}'.format(pretrain_path))
        keys, embeddings = load_pretrain_emb(pretrain_path, keys=['key', 'value'])
        assert embeddings.shape[-1] == pretrain_dim, f'pretrain_dim={pretrain_dim} not correct.'
        vocab, vocab_type = self.load_feature_vocab(vocab_path, feature_name)
        keys = keys.astype(vocab_type)
        for idx, word in enumerate(keys):
            if word in vocab:
                embedding_matrix[vocab[word]] = embeddings[idx]
        embedding_layer.weight = torch.nn.Parameter(torch.from_numpy(embedding_matrix).float())
        if freeze:
            embedding_layer.weight.requires_grad = False
        return embedding_layer

    def forward(self, inputs):
        mask = (inputs <= self.oov_idx).float()
        pretrain_emb = self.pretrain_embedding(inputs)
        if not self.freeze_emb:
            pretrain_emb = pretrain_emb * mask.unsqueeze(-1)
        if self.pretrain_usage == 'init':
            if self.proj is not None:
                feature_emb = self.proj(pretrain_emb)
            else:
                feature_emb = pretrain_emb
        else:
            id_emb = self.id_embedding(inputs)
            id_emb = id_emb * mask.unsqueeze(-1)
            if self.pretrain_usage == 'sum':
                if self.proj is not None:
                    feature_emb = self.proj(pretrain_emb) + id_emb
                else:
                    feature_emb = pretrain_emb + id_emb
            elif self.pretrain_usage == 'concat':
                feature_emb = torch.cat([pretrain_emb, id_emb], dim=-1)
                feature_emb = self.proj(feature_emb)
        return feature_emb

def load_pretrain_emb(pretrain_path, keys=['key', 'value']):
    if type(keys) != list:
        keys = [keys]
    if pretrain_path.endswith('h5'):
        with h5py.File(pretrain_path, 'r') as hf:
            values = [hf[k][:] for k in keys]
    elif pretrain_path.endswith('npz'):
        npz = np.load(pretrain_path)
        values = [npz[k] for k in keys]
    elif pretrain_path.endswith('parquet'):
        df = pd.read_parquet(pretrain_path)
        values = [df[k].values for k in keys]
    else:
        raise ValueError(f'Embedding format not supported: {pretrain_path}')
    return values[0] if len(values) == 1 else values

