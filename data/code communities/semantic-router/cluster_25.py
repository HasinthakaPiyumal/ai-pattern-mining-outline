# Cluster 25

class TestPretrainedTokenizer:

    @pytest.fixture
    def tokenizer(self):
        return PretrainedTokenizer('google-bert/bert-base-uncased')

    def test_initialization(self, tokenizer):
        assert tokenizer.model_ident == 'google-bert/bert-base-uncased'
        assert tokenizer.add_special_tokens is False
        assert tokenizer.pad is True

    def test_vocab_size(self, tokenizer):
        assert isinstance(tokenizer.vocab_size, int)
        assert tokenizer.vocab_size > 0

    def test_config(self, tokenizer):
        config = tokenizer.config
        assert isinstance(config, dict)
        assert 'model_ident' in config
        assert 'add_special_tokens' in config
        assert 'pad' in config

    def test_tokenize_single_text(self, tokenizer):
        text = 'Hello world'
        tokens = tokenizer.tokenize(text)
        assert isinstance(tokens, np.ndarray)
        assert tokens.ndim == 2
        assert tokens.shape[0] == 1
        assert tokens.shape[1] > 0

    def test_tokenize_multiple_texts(self, tokenizer):
        texts = ['Hello world', 'Testing tokenization']
        tokens = tokenizer.tokenize(texts)
        assert isinstance(tokens, np.ndarray)
        assert tokens.ndim == 2
        assert tokens.shape[0] == 2

    def test_save_load_cycle(self, tokenizer):
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            tokenizer.save(tmp.name)
            loaded = PretrainedTokenizer.load(tmp.name)
            assert isinstance(loaded, PretrainedTokenizer)
            assert loaded.model_ident == tokenizer.model_ident
            assert loaded.add_special_tokens == tokenizer.add_special_tokens
            assert loaded.pad == tokenizer.pad

