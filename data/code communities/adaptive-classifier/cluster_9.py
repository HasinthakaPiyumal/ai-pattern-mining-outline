# Cluster 9

@pytest.mark.skipif(not _check_optimum_installed(), reason='optimum[onnxruntime] not installed')
def test_onnx_initialization():
    """Test that ONNX model initializes correctly."""
    model_name = 'prajjwal1/bert-tiny'
    classifier = AdaptiveClassifier(model_name, use_onnx=True, device='cpu')
    assert classifier.use_onnx is True
    assert hasattr(classifier.model, 'model')

def test_auto_detection_cpu():
    """Test that auto-detection uses ONNX on CPU."""
    model_name = 'prajjwal1/bert-tiny'
    classifier = AdaptiveClassifier(model_name, device='cpu', use_onnx='auto')
    if _check_optimum_installed():
        assert classifier.use_onnx is True
    else:
        assert classifier.use_onnx is False

@pytest.mark.skipif(not _check_optimum_installed(), reason='optimum[onnxruntime] not installed')
def test_embedding_consistency():
    """Test that ONNX and PyTorch produce similar embeddings."""
    model_name = 'prajjwal1/bert-tiny'
    test_text = 'This is a test sentence for embedding comparison.'
    classifier_pytorch = AdaptiveClassifier(model_name, use_onnx=False, device='cpu')
    classifier_onnx = AdaptiveClassifier(model_name, use_onnx=True, device='cpu')
    embedding_pytorch = classifier_pytorch._get_embeddings([test_text])[0]
    embedding_onnx = classifier_onnx._get_embeddings([test_text])[0]
    emb_pytorch_np = embedding_pytorch.cpu().numpy()
    emb_onnx_np = embedding_onnx.cpu().numpy()
    assert emb_pytorch_np.shape == emb_onnx_np.shape
    cosine_sim = np.dot(emb_pytorch_np, emb_onnx_np) / (np.linalg.norm(emb_pytorch_np) * np.linalg.norm(emb_onnx_np))
    print(f'Cosine similarity between PyTorch and ONNX embeddings: {cosine_sim:.6f}')
    assert cosine_sim > 0.99, f'Embeddings differ too much: cosine_sim={cosine_sim}'

@pytest.mark.skipif(not _check_optimum_installed(), reason='optimum[onnxruntime] not installed')
def test_onnx_with_training():
    """Test that ONNX model works with adaptive classifier training."""
    model_name = 'prajjwal1/bert-tiny'
    classifier = AdaptiveClassifier(model_name, use_onnx=True, device='cpu')
    texts = ['This is a positive example', 'This is a negative example', 'Another positive case', 'Another negative case']
    labels = ['positive', 'negative', 'positive', 'negative']
    classifier.add_examples(texts, labels)
    predictions = classifier.predict('This seems positive')
    assert len(predictions) > 0
    assert all((isinstance(label, str) and isinstance(score, float) for label, score in predictions))

def test_explicit_disable_onnx():
    """Test that ONNX can be explicitly disabled."""
    model_name = 'prajjwal1/bert-tiny'
    classifier = AdaptiveClassifier(model_name, use_onnx=False, device='cpu')
    assert classifier.use_onnx is False

def test_fallback_on_import_error():
    """Test that classifier falls back to PyTorch if optimum not installed."""
    model_name = 'prajjwal1/bert-tiny'
    classifier = AdaptiveClassifier(model_name, use_onnx=True, device='cpu')
    assert classifier.use_onnx in [True, False]
    embedding = classifier._get_embeddings(['test'])[0]
    assert embedding is not None
    assert embedding.shape[0] > 0

