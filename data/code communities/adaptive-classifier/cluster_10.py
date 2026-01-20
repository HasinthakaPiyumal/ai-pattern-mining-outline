# Cluster 10

def _check_optimum_installed():
    """Helper to check if optimum is installed."""
    try:
        import optimum.onnxruntime
        return True
    except ImportError:
        return False

@pytest.mark.skipif(not _check_optimum_installed(), reason='optimum[onnxruntime] not installed')
def test_export_onnx_basic():
    """Test basic ONNX export functionality."""
    model_name = 'prajjwal1/bert-tiny'
    classifier = AdaptiveClassifier(model_name, use_onnx=False, device='cpu')
    texts = ['positive example', 'negative example']
    labels = ['positive', 'negative']
    classifier.add_examples(texts, labels)
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / 'onnx_model'
        result_path = classifier.export_onnx(onnx_path, quantize=False)
        assert result_path.exists()
        assert (result_path / 'model.onnx').exists()
        print(f'✓ ONNX model exported to {result_path}')

@pytest.mark.skipif(not _check_optimum_installed(), reason='optimum[onnxruntime] not installed')
def test_save_with_onnx():
    """Test saving classifier with ONNX export integrated."""
    model_name = 'prajjwal1/bert-tiny'
    classifier = AdaptiveClassifier(model_name, use_onnx=False, device='cpu')
    texts = ['positive text', 'negative text', 'neutral text']
    labels = ['positive', 'negative', 'neutral']
    classifier.add_examples(texts, labels)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / 'classifier_with_onnx'
        classifier._save_pretrained(save_path, include_onnx=True, quantize_onnx=False)
        assert (save_path / 'config.json').exists()
        assert (save_path / 'examples.json').exists()
        assert (save_path / 'model.safetensors').exists()
        assert (save_path / 'onnx').exists()
        assert (save_path / 'onnx' / 'model.onnx').exists()
        print('✓ Classifier saved with ONNX')

@pytest.mark.skipif(not _check_optimum_installed(), reason='optimum[onnxruntime] not installed')
def test_load_onnx_model():
    """Test loading a saved ONNX model."""
    model_name = 'prajjwal1/bert-tiny'
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / 'classifier_onnx'
        classifier_orig = AdaptiveClassifier(model_name, use_onnx=False, device='cpu')
        texts = ['happy', 'sad', 'angry']
        labels = ['positive', 'negative', 'negative']
        classifier_orig.add_examples(texts, labels)
        classifier_orig._save_pretrained(save_path, include_onnx=True)
        classifier_loaded = AdaptiveClassifier._from_pretrained(str(save_path), use_onnx=True)
        assert classifier_loaded.use_onnx is True
        print('✓ ONNX model loaded successfully')
        predictions = classifier_loaded.predict('very happy')
        assert len(predictions) > 0
        print(f'✓ Predictions work: {predictions[:2]}')

@pytest.mark.skipif(not _check_optimum_installed(), reason='optimum[onnxruntime] not installed')
def test_onnx_prediction_consistency():
    """Test that predictions are consistent after export and reload."""
    model_name = 'prajjwal1/bert-tiny'
    test_text = 'This is a test for consistency'
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / 'classifier_consistency'
        classifier_pytorch = AdaptiveClassifier(model_name, use_onnx=False, device='cpu')
        texts = ['good', 'bad', 'okay']
        labels = ['positive', 'negative', 'neutral']
        classifier_pytorch.add_examples(texts, labels)
        pred_pytorch = classifier_pytorch.predict(test_text, k=3)
        classifier_pytorch._save_pretrained(save_path, include_onnx=True)
        classifier_onnx = AdaptiveClassifier._from_pretrained(str(save_path), use_onnx=True)
        pred_onnx = classifier_onnx.predict(test_text, k=3)
        print(f'PyTorch predictions: {pred_pytorch}')
        print(f'ONNX predictions: {pred_onnx}')
        assert pred_pytorch[0][0] == pred_onnx[0][0], 'Top prediction differs between PyTorch and ONNX'
        for (label_pt, score_pt), (label_ox, score_ox) in zip(pred_pytorch, pred_onnx):
            assert label_pt == label_ox, f'Label mismatch: {label_pt} vs {label_ox}'
            score_diff = abs(score_pt - score_ox)
            assert score_diff < 0.05, f'Score difference too large for {label_pt}: {score_diff}'
        print('✓ Predictions are consistent between PyTorch and ONNX')

@pytest.mark.skipif(not _check_optimum_installed(), reason='optimum[onnxruntime] not installed')
def test_auto_detection_loads_onnx():
    """Test that auto-detection loads ONNX when available on CPU."""
    model_name = 'prajjwal1/bert-tiny'
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / 'classifier_auto'
        classifier_orig = AdaptiveClassifier(model_name, use_onnx=False, device='cpu')
        texts = ['example one', 'example two']
        labels = ['class1', 'class2']
        classifier_orig.add_examples(texts, labels)
        classifier_orig._save_pretrained(save_path, include_onnx=True)
        classifier_auto = AdaptiveClassifier._from_pretrained(str(save_path), use_onnx='auto', device='cpu')
        assert classifier_auto.use_onnx is True
        print('✓ Auto-detection correctly loads ONNX on CPU')

def test_fallback_when_onnx_not_available():
    """Test that loading works even when ONNX not in save directory."""
    model_name = 'prajjwal1/bert-tiny'
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / 'classifier_no_onnx'
        classifier_orig = AdaptiveClassifier(model_name, use_onnx=False, device='cpu')
        texts = ['text one', 'text two']
        labels = ['A', 'B']
        classifier_orig.add_examples(texts, labels)
        classifier_orig._save_pretrained(save_path, include_onnx=False)
        classifier_loaded = AdaptiveClassifier._from_pretrained(str(save_path), use_onnx=True)
        assert classifier_loaded.use_onnx is False
        print('✓ Correctly falls back to PyTorch when ONNX not available')
        predictions = classifier_loaded.predict('test')
        assert len(predictions) > 0

