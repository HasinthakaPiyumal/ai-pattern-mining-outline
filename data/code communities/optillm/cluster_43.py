# Cluster 43

def test_confidence_calculator():
    """Test ConfidenceCalculator functionality."""
    logger.info('Testing ConfidenceCalculator...')
    try:
        import torch
        from optillm.deepconf.confidence import ConfidenceCalculator
        calculator = ConfidenceCalculator(window_size=10, top_k=3)
        dummy_logits = torch.randn(1000)
        entropy = calculator.calculate_token_entropy(dummy_logits)
        assert isinstance(entropy, float) and entropy > 0
        confidence = calculator.calculate_token_confidence(dummy_logits)
        assert isinstance(confidence, float) and confidence > 0
        for _ in range(15):
            calculator.add_token_confidence(dummy_logits)
        stats = calculator.get_trace_statistics()
        assert 'average_confidence' in stats
        assert 'num_tokens' in stats
        assert stats['num_tokens'] == 15
        logger.info('✓ ConfidenceCalculator tests passed')
        return True
    except Exception as e:
        logger.error(f'✗ ConfidenceCalculator test failed: {e}')
        return False

