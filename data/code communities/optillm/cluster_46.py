# Cluster 46

def test_info_function():
    """Test the info function."""
    logger.info('Testing get_deepconf_info...')
    try:
        from optillm.deepconf.deepconf import get_deepconf_info
        info = get_deepconf_info()
        required_keys = ['name', 'description', 'local_models_only', 'variants', 'default_config']
        for key in required_keys:
            assert key in info, f'Missing key: {key}'
        assert info['local_models_only'] == True
        assert 'low' in info['variants'] and 'high' in info['variants']
        logger.info('✓ Info function tests passed')
        return True
    except Exception as e:
        logger.error(f'✗ Info function test failed: {e}')
        return False

def get_deepconf_info() -> Dict[str, Any]:
    """
    Get information about the DeepConf implementation.
    
    Returns:
        Dictionary with implementation details
    """
    return {'name': 'DeepConf', 'description': 'Confidence-aware reasoning with early termination', 'paper': 'Deep Think with Confidence (Fu et al., 2024)', 'arxiv': 'https://arxiv.org/abs/2508.15260', 'local_models_only': True, 'modes': ['online'], 'variants': ['low', 'high'], 'default_config': DEFAULT_CONFIG, 'features': ['Token-level confidence scoring', 'Early termination based on confidence', 'Warmup phase for threshold calibration', 'Consensus-based stopping', 'Weighted majority voting']}

