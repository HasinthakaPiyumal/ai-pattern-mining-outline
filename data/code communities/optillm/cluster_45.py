# Cluster 45

def test_config_validation():
    """Test configuration validation."""
    logger.info('Testing configuration validation...')
    try:
        from optillm.deepconf.deepconf import validate_deepconf_config, DEFAULT_CONFIG
        valid_config = DEFAULT_CONFIG.copy()
        validated = validate_deepconf_config(valid_config)
        assert validated == valid_config
        try:
            invalid_config = {'variant': 'invalid'}
            validate_deepconf_config(invalid_config)
            assert False, 'Should have raised ValueError'
        except ValueError:
            pass
        try:
            invalid_config = {'warmup_samples': -1}
            validate_deepconf_config(invalid_config)
            assert False, 'Should have raised ValueError'
        except ValueError:
            pass
        logger.info('✓ Configuration validation tests passed')
        return True
    except Exception as e:
        logger.error(f'✗ Configuration validation test failed: {e}')
        return False

def validate_deepconf_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize DeepConf configuration.
    
    Args:
        config: Input configuration dictionary
        
    Returns:
        Validated and normalized configuration
        
    Raises:
        ValueError: If configuration is invalid
    """
    validated = config.copy()
    if 'variant' in validated:
        if validated['variant'] not in ['low', 'high']:
            raise ValueError("variant must be 'low' or 'high'")
    numeric_params = {'warmup_samples': (1, 100), 'max_traces': (1, 1000), 'window_size': (100, 10000), 'top_k': (1, 100), 'min_trace_length': (10, 10000), 'max_tokens_per_trace': (100, 100000), 'consensus_threshold': (0.5, 1.0), 'temperature': (0.1, 2.0)}
    for param, (min_val, max_val) in numeric_params.items():
        if param in validated:
            value = validated[param]
            if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                raise ValueError(f'{param} must be between {min_val} and {max_val}')
    if validated.get('warmup_samples', 0) >= validated.get('max_traces', 100):
        raise ValueError('warmup_samples must be less than max_traces')
    return validated

