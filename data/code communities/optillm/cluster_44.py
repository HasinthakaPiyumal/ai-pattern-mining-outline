# Cluster 44

def test_threshold_calibrator():
    """Test ConfidenceThresholdCalibrator functionality."""
    logger.info('Testing ConfidenceThresholdCalibrator...')
    try:
        from optillm.deepconf.confidence import ConfidenceThresholdCalibrator
        calibrator = ConfidenceThresholdCalibrator(variant='low')
        for i in range(5):
            stats = {'average_confidence': 1.0 + i * 0.1, 'bottom_10_percent': 0.8 + i * 0.05, 'lowest_group': 0.7 + i * 0.02}
            calibrator.add_warmup_trace(stats)
        threshold = calibrator.calculate_threshold('average_confidence')
        assert isinstance(threshold, float) and threshold > 0
        should_terminate = calibrator.should_terminate_trace(0.5, threshold)
        import numpy as np
        assert isinstance(should_terminate, (bool, np.bool_))
        logger.info('✓ ConfidenceThresholdCalibrator tests passed')
        return True
    except Exception as e:
        import traceback
        logger.error(f'✗ ConfidenceThresholdCalibrator test failed: {e}')
        logger.error(traceback.format_exc())
        return False

