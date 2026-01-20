# Cluster 24

def print_metrics(metrics: List[Dict[str, Any]], indent: int=0):
    """Print metrics in a hierarchical format with proper indentation and colors"""
    indent_str = '  ' * indent
    for metric in metrics:
        duration = metric['duration']
        total_duration = metric['total_duration']
        operation = metric['operation']
        metadata = metric.get('metadata', {})
        meta_str = ''
        if metadata:
            meta_parts = []
            for key, value in metadata.items():
                if key == 'tokens':
                    meta_parts.append(f'tokens={value}')
                elif key == 'reasoning_length':
                    meta_parts.append(f'reasoning={value}')
                elif key == 'llm':
                    meta_parts.append(f'llm={value}')
                elif key == 'model':
                    meta_parts.append(f'model={value}')
            if meta_parts:
                meta_str = f' ({', '.join(meta_parts)})'
        unacc = duration - total_duration
        timing_str = f'{duration:.3f}s'
        if unacc > 0.1:
            timing_str += f' (unacc: {unacc:.3f}s)'
        print(f'{indent_str}\x1b[93m{operation}{meta_str}:\x1b[0m {timing_str}')
        if metric.get('children'):
            print_metrics(metric['children'], indent + 1)

def log_metrics():
    """Log all collected metrics"""
    metrics = timing_tracker.get_metrics()
    print('\n\x1b[94mTiming Metrics:\x1b[0m')
    print_metrics(metrics)
    timing_tracker.clear()

