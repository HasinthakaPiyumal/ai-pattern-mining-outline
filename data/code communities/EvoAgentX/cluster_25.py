# Cluster 25

def main():
    flow = Workflow()
    registry = PromptRegistry()
    registry.register_path(flow, 'system_prompt', name='sys_prompt')
    registry.register_path(flow, 'sampler.temperature')
    registry.register_path(flow, 'sampler.top_p')
    code_block = CodeBlock('run_workflow', lambda cfg: flow.run())

    def evaluator(cfg, result) -> float:
        return result['score']
    opt = RandomSearchOptimizer(registry, metric='score', max_trials=10)
    best_cfg, history = opt.run(code_block, evaluator)
    print('\n=== Trial history ===')
    for i, (cfg, score) in enumerate(history, 1):
        print(f'{i:02d}: score={score:.3f}, cfg={cfg}')
    print('\n=== Best ===')
    print(best_cfg)

def read_api_key(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

