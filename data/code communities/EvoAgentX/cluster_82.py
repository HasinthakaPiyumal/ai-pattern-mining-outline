# Cluster 82

@contextmanager
def suppress_logger_info():
    token = None
    try:
        current_level = silence_nesting.get()
        token = silence_nesting.set(current_level + 1)
        if current_level == 0:
            logger.remove()
            logger.add(sys.stdout, level='WARNING')
            log_file = get_log_file()
            if log_file is not None:
                logger.add(log_file, encoding='utf-8', level='WARNING', format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}')
        yield
    finally:
        new_level = silence_nesting.get() - 1
        silence_nesting.set(new_level)
        if new_level == 0:
            logger.remove()
            logger.add(sys.stdout, level='INFO')
            log_file = get_log_file()
            if log_file is not None:
                logger.add(log_file, encoding='utf-8', level='INFO', format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}')
        if token:
            silence_nesting.reset(token)

def get_log_file():
    """
    Get the path to the logging file.
    
    Returns:
        str: The path to the logging file
    """
    return save_logging_file

def main():
    llm_config = OpenAILLMConfig(model='gpt-4o-mini-2024-07-18', openai_key=OPENAI_API_KEY, top_p=0.85, temperature=0.2, frequency_penalty=0.0, presence_penalty=0.0)
    llm = OpenAILLM(config=llm_config)
    sew_graph = SEWWorkFlowGraph(llm_config=llm_config)
    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(sew_graph, llm_config=llm_config)
    humaneval = HumanEvalSplits()

    def collate_func(example: dict) -> dict:
        return {'question': example['prompt']}
    evaluator = Evaluator(llm=llm, agent_manager=agent_manager, collate_func=collate_func, num_workers=20, verbose=True)
    optimizer = SEWOptimizer(graph=sew_graph, evaluator=evaluator, llm=llm, max_steps=10, eval_rounds=1, repr_scheme='python', optimize_mode='prompt', order='zero-order')
    with suppress_logger_info():
        metrics = optimizer.evaluate(dataset=humaneval, eval_mode='test')
    print('Evaluation metrics: ', metrics)
    optimizer.optimize(dataset=humaneval)
    with suppress_logger_info():
        metrics = optimizer.evaluate(dataset=humaneval, eval_mode='test')
    print('Evaluation metrics: ', metrics)
    optimizer.save('debug/optimized_sew_workflow.json')

def main():
    llm_config = OpenAILLMConfig(model='gpt-4o-mini', openai_key=OPENAI_API_KEY)
    llm = OpenAILLM(config=llm_config)
    benchmark = HotPotQA(mode='dev')
    workflow = QAActionGraph(llm_config=llm_config, description='This workflow aims to address multi-hop QA tasks.')

    def collate_func(example: dict) -> dict:
        """
        Args:
            example (dict): A dictionary containing the raw example data.

        Returns: 
            The expected input for the (custom) workflow.
        """
        problem = 'Question: {}\n\n'.format(example['question'])
        context_list = []
        for item in example['context']:
            context = 'Title: {}\nText: {}'.format(item[0], ' '.join([t.strip() for t in item[1]]))
            context_list.append(context)
        context = '\n\n'.join(context_list)
        problem += 'Context: {}\n\n'.format(context)
        problem += 'Answer:'
        return {'problem': problem}

    def output_postprocess_func(output: dict) -> dict:
        """
        Args:
            output (dict): The output from the workflow.

        Returns: 
            The processed output that can be used to compute the metrics. The output will be directly passed to the benchmark's `evaluate` method. 
        """
        return output['answer']
    evaluator = Evaluator(llm=llm, collate_func=collate_func, output_postprocess_func=output_postprocess_func, verbose=True, num_workers=3)
    with suppress_logger_info():
        results = evaluator.evaluate(graph=workflow, benchmark=benchmark, eval_mode='dev', sample_k=6)
    print('Evaluation metrics: ', results)

def main():
    openai_config = OpenAILLMConfig(model='gpt-4o-mini', openai_key=OPENAI_API_KEY, stream=True, output_response=False)
    executor_llm = OpenAILLM(config=openai_config)
    optimizer_config = OpenAILLMConfig(model='gpt-4o', openai_key=OPENAI_API_KEY, stream=True, output_response=False)
    optimizer_llm = OpenAILLM(config=optimizer_config)
    benchmark = MathSplits()
    program = CustomProgram(model=executor_llm)
    registry = MiproRegistry()
    registry.track(program, 'prompt', input_names=['problem'], output_names=['solution'])
    optimizer = MiproOptimizer(registry=registry, program=program, optimizer_llm=optimizer_llm, max_bootstrapped_demos=4, max_labeled_demos=4, num_threads=20, eval_rounds=1, auto='medium', save_path='examples/output/mipro/math_plug_and_play')
    logger.info('Optimizing program...')
    optimizer.optimize(dataset=benchmark)
    optimizer.restore_best_program()
    logger.info('Evaluating program on test set...')
    with suppress_logger_info():
        results = optimizer.evaluate(dataset=benchmark, eval_mode='test')
    logger.info(f'Evaluation metrics (after optimization): {results}')

def main():
    openai_config = OpenAILLMConfig(model='gpt-4o-mini', openai_key=OPENAI_API_KEY, stream=True, output_response=False)
    executor_llm = OpenAILLM(config=openai_config)
    optimizer_config = OpenAILLMConfig(model='gpt-4o', openai_key=OPENAI_API_KEY, stream=True, output_response=False)
    optimizer_llm = OpenAILLM(config=optimizer_config)
    benchmark = MathSplits()
    workflow_graph: SequentialWorkFlowGraph = SequentialWorkFlowGraph.from_dict(math_graph_data)
    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)
    evaluator = Evaluator(llm=executor_llm, agent_manager=agent_manager, collate_func=collate_func, num_workers=20, verbose=True)
    optimizer = WorkFlowMiproOptimizer(graph=workflow_graph, evaluator=evaluator, optimizer_llm=optimizer_llm, max_bootstrapped_demos=4, max_labeled_demos=4, eval_rounds=1, auto='medium', save_path='examples/output/mipro/math_mipro')
    logger.info('Optimizing workflow...')
    optimizer.optimize(dataset=benchmark)
    from pdb import set_trace
    set_trace()
    optimizer.restore_best_program()
    logger.info('Evaluating program on test set...')
    with suppress_logger_info():
        results = optimizer.evaluate(dataset=benchmark, eval_mode='test')
    logger.info(f'Evaluation metrics (after optimization): {results}')

def main():
    executor_config = OpenAILLMConfig(model='gpt-4o-mini')
    executor_llm = OpenAILLM(config=executor_config)
    optimizer_config = OpenAILLMConfig(model='gpt-4o')
    optimizer_llm = OpenAILLM(config=optimizer_config)
    benchmark = MBPPSplits()
    workflow_graph = SequentialWorkFlowGraph.from_dict(mbpp_graph_data)
    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(workflow_graph, executor_llm.config)
    evaluator = Evaluator(llm=executor_llm, agent_manager=agent_manager, collate_func=collate_func, num_workers=20, verbose=True)
    textgrad_optimizer = TextGradOptimizer(graph=workflow_graph, optimize_mode='system_prompt', executor_llm=executor_llm, optimizer_llm=optimizer_llm, batch_size=3, max_steps=20, evaluator=evaluator, eval_every_n_steps=1, eval_rounds=1, save_interval=None, save_path='./', rollback=True, constraints=[])
    logger.info('Evaluating workflow on test set...')
    with suppress_logger_info():
        results = textgrad_optimizer.evaluate(dataset=benchmark, eval_mode='test')
    logger.info('Evaluation metrics (before optimization): ', results)
    logger.info('Optimizing workflow...')
    textgrad_optimizer.optimize(benchmark, seed=8)
    textgrad_optimizer.restore_best_graph()
    logger.info('Evaluating workflow on test set...')
    with suppress_logger_info():
        results = textgrad_optimizer.evaluate(dataset=benchmark, eval_mode='test')
    logger.info(f'Evaluation metrics (after optimization): {results}')

def main():
    executor_config = OpenAILLMConfig(model='gpt-4o-mini')
    executor_llm = OpenAILLM(config=executor_config)
    optimizer_config = OpenAILLMConfig(model='gpt-4o')
    optimizer_llm = OpenAILLM(config=optimizer_config)
    benchmark = MathSplits()
    workflow_graph = SequentialWorkFlowGraph.from_dict(math_graph_data)
    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(workflow_graph, executor_llm.config)
    evaluator = Evaluator(llm=executor_llm, agent_manager=agent_manager, collate_func=collate_func, num_workers=20, verbose=True)
    textgrad_optimizer = TextGradOptimizer(graph=workflow_graph, optimize_mode='all', executor_llm=executor_llm, optimizer_llm=optimizer_llm, batch_size=3, max_steps=20, evaluator=evaluator, eval_every_n_steps=1, eval_rounds=1, save_interval=None, save_path='./', rollback=True, constraints=[])
    logger.info('Evaluating workflow on test set...')
    with suppress_logger_info():
        results = textgrad_optimizer.evaluate(dataset=benchmark, eval_mode='test')
    logger.info(f'Evaluation metrics (before optimization): {results}')
    logger.info('Optimizing workflow...')
    textgrad_optimizer.optimize(benchmark, seed=8)
    textgrad_optimizer.restore_best_graph()
    logger.info('Evaluating workflow on test set...')
    with suppress_logger_info():
        results = textgrad_optimizer.evaluate(dataset=benchmark, eval_mode='test')
    logger.info(f'Evaluation metrics (after optimization): {results}')

def main():
    executor_config = OpenAILLMConfig(model='gpt-4o-mini')
    executor_llm = OpenAILLM(config=executor_config)
    optimizer_config = OpenAILLMConfig(model='gpt-4o')
    optimizer_llm = OpenAILLM(config=optimizer_config)
    benchmark = HotPotQASplits()
    workflow_graph = SequentialWorkFlowGraph.from_dict(hotpotqa_graph_data)
    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(workflow_graph, executor_llm.config)
    evaluator = Evaluator(llm=executor_llm, agent_manager=agent_manager, collate_func=collate_func, num_workers=20, verbose=True)
    textgrad_optimizer = TextGradOptimizer(graph=workflow_graph, optimize_mode='all', executor_llm=executor_llm, optimizer_llm=optimizer_llm, batch_size=3, max_steps=20, evaluator=evaluator, eval_every_n_steps=1, eval_rounds=1, save_interval=None, save_path='./', rollback=True, constraints=[])
    logger.info('Evaluating workflow on test set...')
    with suppress_logger_info():
        results = textgrad_optimizer.evaluate(dataset=benchmark, eval_mode='test')
    logger.info(f'Evaluation metrics (before optimization): {results}')
    logger.info('Optimizing workflow...')
    textgrad_optimizer.optimize(benchmark, seed=8)
    textgrad_optimizer.restore_best_graph()
    logger.info('Evaluating workflow on test set...')
    with suppress_logger_info():
        results = textgrad_optimizer.evaluate(dataset=benchmark, eval_mode='test')
    logger.info(f'Evaluation metrics (after optimization): {results}')

