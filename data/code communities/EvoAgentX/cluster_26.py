# Cluster 26

def eval_candidate_program(batch_size: int, evalset: list, candidate_program: Any, evaluator: Callable, rng=None, return_all_scores: bool=False) -> Union[float, Tuple[float, List[float]]]:
    try:
        if batch_size >= len(evalset):
            return evaluator(program=candidate_program, evalset=evalset, return_all_scores=return_all_scores)
        else:
            return evaluator(program=candidate_program, evalset=create_minibatch(evalset, batch_size, rng), return_all_scores=return_all_scores)
    except Exception as e:
        logger.error(f'An exception occurred during evaluation: {str(e)}', exc_info=True)
        if return_all_scores:
            return (0.0, [0.0] * len(evalset))
        return 0.0

def evaluator(cfg, result) -> float:
    return result['score']

