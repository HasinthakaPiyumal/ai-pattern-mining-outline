# Cluster 42

def evaluate_score(args) -> list[bool]:
    gs, (c, i, o) = args
    execution_results = []
    for g in gs:
        if i in g:
            pass
        else:
            code_to_execute = f'{BASE_IMPORTS}\n{c}\nassert {o} == {g}'
            execution_results.append(check_execution_correctness(code_to_execute, 3))
    if len(execution_results) == 0:
        execution_results = [False] * len(gs)
    return execution_results

def check_execution_correctness(check_program, timeout=3):
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """
    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=unsafe_execute, args=(check_program, result, timeout))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        result.append('timed out')
    return result[0] == 'passed'

