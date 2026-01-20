# Cluster 4

def get_expected_answer(problem_id: int) -> Optional[str]:
    """Get the expected answer for a problem"""
    problem = get_problem_by_id(problem_id)
    return problem['expected_answer'] if problem else None

def get_problem_by_id(problem_id: int) -> Optional[Dict[str, Any]]:
    """Get problem data by ID"""
    return next((p for p in IMO_2025_PROBLEMS if p['id'] == problem_id), None)

def get_answer_type(problem_id: int) -> Optional[str]:
    """Get the answer type for a problem"""
    problem = get_problem_by_id(problem_id)
    return problem['answer_type'] if problem else None

