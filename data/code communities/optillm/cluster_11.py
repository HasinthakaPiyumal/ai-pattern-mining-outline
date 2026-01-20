# Cluster 11

def extract_answer(response: str) -> Optional[int]:
    """
    Extract the numerical answer from a math solution response using unified extraction.
    AIME problems expect integer answers between 0 and 999.
    """
    if not response:
        return None
    extracted_answer = unified_extract_answer(response, problem_type='aime', problem_id=None)
    if extracted_answer is None:
        return None
    if isinstance(extracted_answer, list):
        for item in extracted_answer:
            if isinstance(item, (int, float)):
                answer = int(item)
                if 0 <= answer <= 999:
                    return answer
            elif isinstance(item, str) and item.isdigit():
                answer = int(item)
                if 0 <= answer <= 999:
                    return answer
        return None
    if isinstance(extracted_answer, (int, float)):
        answer = int(extracted_answer)
        if 0 <= answer <= 999:
            return answer
    elif isinstance(extracted_answer, str) and extracted_answer.isdigit():
        answer = int(extracted_answer)
        if 0 <= answer <= 999:
            return answer
    return None

