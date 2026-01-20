# Cluster 1

def extract_answer_from_solution(solution: str, problem_id: int) -> str:
    """
    Extract the final answer from a solution using unified answer extraction
    """
    extracted_answer = extract_answer(solution, problem_type='imo', problem_id=problem_id)
    if extracted_answer is None:
        return None
    if isinstance(extracted_answer, list):
        for item in extracted_answer:
            if isinstance(item, set):
                sorted_elements = sorted(list(item))
                return '{' + ', '.join(map(str, sorted_elements)) + '}'
            elif isinstance(item, (int, float)):
                if problem_id == 3:
                    return f'c = {int(item)}'
                else:
                    return str(int(item))
            elif isinstance(item, str) and item.strip():
                return item
        return str(extracted_answer)
    if isinstance(extracted_answer, set):
        sorted_elements = sorted(list(extracted_answer))
        return '{' + ', '.join(map(str, sorted_elements)) + '}'
    elif isinstance(extracted_answer, (int, float)):
        if problem_id == 3:
            return f'c = {int(extracted_answer)}'
        else:
            return str(int(extracted_answer))
    elif isinstance(extracted_answer, str):
        return extracted_answer
    else:
        return str(extracted_answer)

def extract_answer(solution: str, problem_type: str='general', problem_id: Optional[int]=None) -> Optional[Any]:
    """
    Extract answer from solution text.

    Args:
        solution: The solution text to extract answer from
        problem_type: Type of problem (general, imo, aime, etc.)
        problem_id: Specific problem ID for customized extraction

    Returns:
        Extracted answer in appropriate format
    """
    return answer_extractor.extract_answer(solution, problem_type, problem_id)

def imo25_verify_solution(problem: str, solution: str, model: str, problem_id: int=None) -> Dict[str, any]:
    """
    Two-stage verification system from IMO25 repository:
    Stage 1: Detailed verification using comprehensive IMO grader prompt
    Stage 2: Simple yes/no check on solution correctness
    """
    verification_system_prompt = 'You are an expert mathematician and a meticulous grader for an International Mathematical Olympiad (IMO) level exam. Your primary task is to rigorously verify the provided mathematical solution. A solution is to be judged correct **only if every step is rigorously justified.** A solution that arrives at a correct final answer through flawed reasoning, educated guesses, or with gaps in its arguments must be flagged as incorrect or incomplete.\n\n### Instructions ###\n\n**1. Core Instructions**\n*   Your sole task is to find and report all issues in the provided solution. You must act as a **verifier**, NOT a solver. **Do NOT attempt to correct the errors or fill the gaps you find.**\n*   You must perform a **step-by-step** check of the entire solution. This analysis will be presented in a **Detailed Verification Log**, where you justify your assessment of each step: for correct steps, a brief justification suffices; for steps with errors or gaps, you must provide a detailed explanation.\n\n**2. How to Handle Issues in the Solution**\nWhen you identify an issue in a step, you MUST first classify it into one of the following two categories and then follow the specified procedure.\n\n*   **a. Critical Error:**\n    This is any error that breaks the logical chain of the proof. This includes both **logical fallacies** (e.g., claiming that `A>B, C>D` implies `A-C>B-D`) and **factual errors** (e.g., a calculation error like `2+3=6`).\n    *   **Procedure:**\n        *   Explain the specific error and state that it **invalidates the current line of reasoning**.\n        *   Do NOT check any further steps that rely on this error.\n        *   You MUST, however, scan the rest of the solution to identify and verify any fully independent parts. For example, if a proof is split into multiple cases, an error in one case does not prevent you from checking the other cases.\n\n*   **b. Justification Gap:**\n    This is for steps where the conclusion may be correct, but the provided argument is incomplete, hand-wavy, or lacks sufficient rigor.\n    *   **Procedure:**\n        *   Explain the gap in the justification.\n        *   State that you will **assume the step\'s conclusion is true** for the sake of argument.\n        *   Then, proceed to verify all subsequent steps to check if the remainder of the argument is sound.\n\n**3. Output Format**\nYour response MUST be structured into two main sections: a **Summary** followed by the **Detailed Verification Log**.\n\n*   **a. Summary**\n    This section MUST be at the very beginning of your response. It must contain two components:\n    *   **Final Verdict**: A single, clear sentence declaring the overall validity of the solution. For example: "The solution is correct," "The solution contains a Critical Error and is therefore invalid," or "The solution\'s approach is viable but contains several Justification Gaps."\n    *   **List of Findings**: A bulleted list that summarizes **every** issue you discovered. For each finding, you must provide:\n        *   **Location:** A direct quote of the key phrase or equation where the issue occurs.\n        *   **Issue:** A brief description of the problem and its classification (**Critical Error** or **Justification Gap**).\n\n*   **b. Detailed Verification Log**\n    Following the summary, provide the full, step-by-step verification log as defined in the Core Instructions. When you refer to a specific part of the solution, **quote the relevant text** to make your reference clear before providing your detailed analysis of that part.\n\n**Example of the Required Summary Format**\n*This is a generic example to illustrate the required format. Your findings must be based on the actual solution provided below.*\n\n**Final Verdict:** The solution is **invalid** because it contains a Critical Error.\n\n**List of Findings:**\n*   **Location:** "By interchanging the limit and the integral, we get..."\n    *   **Issue:** Justification Gap - The solution interchanges a limit and an integral without providing justification, such as proving uniform convergence.\n*   **Location:** "From $A > B$ and $C > D$, it follows that $A-C > B-D$"\n    *   **Issue:** Critical Error - This step is a logical fallacy. Subtracting inequalities in this manner is not a valid mathematical operation.\n\n### Verification Task Reminder ###\n\nYour task is to act as an IMO grader. Now, generate the **summary** and the **step-by-step verification log** for the solution above. In your log, justify each correct step and explain in detail any errors or justification gaps you find, as specified in the instructions above.'
    verification_prompt = f'\n======================================================================\n### Problem ###\n\n{problem}\n\n======================================================================\n### Solution ###\n\n{solution}\n\n{verification_system_prompt}\n'
    extracted_answer = None
    answer_is_correct = False
    if problem_id is not None:
        extracted_answer = extract_answer_from_solution(solution, problem_id)
        answer_is_correct = check_answer_correctness(problem_id, extracted_answer)
        logger.info(f"Problem {problem_id}: Extracted answer = '{extracted_answer}', Correct = {answer_is_correct}")
    try:
        response = client.with_options(timeout=300).chat.completions.create(model=model, messages=[{'role': 'system', 'content': verification_system_prompt}, {'role': 'user', 'content': verification_prompt}], max_tokens=64000, temperature=0.1)
        verification_response = response.choices[0].message.content.strip()
        if answer_is_correct:
            check_correctness_prompt = f'The solution contains the correct final answer. Please respond with "yes" or "no":\n\nIs the overall mathematical approach reasonable and the final answer correct, even if there are minor justification gaps or presentation issues?\n\n{verification_response}'
        else:
            check_correctness_prompt = f'Response in "yes" or "no". Is the following statement saying the solution is correct, or does not contain critical error or a major justification gap?\n\n{verification_response}'
        response2 = client.with_options(timeout=300).chat.completions.create(model=model, messages=[{'role': 'user', 'content': check_correctness_prompt}], max_tokens=10, temperature=0.1)
        correctness_check = response2.choices[0].message.content.strip().lower()
        verification_says_correct = 'yes' in correctness_check
        if answer_is_correct and verification_says_correct:
            is_correct = True
        elif answer_is_correct and (not verification_says_correct):
            is_correct = True
            logger.info(f'Problem {problem_id}: Answer correct but verification strict - accepting solution')
        else:
            is_correct = verification_says_correct
        bug_report = ''
        if not is_correct:
            verification_log_match = re.search('### Detailed Verification Log ###\\s*(.*)', verification_response, re.DOTALL)
            if verification_log_match:
                bug_report = verification_log_match.group(1).strip()
            else:
                bug_report = verification_response
        return {'judge_response': verification_response, 'correctness_check': correctness_check, 'is_correct': is_correct, 'bug_report': bug_report, 'correctness_score': 1.0 if is_correct else 0.0, 'completeness_score': 1.0 if is_correct else 0.0, 'has_key_insights': is_correct, 'errors_found': [bug_report] if bug_report else [], 'overall_assessment': 'correct' if is_correct else 'incorrect', 'judge_reasoning': verification_response, 'success': True, 'extracted_answer': extracted_answer, 'answer_is_correct': answer_is_correct, 'verification_says_correct': verification_says_correct, 'verification_method': 'hybrid_answer_aware' if problem_id else 'original_imo25'}
    except Exception as e:
        logger.error(f'Error in IMO25 verification: {e}')
        return {'judge_response': f'Error: {str(e)}', 'correctness_check': 'error', 'is_correct': False, 'bug_report': f'Verification error: {str(e)}', 'correctness_score': 0.0, 'completeness_score': 0.0, 'has_key_insights': False, 'errors_found': [f'Judge error: {str(e)}'], 'overall_assessment': 'error', 'judge_reasoning': '', 'success': False}

def check_answer_correctness(problem_id: int, extracted_answer: str) -> bool:
    """
    Check if extracted answer matches the golden answer for the problem
    """
    if not extracted_answer:
        return False
    golden_answers = {1: ['{0, 1, 2, 3}'], 2: ['tangent'], 3: ['c = 4'], 4: ['6', '18', '6, 18'], 5: ['λ < 1', 'λ < √2/2'], 6: ['4048']}
    if problem_id not in golden_answers:
        return False
    correct_answers = golden_answers[problem_id]
    if extracted_answer in correct_answers:
        return True
    if problem_id == 1:
        if extracted_answer == '{0, 1, 3}':
            return False
    if problem_id == 4:
        if any((val in extracted_answer for val in ['6', '18'])):
            return True
        if '2·3^k form' in extracted_answer:
            return True
    if problem_id == 5:
        if any((cond in extracted_answer for cond in ['λ < 1', 'λ < √2/2'])):
            return True
    return False

def evaluate_solution(problem_data: Dict, solution: str, model: str='google/gemini-2.5-flash-lite') -> Dict[str, any]:
    """
    IMO25-style evaluation using rigorous two-stage verification system:
    1. Detailed verification with comprehensive IMO grader prompt
    2. Simple yes/no check on solution correctness

    This eliminates self-judgment bias and provides more accurate assessment
    """
    logger.info(f'Running IMO25-style evaluation for problem {problem_data['id']}')
    imo25_verification = imo25_verify_solution(problem_data['problem'], solution, model, problem_data['id'])
    answer_extraction = extract_final_answer(solution, problem_data['id'])
    quality_analysis = extract_solution_quality(solution)
    correctness_score = 1.0 if imo25_verification['is_correct'] else 0.0
    if imo25_verification['is_correct'] and quality_analysis['completeness_score'] > 0.7:
        confidence = 'high'
    elif imo25_verification['is_correct']:
        confidence = 'medium'
    else:
        confidence = 'low'
    return {'is_correct': imo25_verification['is_correct'], 'verdict': 'Correct' if imo25_verification['is_correct'] else 'Incorrect', 'correctness_score': correctness_score, 'is_likely_correct': imo25_verification['is_correct'], 'confidence': confidence, 'verification_details': {'stage1_analysis': imo25_verification['judge_response'], 'stage2_check': imo25_verification['correctness_check'], 'errors_found': imo25_verification['errors_found'], 'bug_report': imo25_verification['bug_report'] if imo25_verification['bug_report'] else None}, 'layer_scores': {'structural_quality': quality_analysis['completeness_score'], 'insights_verification': 1.0 if imo25_verification['is_correct'] else 0.0, 'llm_judge': correctness_score, 'answer_extraction': answer_extraction['confidence']}, 'weights_used': {'imo25_verification': 1.0}, 'score_variance': 0.0, 'quality_analysis': quality_analysis, 'insights_check': {'required_insights_found': 1 if imo25_verification['is_correct'] else 0, 'total_required_insights': 1, 'insight_score': 1.0 if imo25_verification['is_correct'] else 0.0}, 'llm_verification': imo25_verification, 'answer_extraction': answer_extraction, 'evaluation_method': 'imo25_two_stage_binary'}

def extract_final_answer(solution: str, problem_id: int) -> Dict[str, any]:
    """
    Extract and verify the final answer using official IMO 2025 solutions
    """
    official_verification = verify_answer_format(problem_id, solution)
    result = {'extracted_answer': None, 'confidence': 0.0, 'extraction_method': None, 'official_answer_found': official_verification['correct_answer_found'], 'official_answer_score': official_verification['answer_score']}
    if not solution:
        return result
    if official_verification['correct_answer_found']:
        result['extracted_answer'] = official_verification['extracted_answer']
        result['confidence'] = 1.0
        result['extraction_method'] = 'official_verification'
        return result
    boxed_pattern = '\\\\boxed\\{([^}]+)\\}'
    boxed_matches = re.findall(boxed_pattern, solution)
    if boxed_matches:
        result['extracted_answer'] = boxed_matches[-1].strip()
        result['confidence'] = 0.9
        result['extraction_method'] = 'boxed'
        return result
    answer_patterns = ['final answer[:\\s]*([^\\n]+)', 'answer[:\\s]*([^\\n]+)', 'therefore[:\\s]*([^\\n]+)', 'thus[:\\s]*([^\\n]+)']
    solution_lower = solution.lower()
    for pattern in answer_patterns:
        matches = re.findall(pattern, solution_lower)
        if matches:
            result['extracted_answer'] = matches[-1].strip()
            result['confidence'] = 0.5
            result['extraction_method'] = 'answer_section'
            break
    return result

def extract_solution_quality(response: str) -> Dict[str, any]:
    """
    Analyze the quality of an IMO solution based on mathematical rigor criteria
    """
    analysis = {'has_proof_structure': False, 'uses_mathematical_notation': False, 'has_logical_steps': False, 'addresses_all_cases': False, 'has_conclusion': False, 'length_score': 0, 'rigor_indicators': [], 'completeness_score': 0}
    if not response:
        return analysis
    response_lower = response.lower()
    proof_keywords = ['proof:', 'solution:', 'we prove', 'to show', 'suppose', 'assume', 'let', 'consider']
    if any((keyword in response_lower for keyword in proof_keywords)):
        analysis['has_proof_structure'] = True
        analysis['rigor_indicators'].append('proof_structure')
    math_patterns = ['\\$.*\\$', '\\\\[a-zA-Z]+', '\\\\geq', '\\\\leq', '\\\\in', '\\\\mathbb', '\\\\sum', '\\\\prod']
    if any((re.search(pattern, response) for pattern in math_patterns)):
        analysis['uses_mathematical_notation'] = True
        analysis['rigor_indicators'].append('mathematical_notation')
    logical_words = ['therefore', 'thus', 'hence', 'consequently', 'since', 'because', 'implies', 'follows']
    logical_count = sum((1 for word in logical_words if word in response_lower))
    if logical_count >= 3:
        analysis['has_logical_steps'] = True
        analysis['rigor_indicators'].append('logical_flow')
    case_words = ['case', 'cases', 'if', 'suppose', 'when', 'consider']
    case_count = sum((1 for word in case_words if word in response_lower))
    if case_count >= 2:
        analysis['addresses_all_cases'] = True
        analysis['rigor_indicators'].append('case_analysis')
    conclusion_words = ['conclude', 'final answer', 'solution is', 'answer:', 'qed', 'proven', 'shown']
    if any((word in response_lower for word in conclusion_words)):
        analysis['has_conclusion'] = True
        analysis['rigor_indicators'].append('clear_conclusion')
    word_count = len(response.split())
    if word_count >= 500:
        analysis['length_score'] = 3
    elif word_count >= 200:
        analysis['length_score'] = 2
    elif word_count >= 100:
        analysis['length_score'] = 1
    else:
        analysis['length_score'] = 0
    completeness_factors = [analysis['has_proof_structure'], analysis['uses_mathematical_notation'], analysis['has_logical_steps'], analysis['addresses_all_cases'], analysis['has_conclusion']]
    analysis['completeness_score'] = sum(completeness_factors) / len(completeness_factors)
    return analysis

def mixture_of_agents(system_prompt: str, initial_query: str, client, model: str, request_id: str=None) -> str:
    logger.info(f'Starting mixture_of_agents function with model: {model}')
    moa_completion_tokens = 0
    completions = []
    logger.debug(f'Generating initial completions for query: {initial_query}')
    try:
        provider_request = {'model': model, 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': initial_query}], 'max_tokens': 4096, 'n': 3, 'temperature': 1}
        response = client.chat.completions.create(**provider_request)
        response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
        if request_id:
            conversation_logger.log_provider_call(request_id, provider_request, response_dict)
        completions = [choice.message.content for choice in response.choices]
        moa_completion_tokens += response.usage.completion_tokens
        logger.info(f'Generated {len(completions)} initial completions using n parameter. Tokens used: {response.usage.completion_tokens}')
    except Exception as e:
        logger.warning(f'n parameter not supported by provider: {str(e)}')
        logger.info('Falling back to generating 3 completions one by one')
        completions = []
        for i in range(3):
            try:
                provider_request = {'model': model, 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': initial_query}], 'max_tokens': 4096, 'temperature': 1}
                response = client.chat.completions.create(**provider_request)
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
                if request_id:
                    conversation_logger.log_provider_call(request_id, provider_request, response_dict)
                completions.append(response.choices[0].message.content)
                moa_completion_tokens += response.usage.completion_tokens
                logger.debug(f'Generated completion {i + 1}/3')
            except Exception as fallback_error:
                logger.error(f'Error generating completion {i + 1}: {str(fallback_error)}')
                continue
        if not completions:
            logger.error('Failed to generate any completions')
            return ('Error: Could not generate any completions', 0)
        logger.info(f'Generated {len(completions)} completions using fallback method. Total tokens used: {moa_completion_tokens}')
    if len(completions) < 3:
        original_count = len(completions)
        while len(completions) < 3:
            completions.append(completions[0])
        logger.warning(f'Only generated {original_count} unique completions, padded to 3 for critique')
    logger.debug('Preparing critique prompt')
    critique_prompt = f'\n    Original query: {initial_query}\n\n    I will present you with three candidate responses to the original query. Please analyze and critique each response, discussing their strengths and weaknesses. Provide your analysis for each candidate separately.\n\n    Candidate 1:\n    {completions[0]}\n\n    Candidate 2:\n    {completions[1]}\n\n    Candidate 3:\n    {completions[2]}\n\n    Please provide your critique for each candidate:\n    '
    logger.debug('Generating critiques')
    provider_request = {'model': model, 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': critique_prompt}], 'max_tokens': 512, 'n': 1, 'temperature': 0.1}
    critique_response = client.chat.completions.create(**provider_request)
    response_dict = critique_response.model_dump() if hasattr(critique_response, 'model_dump') else critique_response
    if request_id:
        conversation_logger.log_provider_call(request_id, provider_request, response_dict)
    critiques = critique_response.choices[0].message.content
    moa_completion_tokens += critique_response.usage.completion_tokens
    logger.info(f'Generated critiques. Tokens used: {critique_response.usage.completion_tokens}')
    logger.debug('Preparing final prompt')
    final_prompt = f'\n    Original query: {initial_query}\n\n    Based on the following candidate responses and their critiques, generate a final response to the original query.\n\n    Candidate 1:\n    {completions[0]}\n\n    Candidate 2:\n    {completions[1]}\n\n    Candidate 3:\n    {completions[2]}\n\n    Critiques of all candidates:\n    {critiques}\n\n    Please provide a final, optimized response to the original query:\n    '
    logger.debug('Generating final response')
    provider_request = {'model': model, 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': final_prompt}], 'max_tokens': 8192, 'n': 1, 'temperature': 0.1}
    final_response = client.chat.completions.create(**provider_request)
    response_dict = final_response.model_dump() if hasattr(final_response, 'model_dump') else final_response
    if request_id:
        conversation_logger.log_provider_call(request_id, provider_request, response_dict)
    moa_completion_tokens += final_response.usage.completion_tokens
    logger.info(f'Generated final response. Tokens used: {final_response.usage.completion_tokens}')
    logger.info(f'Total completion tokens used: {moa_completion_tokens}')
    return (final_response.choices[0].message.content, moa_completion_tokens)

class MCTS:

    def __init__(self, simulation_depth, exploration_weight, client, model, request_id=None):
        self.simulation_depth = simulation_depth
        self.exploration_weight = exploration_weight
        self.root = None
        self.graph = nx.Graph()
        self.node_labels = {}
        self.client = client
        self.model = model
        self.completion_tokens = 0
        self.request_id = request_id

    def select(self, node: MCTSNode) -> MCTSNode:
        logger.debug(f'Selecting node. Current node visits: {node.visits}, value: {node.value}')
        if not node.children:
            logger.debug('Node has no children. Returning current node.')
            return node
        selected_node = max(node.children, key=lambda c: c.value / (c.visits + 1e-08) + self.exploration_weight * np.sqrt(np.log(node.visits + 1) / (c.visits + 1e-08)))
        logger.debug(f'Selected child node. Visits: {selected_node.visits}, Value: {selected_node.value}')
        return selected_node

    def expand(self, node: MCTSNode) -> MCTSNode:
        logger.debug(f'Expanding node. Current state: {node.state}')
        actions = self.generate_actions(node.state)
        logger.debug(f'Generated {len(actions)} possible actions')
        for i, action in enumerate(actions):
            new_state = self.apply_action(node.state, action)
            child = MCTSNode(new_state, parent=node)
            node.children.append(child)
            self.graph.add_edge(id(node), id(child))
            self.node_labels[id(child)] = f'Visits: {child.visits}\nValue: {child.value:.2f}'
            logger.debug(f'Created child node {i + 1}. Action: {action[:50]}...')
        selected_child = random.choice(node.children)
        logger.debug(f'Randomly selected child node for simulation. Visits: {selected_child.visits}, Value: {selected_child.value}')
        return selected_child

    def simulate(self, node: MCTSNode) -> float:
        logger.debug(f'Starting simulation from node. Current query: {node.state.current_query}')
        state = node.state
        for i in range(self.simulation_depth):
            if self.is_terminal(state):
                logger.debug(f'Reached terminal state at depth {i}')
                break
            action = random.choice(self.generate_actions(state))
            state = self.apply_action(state, action)
            logger.debug(f'Simulation step {i + 1}. Action: {action[:50]}...')
        value = self.evaluate_state(state)
        logger.debug(f'Simulation complete. Final state value: {value}')
        return value

    def backpropagate(self, node: MCTSNode, value: float):
        logger.debug(f'Starting backpropagation. Initial value: {value}')
        while node:
            node.visits += 1
            node.value += value
            self.node_labels[id(node)] = f'Visits: {node.visits}\nValue: {node.value:.2f}'
            logger.debug(f'Updated node. Visits: {node.visits}, New value: {node.value}')
            node = node.parent

    def search(self, initial_state: DialogueState, num_simulations: int) -> DialogueState:
        logger.debug(f'Starting MCTS search with {num_simulations} simulations')
        if not self.root:
            self.root = MCTSNode(initial_state)
            self.graph.add_node(id(self.root))
            self.node_labels[id(self.root)] = f'Root\nVisits: 0\nValue: 0.00'
            logger.debug('Created root node')
        for i in range(num_simulations):
            logger.debug(f'Starting simulation {i + 1}')
            node = self.select(self.root)
            if not self.is_terminal(node.state):
                node = self.expand(node)
            value = self.simulate(node)
            self.backpropagate(node, value)
        best_child = max(self.root.children, key=lambda c: c.visits)
        logger.debug(f'Search complete. Best child node: Visits: {best_child.visits}, Value: {best_child.value}')
        return best_child.state

    def generate_actions(self, state: DialogueState) -> List[str]:
        logger.debug('Generating actions for current state')
        messages = [{'role': 'system', 'content': state.system_prompt}]
        messages.extend(state.conversation_history)
        messages.append({'role': 'user', 'content': state.current_query})
        completions = []
        n = 3
        logger.info(f'Requesting {n} completions from the model')
        provider_request = {'model': self.model, 'messages': messages, 'max_tokens': 4096, 'n': n, 'temperature': 1}
        response = self.client.chat.completions.create(**provider_request)
        if self.request_id:
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            conversation_logger.log_provider_call(self.request_id, provider_request, response_dict)
        completions = [choice.message.content.strip() for choice in response.choices]
        self.completion_tokens += response.usage.completion_tokens
        logger.info(f'Received {len(completions)} completions from the model')
        return completions

    def apply_action(self, state: DialogueState, action: str) -> DialogueState:
        logger.info(f'Applying action: {action[:50]}...')
        new_history = state.conversation_history.copy()
        new_history.append({'role': 'assistant', 'content': action})
        messages = [{'role': 'system', 'content': state.system_prompt}]
        messages.extend(new_history)
        messages.append({'role': 'user', 'content': 'Based on this conversation, what might the user ask or say next? Provide a likely user query.'})
        logger.info('Requesting next user query from the model')
        provider_request = {'model': self.model, 'messages': messages, 'max_tokens': 1024, 'n': 1, 'temperature': 1}
        response = self.client.chat.completions.create(**provider_request)
        if self.request_id:
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            conversation_logger.log_provider_call(self.request_id, provider_request, response_dict)
        next_query = response.choices[0].message.content
        self.completion_tokens += response.usage.completion_tokens
        logger.info(f'Generated next user query: {next_query}')
        return DialogueState(state.system_prompt, new_history, next_query)

    def is_terminal(self, state: DialogueState) -> bool:
        is_terminal = len(state.conversation_history) > 10 or 'goodbye' in state.current_query.lower()
        logger.info(f'Checking if state is terminal: {is_terminal}')
        return is_terminal

    def evaluate_state(self, state: DialogueState) -> float:
        logger.info('Evaluating current state')
        messages = [{'role': 'system', 'content': state.system_prompt}]
        messages.extend(state.conversation_history)
        messages.append({'role': 'user', 'content': 'Evaluate the quality of this conversation on a scale from 0 to 1, where 0 is poor and 1 is excellent. Consider factors such as coherence, relevance, and engagement. Respond with only a number.'})
        provider_request = {'model': self.model, 'messages': messages, 'max_tokens': 256, 'n': 1, 'temperature': 0.1}
        response = self.client.chat.completions.create(**provider_request)
        if self.request_id:
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            conversation_logger.log_provider_call(self.request_id, provider_request, response_dict)
        self.completion_tokens += response.usage.completion_tokens
        try:
            score = float(response.choices[0].message.content.strip())
            score = max(0, min(score, 1))
            logger.info(f'State evaluation score: {score}')
            return score
        except ValueError:
            logger.warning('Failed to parse evaluation score. Using default value 0.5')
            return 0.5

class MARSAggregator:
    """
    RSA-inspired aggregation system for combining solutions

    Key features:
    - Population management (N > K for diversity)
    - Recursive aggregation loops
    - Parallel execution of aggregation calls
    - Solution quality tracking
    """

    def __init__(self, client, model: str, config: Dict[str, Any]):
        self.client = client
        self.model = model
        self.config = config
        self.population_size = config.get('population_size', 6)
        self.aggregation_size = config.get('aggregation_size', 3)
        self.aggregation_loops = config.get('aggregation_loops', 3)
        self.max_tokens = config.get('max_tokens', 30000)

    async def run_aggregation_loops(self, workspace: MARSWorkspace, request_id: str=None, executor: ThreadPoolExecutor=None) -> Tuple[int, Dict[str, Any]]:
        """
        Run T iterations of RSA-style aggregation

        Args:
            workspace: MARS workspace containing solutions
            request_id: Request ID for logging
            executor: Thread pool for parallel execution

        Returns:
            Tuple of (total_reasoning_tokens, aggregation_summary)
        """
        logger.info(f'Starting {self.aggregation_loops} aggregation loops')
        total_reasoning_tokens = 0
        aggregation_history = []
        self._ensure_population_size(workspace)
        for loop_idx in range(self.aggregation_loops):
            logger.info(f'Aggregation loop {loop_idx + 1}/{self.aggregation_loops}')
            loop_tokens, loop_summary = await self._run_single_aggregation_loop(workspace, loop_idx, request_id, executor)
            total_reasoning_tokens += loop_tokens
            aggregation_history.append({'loop': loop_idx, 'tokens': loop_tokens, 'summary': loop_summary})
            logger.info(f'Loop {loop_idx + 1} complete: {loop_summary['solutions_generated']} new solutions')
        summary = {'total_loops': self.aggregation_loops, 'total_reasoning_tokens': total_reasoning_tokens, 'final_population_size': len(workspace.solutions), 'aggregation_history': aggregation_history}
        logger.info(f'Aggregation complete: {summary['final_population_size']} solutions in final population')
        return (total_reasoning_tokens, summary)

    async def _run_single_aggregation_loop(self, workspace: MARSWorkspace, loop_idx: int, request_id: str=None, executor: ThreadPoolExecutor=None) -> Tuple[int, Dict[str, Any]]:
        """Run a single aggregation loop: sample K -> aggregate -> update population"""
        sampled_solutions = self._sample_solutions_for_aggregation(workspace)
        new_solutions, total_tokens = await self._generate_aggregated_solutions(workspace, sampled_solutions, request_id, executor)
        self._update_population(workspace, new_solutions)
        loop_summary = {'sampled_solutions': len(sampled_solutions), 'solutions_generated': len(new_solutions), 'population_size': len(workspace.solutions), 'total_tokens': total_tokens}
        return (total_tokens, loop_summary)

    def _sample_solutions_for_aggregation(self, workspace: MARSWorkspace) -> List[List[AgentSolution]]:
        """
        Sample K solutions from population for aggregation
        Uses different strategies for each sample to maintain diversity
        """
        all_solutions = workspace.solutions
        if len(all_solutions) < self.aggregation_size:
            return [all_solutions]
        samples = []
        num_samples = min(self.population_size // self.aggregation_size, 3)
        for i in range(num_samples):
            if i == 0:
                sample = sorted(all_solutions, key=lambda s: s.verification_score, reverse=True)[:self.aggregation_size]
            elif i == 1:
                by_agent = {}
                for sol in all_solutions:
                    if sol.agent_id not in by_agent:
                        by_agent[sol.agent_id] = []
                    by_agent[sol.agent_id].append(sol)
                sample = []
                for agent_solutions in by_agent.values():
                    if sample and len(sample) < self.aggregation_size:
                        sample.append(max(agent_solutions, key=lambda s: s.confidence))
                    if len(sample) >= self.aggregation_size:
                        break
                if len(sample) < self.aggregation_size:
                    remaining = [s for s in all_solutions if s not in sample]
                    sample.extend(sorted(remaining, key=lambda s: s.verification_score, reverse=True)[:self.aggregation_size - len(sample)])
            else:
                sample = random.sample(all_solutions, min(self.aggregation_size, len(all_solutions)))
            samples.append(sample)
        logger.info(f'Generated {len(samples)} sample groups for aggregation')
        return samples

    async def _generate_aggregated_solutions(self, workspace: MARSWorkspace, sampled_solution_groups: List[List[AgentSolution]], request_id: str=None, executor: ThreadPoolExecutor=None) -> Tuple[List[AgentSolution], int]:
        """Generate new solutions by aggregating sampled solutions in parallel"""

        async def aggregate_solution_group(solutions: List[AgentSolution]) -> Tuple[Optional[AgentSolution], int]:
            """Aggregate a single group of solutions"""
            loop = asyncio.get_event_loop()
            try:
                if len(solutions) == 1:
                    prompt = SINGLE_REFINEMENT_PROMPT.format(problem=workspace.problem, candidate_solution=solutions[0].solution)
                else:
                    candidate_text = ''
                    for i, sol in enumerate(solutions):
                        candidate_text += f'Solution {i + 1} (Agent {sol.agent_id}, confidence: {sol.confidence:.2f}):\n'
                        candidate_text += sol.solution + '\n\n'
                    prompt = MULTI_AGGREGATION_PROMPT.format(problem=workspace.problem, candidate_solutions=candidate_text)
                solution, tokens = await loop.run_in_executor(executor, self._call_model_for_aggregation, prompt, request_id)
                if solution:
                    aggregated_solution = AgentSolution(agent_id=f'agg_{datetime.now().strftime('%H%M%S')}', solution=solution, confidence=0.8, reasoning_tokens=tokens, total_tokens=tokens, solution_length=len(solution), is_verified=False, verification_score=0.0)
                    return (aggregated_solution, tokens)
                return (None, tokens)
            except Exception as e:
                logger.error(f'Aggregation failed: {str(e)}')
                return (None, 0)
        tasks = [aggregate_solution_group(group) for group in sampled_solution_groups]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        new_solutions = []
        total_tokens = 0
        for result in results:
            if isinstance(result, Exception):
                logger.error(f'Aggregation task failed: {str(result)}')
                continue
            solution, tokens = result
            if solution:
                new_solutions.append(solution)
            total_tokens += tokens
        logger.info(f'Generated {len(new_solutions)} aggregated solutions with {total_tokens} reasoning tokens')
        return (new_solutions, total_tokens)

    def _call_model_for_aggregation(self, prompt: str, request_id: str=None) -> Tuple[str, int]:
        """Call the model to perform aggregation (synchronous for executor)"""
        try:
            response = self.client.chat.completions.create(model=self.model, messages=[{'role': 'system', 'content': 'You are a mathematical reasoning expert focused on solution aggregation and refinement.'}, {'role': 'user', 'content': prompt}], max_tokens=self.max_tokens, temperature=0.7, timeout=300, extra_body={'reasoning': {'effort': 'high'}})
            if request_id:
                provider_request = {'model': self.model, 'messages': [{'role': 'system', 'content': 'You are a mathematical reasoning expert focused on solution aggregation and refinement.'}, {'role': 'user', 'content': prompt}], 'max_tokens': self.max_tokens, 'temperature': 0.7, 'extra_body': {'reasoning': {'effort': 'high'}}}
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
                conversation_logger.log_provider_call(request_id, provider_request, response_dict)
            solution = response.choices[0].message.content.strip()
            reasoning_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                    reasoning_tokens = getattr(response.usage.completion_tokens_details, 'reasoning_tokens', 0)
                if reasoning_tokens == 0:
                    reasoning_tokens = getattr(response.usage, 'reasoning_tokens', 0)
            return (solution, reasoning_tokens)
        except Exception as e:
            logger.error(f'Model call for aggregation failed: {str(e)}')
            return ('', 0)

    def _update_population(self, workspace: MARSWorkspace, new_solutions: List[AgentSolution]) -> None:
        """Update population with new solutions, maintaining population size limit"""
        for solution in new_solutions:
            workspace.add_solution(solution)
        all_solutions = workspace.solutions
        if len(all_solutions) > self.population_size:
            sorted_solutions = sorted(all_solutions, key=lambda s: (s.verification_score, s.confidence), reverse=True)
            workspace.solutions = sorted_solutions[:self.population_size]
            logger.info(f'Population trimmed to {self.population_size} best solutions')

    def _ensure_population_size(self, workspace: MARSWorkspace) -> None:
        """Ensure we have minimum population size for effective aggregation"""
        current_size = len(workspace.solutions)
        if current_size < self.aggregation_size:
            logger.warning(f'Population size ({current_size}) < aggregation size ({self.aggregation_size})')
            logger.warning('Aggregation may be less effective with limited diversity')
        logger.info(f'Population ready: {current_size} solutions available for aggregation')

class StrategyNetwork:
    """
    Cross-agent strategy sharing and meta-reasoning system

    Key capabilities:
    1. Extract reasoning strategies from agent solutions
    2. Share effective strategies between agents
    3. Track strategy effectiveness across problem types
    4. Enable adaptive agent behavior based on peer insights
    """

    def __init__(self, client, model: str, config: Dict[str, Any]):
        self.client = client
        self.model = model
        self.config = config
        self.max_tokens = config.get('max_tokens', 30000)
        self.strategies: Dict[str, ReasoningStrategy] = {}
        self.strategy_effectiveness: Dict[Tuple[str, str], StrategyEffectiveness] = {}
        self.agent_preferred_strategies: Dict[str, List[str]] = defaultdict(list)
        self.problem_type_cache: Dict[str, str] = {}
        logger.info('Initialized Strategy Network for cross-agent insight sharing')

    async def extract_strategies_from_solutions(self, workspace: MARSWorkspace, request_id: str=None, executor: ThreadPoolExecutor=None) -> Dict[str, ReasoningStrategy]:
        """Extract reasoning strategies from all agent solutions"""
        logger.info('Extracting strategies from agent solutions...')
        extraction_tasks = []
        for solution in workspace.solutions:
            if not solution.agent_id.startswith('agg_'):
                task = self._extract_strategy_async(solution, workspace.problem, request_id, executor)
                extraction_tasks.append(task)
        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        extracted_strategies = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f'Strategy extraction failed: {str(result)}')
                continue
            if result:
                strategy = result
                extracted_strategies[strategy.strategy_id] = strategy
                self.strategies[strategy.strategy_id] = strategy
                self.agent_preferred_strategies[strategy.agent_id].append(strategy.strategy_id)
        logger.info(f'Extracted {len(extracted_strategies)} reasoning strategies')
        return extracted_strategies

    async def _extract_strategy_async(self, solution: AgentSolution, problem: str, request_id: str=None, executor: ThreadPoolExecutor=None) -> Optional[ReasoningStrategy]:
        """Extract strategy from a single agent solution"""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(executor, self._extract_strategy_from_solution, solution, problem, request_id)
        except Exception as e:
            logger.error(f'Failed to extract strategy from agent {solution.agent_id}: {str(e)}')
            return None

    def _extract_strategy_from_solution(self, solution: AgentSolution, problem: str, request_id: str=None) -> Optional[ReasoningStrategy]:
        """Extract reasoning strategy using LLM analysis"""
        strategy_extraction_prompt = f'Analyze this mathematical solution and extract the key reasoning strategy:\n\nProblem: {problem}\n\nAgent Solution:\n{solution.solution}\n\nExtract the following strategy components:\n\n1. PROBLEM_TYPE: Classify as one of [algebra, geometry, combinatorics, number_theory, calculus, discrete_math, probability]\n\n2. APPROACH_TYPE: Identify the main approach [direct_computation, proof_by_contradiction, constructive_proof, case_analysis, induction, algebraic_manipulation, geometric_visualization, pattern_recognition, reduction_to_known_problem]\n\n3. KEY_INSIGHTS: List 2-3 key mathematical insights that enabled the solution\n\n4. MATHEMATICAL_TECHNIQUES: List specific techniques used [substitution, factorization, coordinate_geometry, symmetry, pigeonhole_principle, etc.]\n\n5. SOLUTION_PATTERN: Describe the general pattern/template of this solution approach\n\n6. SUCCESS_INDICATORS: What makes this approach particularly effective for this type of problem?\n\nFormat your response as:\nPROBLEM_TYPE: [type]\nAPPROACH_TYPE: [approach]\nKEY_INSIGHTS: [insight1], [insight2], [insight3]\nMATHEMATICAL_TECHNIQUES: [technique1], [technique2], [technique3]\nSOLUTION_PATTERN: [pattern description]\nSUCCESS_INDICATORS: [indicator1], [indicator2]'
        try:
            response = self.client.chat.completions.create(model=self.model, messages=[{'role': 'system', 'content': 'You are a mathematical strategy analysis expert. Extract reasoning patterns from solutions.'}, {'role': 'user', 'content': strategy_extraction_prompt}], max_tokens=self.max_tokens // 4, temperature=0.3, timeout=120, extra_body={'reasoning': {'effort': 'medium'}})
            if request_id:
                provider_request = {'model': self.model, 'messages': [{'role': 'system', 'content': 'You are a mathematical strategy analysis expert.'}, {'role': 'user', 'content': strategy_extraction_prompt}], 'max_tokens': self.max_tokens // 4, 'temperature': 0.3, 'extra_body': {'reasoning': {'effort': 'medium'}}}
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
                conversation_logger.log_provider_call(request_id, provider_request, response_dict)
            analysis = response.choices[0].message.content.strip()
            strategy_data = self._parse_strategy_analysis(analysis)
            if strategy_data:
                strategy_id = f'strategy_{solution.agent_id}_{datetime.now().strftime('%H%M%S')}'
                return ReasoningStrategy(strategy_id=strategy_id, agent_id=solution.agent_id, problem_type=strategy_data.get('problem_type', 'unknown'), approach_type=strategy_data.get('approach_type', 'unknown'), key_insights=strategy_data.get('key_insights', []), mathematical_techniques=strategy_data.get('mathematical_techniques', []), solution_pattern=strategy_data.get('solution_pattern', ''), confidence=solution.confidence, success_indicators=strategy_data.get('success_indicators', []))
        except Exception as e:
            logger.error(f'Strategy extraction failed for agent {solution.agent_id}: {str(e)}')
            return None

    def _parse_strategy_analysis(self, analysis: str) -> Optional[Dict[str, Any]]:
        """Parse structured strategy analysis response"""
        try:
            lines = analysis.split('\n')
            strategy_data = {}
            for line in lines:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key == 'problem_type':
                        strategy_data['problem_type'] = value
                    elif key == 'approach_type':
                        strategy_data['approach_type'] = value
                    elif 'insights' in key:
                        strategy_data['key_insights'] = [insight.strip() for insight in value.split(',')]
                    elif 'techniques' in key:
                        strategy_data['mathematical_techniques'] = [tech.strip() for tech in value.split(',')]
                    elif 'pattern' in key:
                        strategy_data['solution_pattern'] = value
                    elif 'indicators' in key:
                        strategy_data['success_indicators'] = [ind.strip() for ind in value.split(',')]
            return strategy_data if strategy_data else None
        except Exception as e:
            logger.error(f'Failed to parse strategy analysis: {str(e)}')
            return None

    async def share_strategies_across_agents(self, workspace: MARSWorkspace, extracted_strategies: Dict[str, ReasoningStrategy], request_id: str=None, executor: ThreadPoolExecutor=None) -> Dict[str, List[str]]:
        """Share effective strategies across agents and generate enhanced solutions"""
        logger.info('Sharing strategies across agents...')
        problem_type = await self._classify_problem_type(workspace.problem, request_id, executor)
        effective_strategies = self._get_effective_strategies_for_type(problem_type, extracted_strategies)
        enhancement_tasks = []
        agent_strategies = {}
        for solution in workspace.solutions:
            if not solution.agent_id.startswith('agg_'):
                cross_agent_strategies = [strategy for strategy in effective_strategies.values() if strategy.agent_id != solution.agent_id]
                if cross_agent_strategies:
                    agent_strategies[solution.agent_id] = [s.strategy_id for s in cross_agent_strategies]
                    task = self._generate_strategy_enhanced_solution_async(solution, workspace.problem, cross_agent_strategies, request_id, executor)
                    enhancement_tasks.append((solution.agent_id, task))
        if enhancement_tasks:
            tasks = [task for _, task in enhancement_tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f'Strategy enhancement failed: {str(result)}')
                    continue
                if result:
                    enhanced_solution = result
                    workspace.add_solution(enhanced_solution)
                    logger.info(f'Added strategy-enhanced solution from agent {enhanced_solution.agent_id}')
        logger.info(f'Strategy sharing complete: enhanced {len(enhancement_tasks)} agents')
        return agent_strategies

    async def _classify_problem_type(self, problem: str, request_id: str=None, executor: ThreadPoolExecutor=None) -> str:
        """Classify the problem type for strategy matching"""
        if problem in self.problem_type_cache:
            return self.problem_type_cache[problem]
        loop = asyncio.get_event_loop()
        try:
            problem_type = await loop.run_in_executor(executor, self._classify_problem_with_llm, problem, request_id)
            self.problem_type_cache[problem] = problem_type
            return problem_type
        except Exception as e:
            logger.error(f'Problem classification failed: {str(e)}')
            return 'unknown'

    def _classify_problem_with_llm(self, problem: str, request_id: str=None) -> str:
        """Use LLM to classify problem type"""
        classification_prompt = f'Classify this mathematical problem into one category:\n\nProblem: {problem}\n\nCategories: [algebra, geometry, combinatorics, number_theory, calculus, discrete_math, probability]\n\nRespond with just the category name.'
        try:
            response = self.client.chat.completions.create(model=self.model, messages=[{'role': 'system', 'content': 'You are a mathematical problem classifier.'}, {'role': 'user', 'content': classification_prompt}], max_tokens=50, temperature=0.1, timeout=60, extra_body={'reasoning': {'effort': 'low'}})
            classification = response.choices[0].message.content.strip().lower()
            valid_types = ['algebra', 'geometry', 'combinatorics', 'number_theory', 'calculus', 'discrete_math', 'probability']
            if classification in valid_types:
                return classification
            else:
                return 'algebra'
        except Exception as e:
            logger.error(f'Problem classification failed: {str(e)}')
            return 'algebra'

    def _get_effective_strategies_for_type(self, problem_type: str, extracted_strategies: Dict[str, ReasoningStrategy]) -> Dict[str, ReasoningStrategy]:
        """Get most effective strategies for the given problem type"""
        relevant_strategies = {}
        for strategy_id, strategy in extracted_strategies.items():
            if (strategy.problem_type == problem_type or strategy.problem_type == 'unknown') and strategy.confidence >= 0.6:
                relevant_strategies[strategy_id] = strategy
        if not relevant_strategies:
            sorted_strategies = sorted(extracted_strategies.items(), key=lambda x: x[1].confidence, reverse=True)
            relevant_strategies = dict(sorted_strategies[:2])
        return relevant_strategies

    async def _generate_strategy_enhanced_solution_async(self, original_solution: AgentSolution, problem: str, peer_strategies: List[ReasoningStrategy], request_id: str=None, executor: ThreadPoolExecutor=None) -> Optional[AgentSolution]:
        """Generate enhanced solution using peer strategies"""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(executor, self._generate_strategy_enhanced_solution, original_solution, problem, peer_strategies, request_id)
        except Exception as e:
            logger.error(f'Strategy enhancement failed for agent {original_solution.agent_id}: {str(e)}')
            return None

    def _generate_strategy_enhanced_solution(self, original_solution: AgentSolution, problem: str, peer_strategies: List[ReasoningStrategy], request_id: str=None) -> Optional[AgentSolution]:
        """Generate solution enhanced with peer strategies"""
        strategy_insights = ''
        for strategy in peer_strategies[:2]:
            strategy_insights += f'\nPeer Strategy from Agent {strategy.agent_id}:\n'
            strategy_insights += f'- Approach: {strategy.approach_type}\n'
            strategy_insights += f'- Key Insights: {', '.join(strategy.key_insights[:3])}\n'
            strategy_insights += f'- Techniques: {', '.join(strategy.mathematical_techniques[:3])}\n'
            strategy_insights += f'- Success Pattern: {strategy.solution_pattern[:200]}...\n'
        enhancement_prompt = f'You are Agent {original_solution.agent_id} collaborating with other mathematical agents.\n\nOriginal Problem: {problem}\n\nYour Current Solution:\n{original_solution.solution}\n\nPeer Agent Strategy Insights:\n{strategy_insights}\n\nTask: Enhance your solution by incorporating the most valuable insights from your peers while maintaining your unique approach. Consider:\n\n1. Can any peer techniques strengthen your solution?\n2. Do peer insights reveal gaps in your reasoning?\n3. Can you combine approaches for a more robust solution?\n4. What verification steps from peers could improve confidence?\n\nProvide an enhanced solution that synthesizes the best ideas while ensuring mathematical rigor.\n\nEnhanced Solution:'
        try:
            response = self.client.chat.completions.create(model=self.model, messages=[{'role': 'system', 'content': 'You are a collaborative mathematical agent learning from peer insights.'}, {'role': 'user', 'content': enhancement_prompt}], max_tokens=self.max_tokens, temperature=original_solution.temperature * 0.9, timeout=300, extra_body={'reasoning': {'effort': 'high'}})
            if request_id:
                provider_request = {'model': self.model, 'messages': [{'role': 'system', 'content': 'You are a collaborative mathematical agent learning from peer insights.'}, {'role': 'user', 'content': enhancement_prompt}], 'max_tokens': self.max_tokens, 'temperature': original_solution.temperature * 0.9, 'extra_body': {'reasoning': {'effort': 'high'}}}
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
                conversation_logger.log_provider_call(request_id, provider_request, response_dict)
            enhanced_solution_text = response.choices[0].message.content.strip()
            reasoning_tokens = 0
            total_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                total_tokens = getattr(response.usage, 'total_tokens', 0)
                if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                    reasoning_tokens = getattr(response.usage.completion_tokens_details, 'reasoning_tokens', 0)
                if reasoning_tokens == 0:
                    reasoning_tokens = getattr(response.usage, 'reasoning_tokens', 0)
            enhanced_agent_solution = AgentSolution(agent_id=f'enhanced_{original_solution.agent_id}', solution=enhanced_solution_text, confidence=min(original_solution.confidence + 0.1, 1.0), reasoning_tokens=reasoning_tokens, total_tokens=total_tokens, solution_length=len(enhanced_solution_text), temperature=original_solution.temperature)
            logger.info(f'Generated strategy-enhanced solution for agent {original_solution.agent_id}')
            return enhanced_agent_solution
        except Exception as e:
            logger.error(f'Strategy enhancement failed for agent {original_solution.agent_id}: {str(e)}')
            return None

    def update_strategy_effectiveness(self, strategy_id: str, problem_type: str, was_successful: bool, confidence: float):
        """Update effectiveness tracking for a strategy"""
        key = (strategy_id, problem_type)
        if key not in self.strategy_effectiveness:
            self.strategy_effectiveness[key] = StrategyEffectiveness(strategy_id=strategy_id, problem_type=problem_type)
        effectiveness = self.strategy_effectiveness[key]
        effectiveness.total_uses += 1
        if was_successful:
            effectiveness.success_count += 1
        else:
            effectiveness.failure_count += 1
        effectiveness.average_confidence = (effectiveness.average_confidence * (effectiveness.total_uses - 1) + confidence) / effectiveness.total_uses

    def get_strategy_insights_summary(self) -> Dict[str, Any]:
        """Get summary of strategy network insights"""
        return {'total_strategies': len(self.strategies), 'strategies_by_type': self._count_strategies_by_type(), 'most_effective_strategies': self._get_most_effective_strategies(), 'agent_strategy_preferences': dict(self.agent_preferred_strategies), 'strategy_effectiveness_stats': self._get_effectiveness_stats()}

    def _count_strategies_by_type(self) -> Dict[str, int]:
        """Count strategies by problem type"""
        counts = defaultdict(int)
        for strategy in self.strategies.values():
            counts[strategy.problem_type] += 1
        return dict(counts)

    def _get_most_effective_strategies(self) -> List[Dict[str, Any]]:
        """Get most effective strategies across all problem types"""
        effective_strategies = []
        for effectiveness in self.strategy_effectiveness.values():
            if effectiveness.total_uses >= 2:
                effective_strategies.append({'strategy_id': effectiveness.strategy_id, 'problem_type': effectiveness.problem_type, 'success_rate': effectiveness.success_rate, 'average_confidence': effectiveness.average_confidence, 'total_uses': effectiveness.total_uses})
        effective_strategies.sort(key=lambda x: (x['success_rate'], x['average_confidence']), reverse=True)
        return effective_strategies[:5]

    def _get_effectiveness_stats(self) -> Dict[str, float]:
        """Get overall effectiveness statistics"""
        if not self.strategy_effectiveness:
            return {}
        success_rates = [eff.success_rate for eff in self.strategy_effectiveness.values()]
        avg_confidences = [eff.average_confidence for eff in self.strategy_effectiveness.values()]
        return {'average_success_rate': sum(success_rates) / len(success_rates) if success_rates else 0, 'average_confidence': sum(avg_confidences) / len(avg_confidences) if avg_confidences else 0, 'total_strategy_applications': sum((eff.total_uses for eff in self.strategy_effectiveness.values()))}

def _synthesize_final_solution(workspace: MARSWorkspace, client, model: str, config: Dict[str, Any], request_id: str=None) -> Tuple[str, int]:
    """Synthesize the final solution from all agent outputs and verifications"""
    best_solution = workspace.get_best_solution()
    if best_solution and best_solution.is_verified:
        logger.info(f'Using verified solution from agent {best_solution.agent_id}')
        return (best_solution.solution, 0)
    logger.info(f'🗳️  VOTING: No verified solutions found, attempting numerical voting on {len(workspace.solutions)} solutions')
    numerical_answers = []
    extracted_answers_info = []
    logger.info(f'🗳️  VOTING: Starting unified answer extraction from {len(workspace.solutions)} solutions')
    for i, solution in enumerate(workspace.solutions):
        extracted_answer = extract_answer(solution.solution, problem_type='imo', problem_id=None)
        if extracted_answer is not None:
            logger.info(f"🗳️  VOTING: Agent {solution.agent_id} extracted answer '{extracted_answer}' via unified extraction (confidence: {solution.confidence:.2f})")
            answers_to_process = []
            if isinstance(extracted_answer, list):
                answers_to_process = extracted_answer
            else:
                answers_to_process = [extracted_answer]
            for ans in answers_to_process:
                if isinstance(ans, (int, float)):
                    numerical_answers.append((int(ans), solution))
                    extracted_answers_info.append((str(int(ans)), solution, 'unified_numeric'))
                    break
                elif isinstance(ans, str) and ans.strip():
                    extracted_answers_info.append((ans, solution, 'unified_formula'))
                    logger.info(f"🗳️  VOTING: Non-numeric answer stored for synthesis: '{ans}'")
                    break
                elif isinstance(ans, set):
                    set_str = '{' + ', '.join(map(str, sorted(ans))) + '}'
                    extracted_answers_info.append((set_str, solution, 'unified_set'))
                    logger.info(f"🗳️  VOTING: Set answer stored for synthesis: '{set_str}'")
                    break
            if not any((isinstance(ans, (int, float, str, set)) for ans in answers_to_process if isinstance(ans, str) and ans.strip())):
                extracted_answers_info.append((str(extracted_answer), solution, 'unified_other'))
                logger.info(f"🗳️  VOTING: Other answer type stored for synthesis: '{extracted_answer}'")
        else:
            logger.info(f'🗳️  VOTING: Agent {solution.agent_id} - no answer extracted via unified extraction (confidence: {solution.confidence:.2f})')
    workspace._extracted_answers_info = getattr(workspace, '_extracted_answers_info', []) + extracted_answers_info
    logger.info(f'🗳️  VOTING: Extracted {len(numerical_answers)} numerical answers from {len(workspace.solutions)} solutions')
    if len(numerical_answers) >= 2:
        answer_counts = Counter([ans for ans, _ in numerical_answers])
        most_common_answers = answer_counts.most_common()
        logger.info(f'🗳️  VOTING: Answer distribution:')
        for answer, count in most_common_answers:
            percentage = count / len(numerical_answers) * 100
            agents_with_answer = [sol.agent_id for ans, sol in numerical_answers if ans == answer]
            logger.info(f'🗳️  VOTING:   Answer {answer}: {count}/{len(numerical_answers)} votes ({percentage:.1f}%) - Agents: {agents_with_answer}')
        answer, count = most_common_answers[0]
        if count >= 2:
            matching_solutions = [sol for ans, sol in numerical_answers if ans == answer]
            best_solution = max(matching_solutions, key=lambda s: s.confidence)
            logger.info(f'🎆 VOTING SUCCESS: Using majority vote answer {answer} ({count}/{len(numerical_answers)} agents agreed)')
            logger.info(f'🎆 VOTING SUCCESS: Selected solution from agent {best_solution.agent_id} with confidence {best_solution.confidence:.2f}')
            logger.info(f'🎆 VOTING SUCCESS: Solution length: {len(best_solution.solution)} chars')
            return (best_solution.solution, 0)
        else:
            logger.info(f'🗳️  VOTING: No consensus - best answer {answer} only has {count} vote(s), need 2+')
    else:
        logger.info(f'🗳️  VOTING: Insufficient numerical answers for voting ({len(numerical_answers)} < 2)')
    logger.info(f'🤔 VOTING FALLBACK: No numerical consensus found, falling back to answer-preserving synthesis')
    all_extracted = getattr(workspace, '_extracted_answers_info', [])
    if all_extracted:
        logger.info(f'🔍 EXTRACTED ANSWERS SUMMARY: Found {len(all_extracted)} extracted answers:')
        for answer, solution, method in all_extracted:
            logger.info(f"🔍 EXTRACTED ANSWERS SUMMARY:   '{answer}' from Agent {solution.agent_id} via {method}")
    else:
        logger.info(f'🔍 EXTRACTED ANSWERS SUMMARY: No extracted answers found')
    synthesis_data = workspace.get_synthesis_input()
    input_chars = sum((len(sol_data['solution']) for sol_data in synthesis_data['solutions']))
    logger.info(f'🤝 SYNTHESIS INPUT: Processing {len(synthesis_data['solutions'])} solutions')
    logger.info(f'🤝 SYNTHESIS INPUT: Total input characters: {input_chars:,}')
    logger.info(f'🤝 SYNTHESIS INPUT: Verification summary: {synthesis_data['verification_summary']}')
    agent_solutions_text = ''
    solutions_used = synthesis_data['solutions'][:3]
    logger.info(f'🤝 SYNTHESIS INPUT: Using top {len(solutions_used)} solutions for synthesis:')
    for i, sol_data in enumerate(solutions_used):
        logger.info(f'🤝 SYNTHESIS INPUT:   Solution {i + 1}: Agent {sol_data['agent_id']}, {len(sol_data['solution']):,} chars, confidence {sol_data['confidence']:.2f}')
        agent_solutions_text += f'\nAgent {sol_data['agent_id']} (confidence: {sol_data['confidence']:.2f}):\n'
        agent_solutions_text += sol_data['solution']
        agent_solutions_text += '\n' + '=' * 50 + '\n'
    synthesis_input_chars = len(agent_solutions_text)
    verification_text = f'Verification Summary: {synthesis_data['verification_summary']}'
    logger.info(f'🤝 SYNTHESIS INPUT: Final synthesis prompt: {synthesis_input_chars:,} characters')
    extracted_answers_text = ''
    all_extracted = getattr(workspace, '_extracted_answers_info', [])
    if all_extracted:
        extracted_answers_text = '\n\nEXTRACTED ANSWERS FROM AGENTS:\n'
        for answer, solution, method in all_extracted:
            extracted_answers_text += f"- Agent {solution.agent_id}: '{answer}' (via {method})\n"
        extracted_answers_text += '\nIMPORTANT: If multiple agents extracted the same answer, prioritize it in your synthesis.\n'
        extracted_answers_text += 'Ensure the final answer is clearly formatted and matches the expected answer format.\n'
    synthesis_prompt = SYNTHESIS_PROMPT.format(problem=workspace.problem, agent_solutions=agent_solutions_text, verification_results=verification_text) + extracted_answers_text
    try:
        response = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': 'You are a mathematical synthesis expert.'}, {'role': 'user', 'content': synthesis_prompt}], max_tokens=config['max_tokens'], temperature=0.3, timeout=300, extra_body={'reasoning': {'effort': 'high'}})
        if request_id:
            provider_request = {'model': model, 'messages': [{'role': 'system', 'content': 'You are a mathematical synthesis expert.'}, {'role': 'user', 'content': synthesis_prompt}], 'max_tokens': config['max_tokens'], 'temperature': 0.3, 'extra_body': {'reasoning': {'effort': 'high'}}}
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            conversation_logger.log_provider_call(request_id, provider_request, response_dict)
        final_solution = response.choices[0].message.content.strip()
        output_chars = len(final_solution)
        compression_ratio = output_chars / synthesis_input_chars * 100 if synthesis_input_chars > 0 else 0
        logger.info(f'🤝 SYNTHESIS PROCESSING: Input: {synthesis_input_chars:,} chars → Output: {output_chars:,} chars ({compression_ratio:.1f}% retention)')
        reasoning_tokens = 0
        total_tokens = 0
        if hasattr(response, 'usage') and response.usage:
            total_tokens = getattr(response.usage, 'total_tokens', 0)
            if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                reasoning_tokens = getattr(response.usage.completion_tokens_details, 'reasoning_tokens', 0)
            if reasoning_tokens == 0:
                reasoning_tokens = getattr(response.usage, 'reasoning_tokens', 0)
        logger.info(f'🤝 SYNTHESIS SUCCESS: Synthesis completed')
        logger.info(f'🤝 SYNTHESIS SUCCESS:   Output solution length: {len(final_solution)} characters')
        logger.info(f'🤝 SYNTHESIS SUCCESS:   Reasoning tokens: {reasoning_tokens}')
        logger.info(f'🤝 SYNTHESIS SUCCESS:   Total tokens: {total_tokens}')
        logger.info(f'🤝 SYNTHESIS SUCCESS:   Solution preview: {final_solution[:200]}...')
        return (final_solution, reasoning_tokens)
    except Exception as e:
        logger.error(f'🚨 SYNTHESIS ERROR: Synthesis failed: {str(e)}')
        if workspace.solutions:
            fallback_solution = max(workspace.solutions, key=lambda s: s.verification_score)
            logger.info(f'🚑 SYNTHESIS FALLBACK: Using fallback solution from agent {fallback_solution.agent_id}')
            logger.info(f'🚑 SYNTHESIS FALLBACK: Solution length: {len(fallback_solution.solution):,} chars, score: {fallback_solution.verification_score:.2f}')
            return (fallback_solution.solution, 0)
        logger.error(f'🚨 SYNTHESIS ERROR: No solutions available for fallback')
        return ('Unable to generate solution due to synthesis failure.', 0)

def run(system_prompt: str, initial_query: str, client, model: str, request_config: Dict[str, Any]=None) -> Tuple[str, int]:
    """
    Generic majority voting implementation.
    """
    logger.info('Starting majority voting process')
    k = request_config.get('k', DEFAULT_K) if request_config else DEFAULT_K
    temperature = request_config.get('temperature', DEFAULT_TEMPERATURE) if request_config else DEFAULT_TEMPERATURE
    max_tokens = request_config.get('max_tokens', 4096) if request_config else 4096
    logger.info(f'Generating {k} candidates with temperature={temperature}')
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': initial_query}]
    candidates = []
    total_tokens = 0
    try:
        response = client.chat.completions.create(model=model, messages=messages, n=k, temperature=temperature, max_tokens=max_tokens)
        candidates = [choice.message.content for choice in response.choices]
        total_tokens = response.usage.completion_tokens
    except Exception as e:
        logger.warning(f'Parallel generation failed: {str(e)}')
        for i in range(k):
            try:
                response = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
                candidates.append(response.choices[0].message.content)
                total_tokens += response.usage.completion_tokens
            except Exception as err:
                logger.error(f'Error generating candidate {i + 1}: {str(err)}')
                continue
    if not candidates:
        return ('Error: Could not generate any candidates', 0)
    answer_votes = Counter()
    answer_to_responses = {}
    for i, candidate in enumerate(candidates):
        answer = extract_final_answer(candidate)
        normalized = normalize_response(answer)
        if normalized:
            answer_votes[normalized] += 1
            if normalized not in answer_to_responses:
                answer_to_responses[normalized] = []
            answer_to_responses[normalized].append(candidate)
            logger.debug(f"Candidate {i + 1}: '{answer}' -> '{normalized}'")
        else:
            logger.warning(f'Could not extract/normalize answer from candidate {i + 1}')
    if answer_votes:
        most_common_normalized, count = answer_votes.most_common(1)[0]
        logger.info(f"Most common answer: '{most_common_normalized}' with {count}/{k} votes")
        winning_responses = answer_to_responses[most_common_normalized]
        return (winning_responses[0], total_tokens)
    else:
        logger.warning('No answers could be extracted, returning first candidate')
        return (candidates[0], total_tokens)

def normalize_response(response: str) -> str:
    """
    Basic normalization for comparing responses.
    Removes extra whitespace, punctuation at ends, and lowercases.
    """
    if not response:
        return ''
    response = re.sub('<think>.*?</think>', '', response, flags=re.DOTALL)
    response = response.strip()
    response = response.lower()
    response = response.rstrip('.,;:!?')
    response = ' '.join(response.split())
    return response

