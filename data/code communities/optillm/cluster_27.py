# Cluster 27

def execute_code(code: str) -> Tuple[Any, str]:
    """Attempt to execute the code using Jupyter notebook kernel and return result or error."""
    logger.info('Attempting to execute code in notebook kernel')
    logger.info(f'Code:\n{code}')
    try:
        sanitized_code = sanitize_code(code)
        notebook = nbformat.v4.new_notebook()
        enhanced_code = f"""\n{sanitized_code}\n\n# Capture the answer variable for output\nif 'answer' in locals():\n    print(f"ANSWER_RESULT: {{answer}}")\nelse:\n    print("ANSWER_RESULT: No answer variable found")\n"""
        notebook['cells'] = [nbformat.v4.new_code_cell(enhanced_code)]
        notebook_json = nbformat.writes(notebook)
        notebook_bytes = notebook_json.encode('utf-8')
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.ipynb', delete=False) as tmp:
            tmp.write(notebook_bytes)
            tmp.flush()
            tmp_name = tmp.name
        try:
            with open(tmp_name, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=30, kernel_name='python3')
            ep.preprocess(nb, {'metadata': {'path': './'}})
            output = ''
            error_output = ''
            for cell in nb.cells:
                if cell.cell_type == 'code' and cell.outputs:
                    for output_item in cell.outputs:
                        if output_item.output_type == 'stream':
                            if output_item.name == 'stdout':
                                output += output_item.text
                            elif output_item.name == 'stderr':
                                error_output += output_item.text
                        elif output_item.output_type == 'execute_result':
                            output += str(output_item.data.get('text/plain', ''))
                        elif output_item.output_type == 'error':
                            error_output += f'{output_item.ename}: {output_item.evalue}'
            if error_output:
                logger.error(f'Execution failed: {error_output}')
                return (None, error_output)
            output = output.strip()
            if 'ANSWER_RESULT:' in output:
                answer_line = [line for line in output.split('\n') if 'ANSWER_RESULT:' in line][-1]
                answer_str = answer_line.split('ANSWER_RESULT:', 1)[1].strip()
                if answer_str == 'No answer variable found':
                    error = 'Code executed but did not produce an answer variable'
                    logger.warning(error)
                    return (None, error)
                try:
                    answer = ast.literal_eval(answer_str)
                except (ValueError, SyntaxError):
                    answer = answer_str
                logger.info(f'Execution successful. Answer: {answer}')
                return (answer, None)
            elif output:
                logger.info(f'Execution completed with output: {output}')
                return (output, None)
            else:
                error = 'Code executed but produced no output'
                logger.warning(error)
                return (None, error)
        finally:
            try:
                os.unlink(tmp_name)
            except:
                pass
    except Exception as e:
        error = f'Notebook execution failed: {str(e)}'
        logger.error(error)
        return (None, error)

def sanitize_code(code: str) -> str:
    """Prepare code for safe execution by removing problematic visualization code."""
    lines = code.split('\n')
    safe_lines = []
    for line in lines:
        if any((x in line.lower() for x in ['matplotlib', 'plt.', '.plot(', '.show(', 'figure', 'subplot'])):
            safe_lines.append(f'# {line}  # Removed for safety')
        else:
            safe_lines.append(line)
    return '\n'.join(safe_lines)

def generate_fixed_code(original_code: str, error: str, client, model: str) -> Tuple[str, int]:
    """Ask LLM to fix the broken code."""
    logger.info('Requesting code fix from LLM')
    logger.info(f'Original error: {error}')
    response = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': CODE_FIX_PROMPT.format(code=original_code, error=error)}, {'role': 'user', 'content': 'Fix the code to make it work.'}], temperature=0.2)
    fixed_code = response.choices[0].message.content
    code_blocks = extract_code_blocks(fixed_code)
    if code_blocks:
        logger.info('Received fixed code from LLM')
        return (code_blocks[0], response.usage.completion_tokens)
    else:
        logger.warning('No code block found in LLM response')
        return (None, response.usage.completion_tokens)

def extract_code_blocks(text: str) -> List[str]:
    """Extract Python code blocks from text."""
    pattern = '```python\\s*(.*?)\\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    blocks = [m.strip() for m in matches]
    logger.info(f'Extracted {len(blocks)} code blocks')
    for i, block in enumerate(blocks):
        logger.info(f'Code block {i + 1}:\n{block}')
    return blocks

def run(system_prompt: str, initial_query: str, client, model: str) -> Tuple[str, int]:
    """Main Chain of Code execution function."""
    logger.info('Starting Chain of Code execution')
    logger.info(f'Query: {initial_query}')
    messages = [{'role': 'system', 'content': system_prompt + '\n' + CHAIN_OF_CODE_PROMPT}, {'role': 'user', 'content': initial_query}]
    response = client.chat.completions.create(model=model, messages=messages, temperature=0.7)
    total_tokens = response.usage.completion_tokens
    code_blocks = extract_code_blocks(response.choices[0].message.content)
    if not code_blocks:
        logger.warning('No code blocks found in response')
        return (response.choices[0].message.content, total_tokens)
    current_code = code_blocks[0]
    fix_attempts = 0
    last_error = None
    while fix_attempts < MAX_FIX_ATTEMPTS:
        fix_attempts += 1
        logger.info(f'Execution attempt {fix_attempts}/{MAX_FIX_ATTEMPTS}')
        answer, error = execute_code(current_code)
        if error is None:
            logger.info(f'Successful execution on attempt {fix_attempts}')
            return (str(answer), total_tokens)
        last_error = error
        if fix_attempts >= MAX_FIX_ATTEMPTS:
            logger.warning(f'Failed after {fix_attempts} fix attempts')
            break
        logger.info(f'Requesting code fix, attempt {fix_attempts}')
        fixed_code, fix_tokens = generate_fixed_code(current_code, error, client, model)
        total_tokens += fix_tokens
        if fixed_code:
            current_code = fixed_code
        else:
            logger.error('Failed to get fixed code from LLM')
            break
    logger.info('All execution attempts failed, trying simulation')
    simulated_answer, sim_tokens = simulate_execution(current_code, last_error, client, model)
    total_tokens += sim_tokens
    if simulated_answer is not None:
        logger.info('Successfully got answer from simulation')
        return (str(simulated_answer), total_tokens)
    logger.warning('All strategies failed')
    return (f'Error: Could not solve problem after all attempts. Last error: {last_error}', total_tokens)

def simulate_execution(code: str, error: str, client, model: str) -> Tuple[Any, int]:
    """Ask LLM to simulate code execution."""
    logger.info('Attempting code simulation with LLM')
    response = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': SIMULATION_PROMPT.format(code=code, error=error)}, {'role': 'user', 'content': 'Simulate this code and return the final answer value.'}], temperature=0.2)
    try:
        result = response.choices[0].message.content.strip()
        try:
            answer = ast.literal_eval(result)
        except:
            answer = result
        logger.info(f'Simulation successful. Result: {answer}')
        return (answer, response.usage.completion_tokens)
    except Exception as e:
        logger.error(f'Failed to parse simulation result: {str(e)}')
        return (None, response.usage.completion_tokens)

def run(system_prompt: str, initial_query: str, client, model: str) -> Tuple[str, int]:
    query, request_code = extract_python_code(initial_query)[0] if extract_python_code(initial_query) else (initial_query, '')
    if should_execute_request_code(query) and request_code:
        code_output = execute_code(request_code)
        context = f'Query: {query}\nCode:\n```python\n{request_code}\n```\nOutput:\n{code_output}'
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': context}]
        response = client.chat.completions.create(model=model, messages=messages)
        return (response.choices[0].message.content.strip(), response.usage.completion_tokens)
    else:
        messages = [{'role': 'system', 'content': system_prompt + EXECUTE_CODE_PROMPT}, {'role': 'user', 'content': initial_query}]
        response = client.chat.completions.create(model=model, messages=messages)
        initial_response = response.choices[0].message.content.strip()
        response_code = extract_python_code(initial_response)
        if response_code:
            code_output = execute_code(response_code[0])
            context = f'Initial response:\n{initial_response}\n\nCode output:\n{code_output}'
            messages.append({'role': 'assistant', 'content': initial_response})
            messages.append({'role': 'user', 'content': f'Based on the code execution output, please provide a final response:\n{context}'})
            final_response = client.chat.completions.create(model=model, messages=messages)
            return (final_response.choices[0].message.content.strip(), response.usage.completion_tokens + final_response.usage.completion_tokens)
        else:
            return (initial_response, response.usage.completion_tokens)

def extract_python_code(text: str) -> List[str]:
    """Extract Python code blocks from text."""
    pattern = '```python\\s*(.*?)\\s*```'
    return re.findall(pattern, text, re.DOTALL)

def should_execute_request_code(query: str) -> bool:
    """Decide whether to execute code from the request based on the query."""
    keywords = ['run', 'execute', 'output', 'result']
    return any((keyword in query.lower() for keyword in keywords))

