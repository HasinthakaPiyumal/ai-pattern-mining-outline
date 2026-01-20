# Cluster 33

def chunk_context(doc: str, chunk_size: int, tokenizer, separator='\n') -> List[str]:
    """
    Splits a long document into token-limited chunks based on a separator, ensuring each chunk fits within `chunk_size`.

    Uses a greedy approach to accumulate text segments (split by `separator`) into chunks that fit within the
    token limit. If a segment alone exceeds the limit, it is recursively broken down using sentence-level
    splitting. Attempts to preserve natural boundaries while minimizing excessive chunking.

    Args:
        doc (str): Input document to split.
        chunk_size (int): Maximum number of tokens allowed per chunk.
        tokenizer: Tokenizer instance with `.encode()` method to compute token length.
        separator (str): Delimiter to split initial segments (default: newline).

    Returns:
        List[str]: List of non-empty, token-constrained document chunks.
    """
    paragraphs = doc.split(separator)
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    separator_len = get_prompt_length(separator, tokenizer, no_special_tokens=True)
    docs = []
    current_doc = []
    total = 0
    for paragraph in paragraphs:
        plen = get_prompt_length(paragraph, tokenizer, no_special_tokens=True)
        if total + plen + (separator_len if len(current_doc) > 0 else 0) > chunk_size:
            if total > chunk_size:
                logger.info(f'Created a chunk of size {total}, which is longer than the specified {chunk_size}')
                if len(current_doc) == 1:
                    split_again = split_into_granular_chunks(current_doc[0], chunk_size, tokenizer)
                    docs.extend(split_again)
                    current_doc = []
                    total = 0
            if len(current_doc) > 0:
                doc = separator.join(current_doc)
                if doc is not None:
                    docs.append(doc)
                while total > 0 or (total + plen + (separator_len if len(current_doc) > 0 else 0) > chunk_size and total > 0):
                    total -= get_prompt_length(current_doc[0], tokenizer, no_special_tokens=True) + (separator_len if len(current_doc) > 1 else 0)
                    current_doc = current_doc[1:]
        current_doc.append(paragraph)
        total += plen + (separator_len if len(current_doc) > 1 else 0)
    if get_prompt_length(current_doc[-1], tokenizer, no_special_tokens=True) > chunk_size and len(current_doc) == 1:
        split_again = split_into_granular_chunks(current_doc[0], chunk_size, tokenizer)
        docs.extend(split_again)
        current_doc = []
    else:
        doc = separator.join(current_doc)
        if doc is not None:
            docs.append(doc)
    return [doc for doc in docs if doc.strip()]

def get_prompt_length(prompt: str, tokenizer, no_special_tokens=False, **kwargs) -> int:
    """
    Returns the token length of a prompt using the given tokenizer.
    """
    if isinstance(prompt, list):
        prompt = '\n\n'.join(prompt)
    if no_special_tokens:
        kwargs['add_special_tokens'] = False
    return len(tokenizer.encode(prompt, **kwargs))

def split_into_granular_chunks(text: str, chunk_size: int, tokenizer, spliter='([。！？；.?!;])') -> List[str]:
    """
    Splits long text into granular, token-length-constrained chunks using sentence boundaries.

    Sentences are first extracted using a delimiter pattern (`spliter`), then grouped into chunks such that
    each chunk does not exceed the specified `chunk_size` (in tokens). If a chunk still exceeds the limit,
    it is recursively broken down further using whitespace as a fallback.

    Ensures that the final chunks are balanced: if the last chunk is too small, it redistributes the last two
    chunks more evenly by re-splitting and re-allocating their sentences.

    Args:
        text (str): Input text to be chunked.
        chunk_size (int): Maximum number of tokens per chunk.
        tokenizer: Tokenizer instance with `.encode()` method to compute token length.
        spliter (str): Regex pattern to split sentences.

    Returns:
        List[str]: List of token-limited chunks, each composed of one or more sentences.
    """
    sentences = split_sentences(text, spliter)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        sentence_length = get_prompt_length(sentence, tokenizer)
        if get_prompt_length(current_chunk, tokenizer) + sentence_length <= chunk_size:
            current_chunk += sentence
        else:
            if current_chunk:
                if get_prompt_length(current_chunk, tokenizer) <= chunk_size:
                    chunks.append(current_chunk)
                elif spliter != ' ':
                    chunks.extend(split_into_granular_chunks(current_chunk, chunk_size=chunk_size, tokenizer=tokenizer, spliter=' '))
            current_chunk = sentence
    if current_chunk != '':
        if get_prompt_length(current_chunk, tokenizer) <= chunk_size:
            chunks.append(current_chunk)
        elif spliter != ' ':
            chunks.extend(split_into_granular_chunks(current_chunk, chunk_size=chunk_size, tokenizer=tokenizer, spliter=' '))
    if len(chunks) > 1 and get_prompt_length(chunks[-1], tokenizer) < chunk_size // 2:
        last_chunk = chunks.pop()
        penultimate_chunk = chunks.pop()
        combined_text = penultimate_chunk + last_chunk
        new_sentences = split_sentences(combined_text, spliter)
        new_penultimate_chunk = ''
        new_last_chunk = ''
        start, end = (0, len(new_sentences) - 1)
        while start <= end and len(new_sentences) != 1:
            flag = False
            if get_prompt_length(new_penultimate_chunk + new_sentences[start], tokenizer) <= chunk_size:
                flag = True
                new_penultimate_chunk += new_sentences[start]
                if start == end:
                    break
                start += 1
            if get_prompt_length(new_last_chunk + new_sentences[end], tokenizer) <= chunk_size:
                new_last_chunk = new_sentences[end] + new_last_chunk
                end -= 1
                flag = True
            if flag == False:
                break
        if start < end:
            remaining_sentences = new_sentences[start:end + 1]
            if remaining_sentences:
                remaining_text = ''.join(remaining_sentences)
                words = remaining_text.split(' ')
                end_index = len(words) - 1
                for index, w in enumerate(words):
                    if get_prompt_length(' '.join([new_penultimate_chunk, w]), tokenizer) <= chunk_size:
                        new_penultimate_chunk = ' '.join([new_penultimate_chunk, w])
                    else:
                        end_index = index
                        break
                if end_index != len(words) - 1:
                    new_last_chunk = ' '.join(words[end_index:]) + ' ' + new_last_chunk
        if len(new_sentences) == 1:
            chunks.append(penultimate_chunk)
            chunks.append(last_chunk)
        else:
            chunks.append(new_penultimate_chunk)
            chunks.append(new_last_chunk)
    return chunks

def split_sentences(text: str, spliter: str):
    """
    Splits text into sentences or segments based on a given delimiter while preserving punctuation.

    For punctuation-based splitters (e.g., ".", "!", "。"), it interleaves text and punctuation.
    For space-based splitting, it preserves trailing spaces.

    Args:
        text (str): The input text to split.
        spliter (str): Delimiter regex pattern (e.g., r"([.!?])", r"(。)", or " ").

    Returns:
        List[str]: List of split sentence-like segments with punctuation retained.
    """
    text = text.strip()
    sentence_list = re.split(spliter, text)
    if spliter != ' ':
        sentences = [''.join(i) for i in zip(sentence_list[0::2], sentence_list[1::2])]
        if len(sentence_list) % 2 != 0 and sentence_list[-1] != '':
            sentences.append(sentence_list[-1])
    else:
        sentences = [i + ' ' for i in sentence_list if i != '']
        sentences[-1] = sentences[-1].strip()
    return sentences

def mapreduce(system_prompt: str, query: str, context: str, qa_history: str, client, model: str, tokenizer, longcepo_config: LongCepoConfig, cb_log: CBLog, answer_tags: Tuple[str]=('Answer:', '**Answer**:', '**Answer**'), irrelevance_tags: Tuple[str]=('[NO INFORMATION]',)) -> Tuple[str, CBLog]:
    """
    Executes a MapReduce-style inference pipeline to answer a query from long context.

    The function splits the input context into chunks, summarizes and evaluates each with the model (Map),
    collapses intermediate answers to reduce redundancy, and then generates a final answer (Reduce).
    Irrelevant responses are filtered based on `irrelevance_tags`.

    Args:
        system_prompt (str): System prompt string.
        query (str): User query.
        context (str): Long-form input context to process.
        qa_history (str): QA history string for prompt injection.
        client: LLM API client.
        model (str): Base model name.
        tokenizer: Tokenizer to compute token lengths.
        longcepo_config (LongCepoConfig): Config with hyper-parameters and token limits.
        cb_log (CBLog): Log object for tracking model calls.
        answer_tags (Tuple[str]): Tags used to extract the final answer from model output.
        irrelevance_tags (Tuple[str]): Tags used to identify and remove irrelevant outputs.

    Returns:
        Tuple[str, CBLog]: Final extracted answer and updated log object.
    """
    logger.info(f'MapReduce query: {query}')
    qa_history_stub = f'\n\nAnswers to related questions:\n\n{qa_history}' if qa_history else ''
    context_chunks = chunk_context(context, longcepo_config.chunk_size, tokenizer)

    def fetch_chunk_summary(client, model, chunk, query, system_prompt):
        return get_prompt_response(client, model, longcepo_config.summary_prompt.format(question=query, context=chunk), system_prompt, max_tokens=longcepo_config.max_output_tokens_summary, temperature=longcepo_config.temperature_map)
    summaries, cb_log = concurrent_map(fetch_chunk_summary, client, model, context_chunks, query, system_prompt, cb_log)
    num_summaries = longcepo_config.num_neighbor_summaries
    summaries_per_chunk = ['\n\n'.join(summaries[max(0, summary_idx - num_summaries):min(len(summaries) - 1, summary_idx + num_summaries)]) for summary_idx in range(len(summaries))]

    def fetch_map_response(client, model, chunk, query, system_prompt, summary):
        return get_prompt_response(client, model, longcepo_config.map_prompt.format(question=query, context=chunk, summary=summary, qa_history_stub=qa_history_stub), system_prompt, max_tokens=longcepo_config.max_output_tokens, temperature=longcepo_config.temperature_map)
    result, cb_log = concurrent_map(fetch_map_response, client, model, context_chunks, query, system_prompt, cb_log, summaries_per_chunk=summaries_per_chunk)
    result = remove_chunks(result, irrelevance_tags)
    if not result:
        return ('No information', cb_log)
    logger.info(f'Removed {len(context_chunks) - len(result)} chunks from total {len(context_chunks)} chunks')
    result, cb_log = collapse_chunks(client, model, result, query, system_prompt, qa_history_stub, tokenizer, cb_log, longcepo_config, irrelevance_tags)
    if not result:
        return ('No information', cb_log)
    prompt = longcepo_config.reduce_prompt.format(question=query, context=format_chunk_list(result), qa_history_stub=qa_history_stub)
    gen_fn = partial(get_prompt_response, client=client, model=model, prompt=prompt, system_prompt=system_prompt, max_tokens=longcepo_config.max_output_tokens, temperature=longcepo_config.temperature_reduce)
    reduce_result, upd_log = loop_until_match(gen_fn, answer_tags)
    cb_log.update(upd_log)
    final_answer = reduce_result
    for answer_tag in answer_tags:
        if answer_tag in reduce_result:
            final_answer = reduce_result.split(answer_tag)[-1].strip()
            break
    return (final_answer, cb_log)

def fetch_chunk_summary(client, model, chunk, query, system_prompt):
    return get_prompt_response(client, model, longcepo_config.summary_prompt.format(question=query, context=chunk), system_prompt, max_tokens=longcepo_config.max_output_tokens_summary, temperature=longcepo_config.temperature_map)

def get_prompt_response(client, model: str, prompt: str, system_prompt: str, max_tokens: int, temperature: float=0.7, top_p: float=0.7):
    """
    Helper function that sends a prompt to the chat-based LLM API and returns the generated response along with usage logging.

    Args:
        client: LLM API client.
        model (str): Base model name.
        prompt (str): The user prompt to send.
        system_prompt (str): System prompt string.
        max_tokens (int): Maximum number of tokens in the response.
        temperature (float): Sampling temperature for randomness (default: 0.7).
        top_p (float): Cumulative probability cutoff for token selection (default: 0.7).

    Returns:
        Tuple[str, CBLog]: The model's response text and a CBLog object tracking token usage.
    """
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, top_p=top_p, temperature=temperature, stream=False)
    upd_log = CBLog(llm_calls=1, total_tokens=response.usage.total_tokens, completion_tokens=response.usage.completion_tokens)
    return (response.choices[0].message.content, upd_log)

def concurrent_map(gen_function: Callable, client, model: str, context_chunks: List[str], query: str, system_prompt: str, cb_log: CBLog, summaries_per_chunk: Optional[List[str]]=None, workers: int=16) -> Tuple[List[str], CBLog]:
    """
    Runs `gen_function` concurrently over a list of context chunks.

    Args:
        gen_function (Callable): Function to call with each chunk and associated arguments.
        client: LLM API client.
        model (str): Base model name.
        context_chunks (List[str]): Input context chunks.
        query (str): User query.
        system_prompt (str): System prompt string.
        cb_log (CBLog): Log object for tracking model calls.
        summaries_per_chunk (Optional[List[str]]): Concatenated neighbor summaries for each chunk.
        workers (int): Number of threads to use.

    Returns:
        Tuple[List[str], CBLog]: List of responses (in original order) and updated log object.
    """
    result = [None] * len(context_chunks)
    wrapped_gen_function = lambda index, *args: (index, gen_function(*args))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {}
        for idx, chunk in enumerate(context_chunks):
            args = [client, model, chunk, query, system_prompt]
            if summaries_per_chunk is not None:
                args.append(summaries_per_chunk[idx])
            future_to_idx[executor.submit(wrapped_gen_function, idx, *args)] = idx
        for future in as_completed(future_to_idx):
            try:
                index, (response, upd_log) = future.result()
                result[index] = response
                cb_log.update(upd_log)
            except Exception as e:
                logger.error(f'Error processing chunk: {e}')
    return (result, cb_log)

def fetch_map_response(client, model, chunk, query, system_prompt, summary):
    return get_prompt_response(client, model, longcepo_config.map_prompt.format(question=query, context=chunk, summary=summary, qa_history_stub=qa_history_stub), system_prompt, max_tokens=longcepo_config.max_output_tokens, temperature=longcepo_config.temperature_map)

def remove_chunks(chunks: List[str], irrelevance_tags: Tuple[str]) -> List[str]:
    """
    Filter out chunks that contain at least one of irrelevance tags.
    """
    new_chunks = []
    for chunk in chunks:
        flag = False
        for tag in irrelevance_tags:
            if tag.upper() in chunk.upper():
                flag = True
                break
        if not flag:
            new_chunks.append(chunk)
    return new_chunks

def loop_until_match(function: Callable, pattern_list: Tuple[str], num_attempts: int=10):
    """
    Repeatedly calls a function until its output matches one of the given patterns or max attempts is reached.

    Args:
        function (Callable): Function returning (answer: str, cb_log).
        pattern_list (Tuple[str]): Patterns to match in the answer.
        num_attempts (int): Max number of attempts (default: 10).

    Returns:
        Tuple[str, Any]: The matching answer and its corresponding log object.
    """
    correct_format = False
    for _ in range(num_attempts):
        answer, cb_log = function()
        for pattern in pattern_list:
            if pattern in answer:
                correct_format = True
        if correct_format:
            break
        logger.info('Wrong output formatting, retrying...')
    return (answer, cb_log)

def collapse_chunks(client, model: str, context_chunks: List[str], query: str, system_prompt: str, qa_history_stub: str, tokenizer, cb_log: CBLog, longcepo_config: LongCepoConfig, irrelevance_tags: Tuple[str]=('[NO INFORMATION]',)) -> Tuple[List[str], CBLog]:
    """
    Collapses context chunk pairs in sliding window until the total token count fits within the context window.

    Args:
        client: LLM API client.
        model (str): Base model name.
        context_chunks (List[str]): Input context chunks.
        query (str): User query.
        system_prompt (str): System prompt string.
        qa_history_stub (str): QA history prefix.
        tokenizer: Tokenizer to compute token lengths.
        cb_log (CBLog): Log object for tracking model calls.
        longcepo_config (LongCepoConfig): Config with hyper-parameters and token limits.

    Returns:
        Tuple[List[str], CBLog]: Final context chunks and updated logs.
    """
    num_tokens = get_prompt_length(format_chunk_list(context_chunks), tokenizer)
    token_budget = longcepo_config.max_context_window - get_prompt_length(longcepo_config.reduce_prompt, tokenizer) - longcepo_config.max_output_tokens
    logger.info(f'Pre-collapse length of chunks {num_tokens}, allowed {token_budget}')

    def fetch_collapse_response(client, model, docs, query, system_prompt):
        if len(docs) == 1:
            return (docs[0], CBLog())
        return get_prompt_response(client, model, longcepo_config.collapse_prompt.format(question=query, context='\n\n'.join(docs), qa_history_stub=qa_history_stub), system_prompt, max_tokens=longcepo_config.max_output_tokens, temperature=longcepo_config.temperature_collapse)
    merge_pair_idx = 0
    collapse_step = 0
    while num_tokens is not None and num_tokens > token_budget:
        logger.info(f'Length at collapse stage {collapse_step}: {collapse_step}')
        if len(context_chunks) == 1:
            logger.info(f'Post-collapse length of chunks {num_tokens}')
            return (context_chunks, cb_log)
        chunk_groups = [(context_chunks[i],) for i in range(merge_pair_idx)] + [(context_chunks[merge_pair_idx], context_chunks[merge_pair_idx + 1])] + [(context_chunks[i],) for i in range(merge_pair_idx + 2, len(context_chunks))]
        context_chunks, cb_log = concurrent_map(fetch_collapse_response, client, model, chunk_groups, query, system_prompt, cb_log)
        context_chunks = remove_chunks(context_chunks, irrelevance_tags)
        merge_pair_idx = (merge_pair_idx + 1) % max(len(context_chunks) - 1, 1)
        num_tokens = get_prompt_length(format_chunk_list(context_chunks), tokenizer)
        collapse_step += 1
    logger.info(f'Post-collapse length of chunks {num_tokens}')
    return (context_chunks, cb_log)

def fetch_collapse_response(client, model, docs, query, system_prompt):
    if len(docs) == 1:
        return (docs[0], CBLog())
    return get_prompt_response(client, model, longcepo_config.collapse_prompt.format(question=query, context='\n\n'.join(docs), qa_history_stub=qa_history_stub), system_prompt, max_tokens=longcepo_config.max_output_tokens, temperature=longcepo_config.temperature_collapse)

def run_longcepo(system_prompt: str, initial_query: str, client, model: str) -> Tuple[str, int]:
    """
    Executes the full LongCePO multi-stage pipeline to answer a complex query from long context.

    The pipeline includes:
      - Normalizing the format of the original query
      - Generating a plan of sub-questions
      - Iteratively answering each sub-question using a MapReduce-style question-answering engine
      - Aggregating QA history and producing a final answer with MapReduce

    Args:
        system_prompt (str): System prompt string.
        initial_query (str): Raw input string containing context and query separated by delimiter string.
        client: LLM API client instance.
        model (str): Base model name.

    Returns:
        Tuple[str, int]: Final answer and total number of tokens used across the pipeline.
    """
    context, query, tokenizer, cb_log, longcepo_config = longcepo_init(initial_query)
    normalized_query, upd_log = get_prompt_response(client, model, longcepo_config.query_format_prompt.format(full_query=query), system_prompt, max_tokens=longcepo_config.max_output_tokens)
    cb_log.update(upd_log)
    logger.info(f'Normalized query: {normalized_query}')
    prompt = f'The question is: {normalized_query}'
    gen_fn = partial(get_prompt_response, client=client, model=model, prompt=prompt, system_prompt=longcepo_config.planning_system_prompt, max_tokens=longcepo_config.max_output_tokens)
    planning_response, upd_log = loop_until_match(gen_fn, pattern_list=('<SUB-QUESTIONS>',))
    logger.info(f'Planning stage output:\n\n{planning_response}')
    questions = re.findall('<SUB-QUESTIONS>\\s*(.*?)\\s*</SUB-QUESTIONS>', planning_response, re.DOTALL)[0].strip().splitlines()
    qa_system_prompt = longcepo_config.system_prompt if longcepo_config.system_prompt is not None else system_prompt
    qa_history = ''
    for question in questions:
        if not question:
            continue
        question = re.sub('^\\d+\\.', '', question)
        answer, cb_log = mapreduce(qa_system_prompt, question, context, qa_history, client, model, tokenizer, longcepo_config=longcepo_config, cb_log=cb_log)
        qa_history += f'- Previous question: {question}\n\n'
        answer = re.sub('^:+', '', answer)
        qa_history += f'- Previous answer: {answer}\n\n'
        logger.info(f'QA history:\n\n{qa_history}')
    answer, cb_log = mapreduce(qa_system_prompt, query, context, qa_history, client, model, tokenizer, longcepo_config=longcepo_config, cb_log=cb_log)
    return (answer, cb_log['total_tokens'])

def longcepo_init(initial_query: str) -> Tuple[str, str, PreTrainedTokenizerBase, CBLog, LongCepoConfig]:
    """
    Initializes context, query, tokenizer, logging, and config from an input string.

    Args:
        initial_query (str): Input string containing context and query separated by a delimiter string.

    Returns:
        Tuple[str, str, PreTrainedTokenizerBase, CBLog, LongCepoConfig]:
        Parsed context, query, tokenizer instance, log object, and LongCePO config.
    """
    cb_log = CBLog()
    config = LongCepoConfig()
    context, query = initial_query.split(config.context_query_delimiter)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    return (context.strip(), query.strip(), tokenizer, cb_log, config)

