# Cluster 2

def generate_features(ds, df, model='gpt-3.5-turbo', just_print_prompt=False, iterative=1, metric_used=None, iterative_method='logistic', display_method='markdown', n_splits=10, n_repeats=2):

    def format_for_display(code):
        code = code.replace('```python', '').replace('```', '').replace('<end>', '')
        return code
    if display_method == 'markdown':
        from IPython.display import display, Markdown
        display_method = lambda x: display(Markdown(x))
    else:
        display_method = print
    assert iterative == 1 or metric_used is not None, 'metric_used must be set if iterative'
    prompt = build_prompt_from_df(ds, df, iterative=iterative)
    if just_print_prompt:
        code, prompt = (None, prompt)
        return (code, prompt, None)

    def generate_code(messages):
        if model == 'skip':
            return ''
        client = openai.OpenAI()
        completion = client.chat.completions.create(model=model, messages=messages, stop=['```end'], temperature=0.5, max_completion_tokens=500)
        completion = response.model_dump()
        code = completion['choices'][0]['message']['content']
        code = code.replace('```python', '').replace('```', '').replace('<end>', '')
        return code

    def execute_and_evaluate_code_block(full_code, code):
        old_accs, old_rocs, accs, rocs = ([], [], [], [])
        ss = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
        for train_idx, valid_idx in ss.split(df):
            df_train, df_valid = (df.iloc[train_idx], df.iloc[valid_idx])
            target_train = df_train[ds[4][-1]]
            target_valid = df_valid[ds[4][-1]]
            df_train = df_train.drop(columns=[ds[4][-1]])
            df_valid = df_valid.drop(columns=[ds[4][-1]])
            df_train_extended = copy.deepcopy(df_train)
            df_valid_extended = copy.deepcopy(df_valid)
            try:
                df_train = run_llm_code(full_code, df_train, convert_categorical_to_integer=not ds[0].startswith('kaggle'))
                df_valid = run_llm_code(full_code, df_valid, convert_categorical_to_integer=not ds[0].startswith('kaggle'))
                df_train_extended = run_llm_code(full_code + '\n' + code, df_train_extended, convert_categorical_to_integer=not ds[0].startswith('kaggle'))
                df_valid_extended = run_llm_code(full_code + '\n' + code, df_valid_extended, convert_categorical_to_integer=not ds[0].startswith('kaggle'))
            except Exception as e:
                display_method(f'Error in code execution. {type(e)} {e}')
                display_method(f'```python\n{format_for_display(code)}\n```\n')
                return (e, None, None, None, None)
            df_train[ds[4][-1]] = target_train
            df_valid[ds[4][-1]] = target_valid
            df_train_extended[ds[4][-1]] = target_train
            df_valid_extended[ds[4][-1]] = target_valid
            from contextlib import contextmanager
            import sys, os
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    result_old = evaluate_dataset(df_train=df_train, df_test=df_valid, prompt_id='XX', name=ds[0], method=iterative_method, metric_used=metric_used, seed=0, target_name=ds[4][-1])
                    result_extended = evaluate_dataset(df_train=df_train_extended, df_test=df_valid_extended, prompt_id='XX', name=ds[0], method=iterative_method, metric_used=metric_used, seed=0, target_name=ds[4][-1])
                finally:
                    sys.stdout = old_stdout
            old_accs += [result_old['roc']]
            old_rocs += [result_old['acc']]
            accs += [result_extended['roc']]
            rocs += [result_extended['acc']]
        return (None, rocs, accs, old_rocs, old_accs)
    messages = [{'role': 'system', 'content': 'You are an expert datascientist assistant solving Kaggle problems. You answer only by generating code. Answer as concisely as possible.'}, {'role': 'user', 'content': prompt}]
    display_method(f'*Dataset description:*\n {ds[-1]}')
    n_iter = iterative
    full_code = ''
    i = 0
    while i < n_iter:
        try:
            code = generate_code(messages)
        except Exception as e:
            display_method('Error in LLM API.' + str(e))
            continue
        i = i + 1
        e, rocs, accs, old_rocs, old_accs = execute_and_evaluate_code_block(full_code, code)
        if e is not None:
            messages += [{'role': 'assistant', 'content': code}, {'role': 'user', 'content': f'Code execution failed with error: {type(e)} {e}.\n Code: ```python{code}```\n Generate next feature (fixing error?):\n                                ```python\n                                '}]
            continue
        improvement_roc = np.nanmean(rocs) - np.nanmean(old_rocs)
        improvement_acc = np.nanmean(accs) - np.nanmean(old_accs)
        add_feature = True
        add_feature_sentence = 'The code was executed and changes to ´df´ were kept.'
        if improvement_roc + improvement_acc <= 0:
            add_feature = False
            add_feature_sentence = f'The last code changes to ´df´ were discarded. (Improvement: {improvement_roc + improvement_acc})'
        display_method('\n' + f'*Iteration {i}*\n' + f'```python\n{format_for_display(code)}\n```\n' + f'Performance before adding features ROC {np.nanmean(old_rocs):.3f}, ACC {np.nanmean(old_accs):.3f}.\n' + f'Performance after adding features ROC {np.nanmean(rocs):.3f}, ACC {np.nanmean(accs):.3f}.\n' + f'Improvement ROC {improvement_roc:.3f}, ACC {improvement_acc:.3f}.\n' + f'{add_feature_sentence}\n' + f'\n')
        if len(code) > 10:
            messages += [{'role': 'assistant', 'content': code}, {'role': 'user', 'content': f'Performance after adding feature ROC {np.nanmean(rocs):.3f}, ACC {np.nanmean(accs):.3f}. {add_feature_sentence}\nNext codeblock:\n'}]
        if add_feature:
            full_code += code
    return (full_code, prompt, messages)

def execute_and_evaluate_code_block(full_code, code):
    old_accs, old_rocs, accs, rocs = ([], [], [], [])
    ss = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
    for train_idx, valid_idx in ss.split(df):
        df_train, df_valid = (df.iloc[train_idx], df.iloc[valid_idx])
        target_train = df_train[ds[4][-1]]
        target_valid = df_valid[ds[4][-1]]
        df_train = df_train.drop(columns=[ds[4][-1]])
        df_valid = df_valid.drop(columns=[ds[4][-1]])
        df_train_extended = copy.deepcopy(df_train)
        df_valid_extended = copy.deepcopy(df_valid)
        try:
            df_train = run_llm_code(full_code, df_train, convert_categorical_to_integer=not ds[0].startswith('kaggle'))
            df_valid = run_llm_code(full_code, df_valid, convert_categorical_to_integer=not ds[0].startswith('kaggle'))
            df_train_extended = run_llm_code(full_code + '\n' + code, df_train_extended, convert_categorical_to_integer=not ds[0].startswith('kaggle'))
            df_valid_extended = run_llm_code(full_code + '\n' + code, df_valid_extended, convert_categorical_to_integer=not ds[0].startswith('kaggle'))
        except Exception as e:
            display_method(f'Error in code execution. {type(e)} {e}')
            display_method(f'```python\n{format_for_display(code)}\n```\n')
            return (e, None, None, None, None)
        df_train[ds[4][-1]] = target_train
        df_valid[ds[4][-1]] = target_valid
        df_train_extended[ds[4][-1]] = target_train
        df_valid_extended[ds[4][-1]] = target_valid
        from contextlib import contextmanager
        import sys, os
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                result_old = evaluate_dataset(df_train=df_train, df_test=df_valid, prompt_id='XX', name=ds[0], method=iterative_method, metric_used=metric_used, seed=0, target_name=ds[4][-1])
                result_extended = evaluate_dataset(df_train=df_train_extended, df_test=df_valid_extended, prompt_id='XX', name=ds[0], method=iterative_method, metric_used=metric_used, seed=0, target_name=ds[4][-1])
            finally:
                sys.stdout = old_stdout
        old_accs += [result_old['roc']]
        old_rocs += [result_old['acc']]
        accs += [result_extended['roc']]
        rocs += [result_extended['acc']]
    return (None, rocs, accs, old_rocs, old_accs)

def format_for_display(code):
    code = code.replace('```python', '').replace('```', '').replace('<end>', '')
    return code

def generate_code(messages):
    if model == 'skip':
        return ''
    client = openai.OpenAI()
    completion = client.chat.completions.create(model=model, messages=messages, stop=['```end'], temperature=0.5, max_completion_tokens=500)
    completion = response.model_dump()
    code = completion['choices'][0]['message']['content']
    code = code.replace('```python', '').replace('```', '').replace('<end>', '')
    return code

