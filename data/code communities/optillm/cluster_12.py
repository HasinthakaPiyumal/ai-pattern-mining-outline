# Cluster 12

def analyze_results(results: List[Dict], n: int, analyze_thoughts: bool=False, analyze_logits: bool=False):
    """
    Analyze and print summary statistics of the results.
    
    Args:
        results (List[Dict]): List of evaluation results
        n (int): Number of attempts per problem
        analyze_thoughts (bool): Whether to analyze thinking patterns
        analyze_logits (bool): Whether to analyze token probabilities
    """
    total = len(results)
    correct = sum((1 for r in results if r['is_correct']))
    accuracy = correct / total if total > 0 else 0
    print('\n=== Results Summary ===')
    print(f'Evaluation mode: pass@{n}')
    print(f'Total problems: {total}')
    print(f'Correct answers: {correct}')
    print(f'Accuracy: {accuracy:.2%}')
    successful_attempts = [r['first_correct_attempt'] for r in results if r['is_correct']]
    if successful_attempts:
        avg_attempts = sum(successful_attempts) / len(successful_attempts)
        print(f'\nFor correct solutions:')
        print(f'Average attempts needed: {avg_attempts:.2f}')
        print(f'Attempt distribution:')
        for i in range(1, n + 1):
            count = sum((1 for x in successful_attempts if x == i))
            print(f'  Attempt {i}: {count} problems')
    if analyze_thoughts:
        print('\n=== Thinking Pattern Analysis ===')
        correct_attempts = []
        incorrect_attempts = []
        for result in results:
            for attempt in result['attempts']:
                if 'thought_analysis' in attempt:
                    if result['is_correct'] and attempt['predicted_answer'] == result['correct_answer']:
                        correct_attempts.append(attempt)
                    else:
                        incorrect_attempts.append(attempt)

        def calc_stats(attempts):
            if not attempts:
                return {'count': 0, 'avg_thinking_tokens': 0, 'avg_thought_transitions': 0, 'transition_usage': {phrase: 0 for phrase in THOUGHT_TRANSITIONS}, 'has_think_tags_pct': 0}
            thinking_tokens = [a['thought_analysis']['thinking_tokens'] for a in attempts]
            thought_transitions = [a['thought_analysis']['thought_transitions'] for a in attempts]
            has_think_tags = sum((1 for a in attempts if a['thought_analysis']['has_think_tags']))
            transition_usage = defaultdict(int)
            for attempt in attempts:
                for phrase, count in attempt['thought_analysis']['transition_counts'].items():
                    transition_usage[phrase] += count
            return {'count': len(attempts), 'avg_thinking_tokens': statistics.mean(thinking_tokens) if thinking_tokens else 0, 'median_thinking_tokens': statistics.median(thinking_tokens) if thinking_tokens else 0, 'min_thinking_tokens': min(thinking_tokens) if thinking_tokens else 0, 'max_thinking_tokens': max(thinking_tokens) if thinking_tokens else 0, 'avg_thought_transitions': statistics.mean(thought_transitions) if thought_transitions else 0, 'median_thought_transitions': statistics.median(thought_transitions) if thought_transitions else 0, 'transition_usage': dict(transition_usage), 'has_think_tags_pct': has_think_tags / len(attempts) * 100 if attempts else 0}
        correct_stats = calc_stats(correct_attempts)
        incorrect_stats = calc_stats(incorrect_attempts)
        all_stats = calc_stats(correct_attempts + incorrect_attempts)
        print(f'\nOverall Thinking Statistics (All {all_stats['count']} Attempts):')
        print(f'- Average thinking tokens: {all_stats['avg_thinking_tokens']:.2f}')
        print(f'- Median thinking tokens: {all_stats['median_thinking_tokens']}')
        print(f'- Range: {all_stats['min_thinking_tokens']} - {all_stats['max_thinking_tokens']} tokens')
        print(f'- Average thought transitions: {all_stats['avg_thought_transitions']:.2f}')
        print(f'- Median thought transitions: {all_stats['median_thought_transitions']}')
        print(f'- Percentage with <think> tags: {all_stats['has_think_tags_pct']:.2f}%')
        print(f'- Transition phrase usage:')
        for phrase, count in all_stats['transition_usage'].items():
            print(f'  - {phrase}: {count} occurrences')
        print(f'\nCorrect Attempts ({correct_stats['count']}):')
        print(f'- Average thinking tokens: {correct_stats['avg_thinking_tokens']:.2f}')
        print(f'- Median thinking tokens: {correct_stats['median_thinking_tokens']}')
        print(f'- Average thought transitions: {correct_stats['avg_thought_transitions']:.2f}')
        print(f'- Median thought transitions: {correct_stats['median_thought_transitions']}')
        print(f'- Percentage with <think> tags: {correct_stats['has_think_tags_pct']:.2f}%')
        print(f'- Transition phrase usage:')
        for phrase, count in correct_stats['transition_usage'].items():
            print(f'  - {phrase}: {count} occurrences')
        print(f'\nIncorrect Attempts ({incorrect_stats['count']}):')
        print(f'- Average thinking tokens: {incorrect_stats['avg_thinking_tokens']:.2f}')
        print(f'- Median thinking tokens: {incorrect_stats['median_thinking_tokens']}')
        print(f'- Average thought transitions: {incorrect_stats['avg_thought_transitions']:.2f}')
        print(f'- Median thought transitions: {incorrect_stats['median_thought_transitions']}')
        print(f'- Percentage with <think> tags: {incorrect_stats['has_think_tags_pct']:.2f}%')
        print(f'- Transition phrase usage:')
        for phrase, count in incorrect_stats['transition_usage'].items():
            print(f'  - {phrase}: {count} occurrences')
        if correct_attempts and incorrect_attempts:
            print('\nCorrelation Analysis:')
            problems_with_both = defaultdict(lambda: {'correct': [], 'incorrect': []})
            for result in results:
                problem_id = result['index']
                for attempt in result['attempts']:
                    if 'thought_analysis' in attempt:
                        category = 'correct' if attempt['predicted_answer'] == result['correct_answer'] else 'incorrect'
                        problems_with_both[problem_id][category].append(attempt)
            valid_problems = {k: v for k, v in problems_with_both.items() if v['correct'] and v['incorrect']}
            if valid_problems:
                print(f'Found {len(valid_problems)} problems with both correct and incorrect attempts')
                avg_token_diff = []
                avg_transition_diff = []
                for problem_id, attempts in valid_problems.items():
                    correct_tokens = [a['thought_analysis']['thinking_tokens'] for a in attempts['correct']]
                    incorrect_tokens = [a['thought_analysis']['thinking_tokens'] for a in attempts['incorrect']]
                    correct_transitions = [a['thought_analysis']['thought_transitions'] for a in attempts['correct']]
                    incorrect_transitions = [a['thought_analysis']['thought_transitions'] for a in attempts['incorrect']]
                    avg_correct_tokens = statistics.mean(correct_tokens) if correct_tokens else 0
                    avg_incorrect_tokens = statistics.mean(incorrect_tokens) if incorrect_tokens else 0
                    avg_correct_transitions = statistics.mean(correct_transitions) if correct_transitions else 0
                    avg_incorrect_transitions = statistics.mean(incorrect_transitions) if incorrect_transitions else 0
                    avg_token_diff.append(avg_correct_tokens - avg_incorrect_tokens)
                    avg_transition_diff.append(avg_correct_transitions - avg_incorrect_transitions)
                print(f'Average token difference (correct - incorrect): {statistics.mean(avg_token_diff):.2f}')
                print(f'Average transition difference (correct - incorrect): {statistics.mean(avg_transition_diff):.2f}')
    if analyze_logits:
        print('\n=== Logit Analysis ===')
        correct_attempts = []
        incorrect_attempts = []
        for result in results:
            for attempt in result['attempts']:
                if 'logit_analysis' in attempt:
                    if result['is_correct'] and attempt['predicted_answer'] == result['correct_answer']:
                        correct_attempts.append(attempt)
                    else:
                        incorrect_attempts.append(attempt)

        def calc_logit_stats(attempts):
            if not attempts:
                return {'count': 0, 'entropy': None, 'transitions': None}
            entropy_means = []
            entropy_stds = []
            entropy_quartiles = []
            transition_entropies = defaultdict(lambda: {'before': [], 'after': []})
            for attempt in attempts:
                if attempt['logit_analysis'].get('entropy_stats') and attempt['logit_analysis']['entropy_stats'].get('mean'):
                    entropy_means.append(attempt['logit_analysis']['entropy_stats']['mean'])
                    entropy_stds.append(attempt['logit_analysis']['entropy_stats']['std'])
                    if attempt['logit_analysis']['entropy_stats'].get('quartiles'):
                        entropy_quartiles.append(attempt['logit_analysis']['entropy_stats']['quartiles'])
                    if attempt['logit_analysis'].get('transition_entropy'):
                        for phrase, stats in attempt['logit_analysis']['transition_entropy'].items():
                            if stats.get('before_mean') is not None:
                                transition_entropies[phrase]['before'].append(stats['before_mean'])
                            if stats.get('after_mean') is not None:
                                transition_entropies[phrase]['after'].append(stats['after_mean'])
            avg_quartiles = []
            if entropy_quartiles:
                max_quartiles = max((len(q) for q in entropy_quartiles))
                padded_quartiles = [q + [0] * (max_quartiles - len(q)) for q in entropy_quartiles]
                for i in range(max_quartiles):
                    quartile_values = [q[i] for q in padded_quartiles if i < len(q)]
                    avg_quartiles.append(statistics.mean(quartile_values) if quartile_values else 0)
            transition_stats = {}
            for phrase, values in transition_entropies.items():
                if values['before'] and values['after']:
                    before_mean = statistics.mean(values['before'])
                    after_mean = statistics.mean(values['after'])
                    transition_stats[phrase] = {'before_mean': before_mean, 'after_mean': after_mean, 'entropy_change': after_mean - before_mean, 'count': len(values['before'])}
            return {'count': len(attempts), 'entropy': {'mean': statistics.mean(entropy_means) if entropy_means else 0, 'std': statistics.mean(entropy_stds) if entropy_stds else 0, 'quartiles': avg_quartiles}, 'transitions': transition_stats}
        correct_stats = calc_logit_stats(correct_attempts)
        incorrect_stats = calc_logit_stats(incorrect_attempts)
        all_stats = calc_logit_stats(correct_attempts + incorrect_attempts)
        print(f'\nOverall Logit Statistics (All {all_stats['count']} Attempts):')
        if all_stats['entropy'] and all_stats['entropy']['mean']:
            print(f'- Average entropy: {all_stats['entropy']['mean']:.4f}')
            print(f'- Average entropy std: {all_stats['entropy']['std']:.4f}')
            if all_stats['entropy']['quartiles']:
                print(f'- Entropy by generation quartile:')
                for i, q in enumerate(all_stats['entropy']['quartiles']):
                    print(f'  - Q{i + 1}: {q:.4f}')
            if all_stats['transitions']:
                print(f'- Entropy around thought transitions:')
                for phrase, stats in all_stats['transitions'].items():
                    change = stats['entropy_change']
                    change_dir = 'increases' if change > 0 else 'decreases'
                    print(f'  - {phrase} (n={stats['count']}): Entropy {change_dir} by {abs(change):.4f}')
                    print(f'    - Before: {stats['before_mean']:.4f}, After: {stats['after_mean']:.4f}')
        if correct_stats['count'] > 0 and incorrect_stats['count'] > 0:
            print('\nEntropy Comparison (Correct vs Incorrect Attempts):')
            if correct_stats['entropy'] and correct_stats['entropy']['mean'] and incorrect_stats['entropy'] and incorrect_stats['entropy']['mean']:
                correct_entropy = correct_stats['entropy']['mean']
                incorrect_entropy = incorrect_stats['entropy']['mean']
                diff = correct_entropy - incorrect_entropy
                print(f'- Correct attempts avg entropy: {correct_entropy:.4f}')
                print(f'- Incorrect attempts avg entropy: {incorrect_entropy:.4f}')
                print(f'- Difference (correct - incorrect): {diff:.4f}')
                if correct_stats['entropy']['quartiles'] and incorrect_stats['entropy']['quartiles']:
                    print(f'- Entropy progression through generation:')
                    for i in range(min(len(correct_stats['entropy']['quartiles']), len(incorrect_stats['entropy']['quartiles']))):
                        c_q = correct_stats['entropy']['quartiles'][i]
                        i_q = incorrect_stats['entropy']['quartiles'][i]
                        q_diff = c_q - i_q
                        print(f'  - Q{i + 1}: Correct: {c_q:.4f}, Incorrect: {i_q:.4f}, Diff: {q_diff:.4f}')
                common_transitions = set(correct_stats['transitions'].keys()) & set(incorrect_stats['transitions'].keys())
                if common_transitions:
                    print(f'- Entropy changes around thought transitions:')
                    for phrase in common_transitions:
                        c_stats = correct_stats['transitions'][phrase]
                        i_stats = incorrect_stats['transitions'][phrase]
                        c_change = c_stats['entropy_change']
                        i_change = i_stats['entropy_change']
                        print(f'  - {phrase}:')
                        print(f'    - Correct: {c_stats['before_mean']:.4f} → {c_stats['after_mean']:.4f} (Δ {c_change:.4f})')
                        print(f'    - Incorrect: {i_stats['before_mean']:.4f} → {i_stats['after_mean']:.4f} (Δ {i_change:.4f})')
                        print(f'    - Difference in entropy change: {c_change - i_change:.4f}')
    print('\n=== Incorrect Problems ===')
    for r in results:
        if not r['is_correct']:
            print(f'Problem {r['index']}:')
            print(f'Expected: {r['correct_answer']}')
            print('Predicted answers across attempts:', [attempt['predicted_answer'] for attempt in r['attempts']])
            print('---')

def calc_stats(attempts):
    if not attempts:
        return {'count': 0, 'avg_thinking_tokens': 0, 'avg_thought_transitions': 0, 'transition_usage': {phrase: 0 for phrase in THOUGHT_TRANSITIONS}, 'has_think_tags_pct': 0}
    thinking_tokens = [a['thought_analysis']['thinking_tokens'] for a in attempts]
    thought_transitions = [a['thought_analysis']['thought_transitions'] for a in attempts]
    has_think_tags = sum((1 for a in attempts if a['thought_analysis']['has_think_tags']))
    transition_usage = defaultdict(int)
    for attempt in attempts:
        for phrase, count in attempt['thought_analysis']['transition_counts'].items():
            transition_usage[phrase] += count
    return {'count': len(attempts), 'avg_thinking_tokens': statistics.mean(thinking_tokens) if thinking_tokens else 0, 'median_thinking_tokens': statistics.median(thinking_tokens) if thinking_tokens else 0, 'min_thinking_tokens': min(thinking_tokens) if thinking_tokens else 0, 'max_thinking_tokens': max(thinking_tokens) if thinking_tokens else 0, 'avg_thought_transitions': statistics.mean(thought_transitions) if thought_transitions else 0, 'median_thought_transitions': statistics.median(thought_transitions) if thought_transitions else 0, 'transition_usage': dict(transition_usage), 'has_think_tags_pct': has_think_tags / len(attempts) * 100 if attempts else 0}

def calc_logit_stats(attempts):
    if not attempts:
        return {'count': 0, 'entropy': None, 'transitions': None}
    entropy_means = []
    entropy_stds = []
    entropy_quartiles = []
    transition_entropies = defaultdict(lambda: {'before': [], 'after': []})
    for attempt in attempts:
        if attempt['logit_analysis'].get('entropy_stats') and attempt['logit_analysis']['entropy_stats'].get('mean'):
            entropy_means.append(attempt['logit_analysis']['entropy_stats']['mean'])
            entropy_stds.append(attempt['logit_analysis']['entropy_stats']['std'])
            if attempt['logit_analysis']['entropy_stats'].get('quartiles'):
                entropy_quartiles.append(attempt['logit_analysis']['entropy_stats']['quartiles'])
            if attempt['logit_analysis'].get('transition_entropy'):
                for phrase, stats in attempt['logit_analysis']['transition_entropy'].items():
                    if stats.get('before_mean') is not None:
                        transition_entropies[phrase]['before'].append(stats['before_mean'])
                    if stats.get('after_mean') is not None:
                        transition_entropies[phrase]['after'].append(stats['after_mean'])
    avg_quartiles = []
    if entropy_quartiles:
        max_quartiles = max((len(q) for q in entropy_quartiles))
        padded_quartiles = [q + [0] * (max_quartiles - len(q)) for q in entropy_quartiles]
        for i in range(max_quartiles):
            quartile_values = [q[i] for q in padded_quartiles if i < len(q)]
            avg_quartiles.append(statistics.mean(quartile_values) if quartile_values else 0)
    transition_stats = {}
    for phrase, values in transition_entropies.items():
        if values['before'] and values['after']:
            before_mean = statistics.mean(values['before'])
            after_mean = statistics.mean(values['after'])
            transition_stats[phrase] = {'before_mean': before_mean, 'after_mean': after_mean, 'entropy_change': after_mean - before_mean, 'count': len(values['before'])}
    return {'count': len(attempts), 'entropy': {'mean': statistics.mean(entropy_means) if entropy_means else 0, 'std': statistics.mean(entropy_stds) if entropy_stds else 0, 'quartiles': avg_quartiles}, 'transitions': transition_stats}

