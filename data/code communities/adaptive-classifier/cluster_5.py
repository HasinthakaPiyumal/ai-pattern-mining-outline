# Cluster 5

def load_and_split_dataset(args):
    """Load RAGTruth dataset and split according to specified parameters."""
    try:
        dataset = load_dataset('flowaicom/RAGTruth_test')
    except Exception as e:
        logger.error(f'Error loading dataset: {e}')
        logger.info('Trying alternative dataset ID...')
        try:
            dataset = load_dataset('RAGTruth/test')
        except Exception as e:
            logger.error(f'Error loading alternative dataset: {e}')
            raise ValueError('Failed to load RAGTruth dataset. Please check the dataset ID or your internet connection.')
    logger.info(f'Dataset loaded with structure: {dataset}')
    splits = {}
    if hasattr(dataset, 'keys') and callable(dataset.keys):
        task_types = list(dataset.keys())
        logger.info(f'Found task types in DatasetDict: {task_types}')
        for task_type in task_types:
            task_dataset = dataset[task_type]
            dataset_size = len(task_dataset)
            if args.use_all_data:
                logger.info(f'Using all {dataset_size} examples for both training and testing for task {task_type}')
                splits[task_type] = {'train': task_dataset, 'test': task_dataset}
                continue
            if args.train_count is not None:
                train_size = min(args.train_count, dataset_size)
                logger.info(f'Using fixed count of {train_size} examples for training for task {task_type}')
            else:
                train_size = int(dataset_size * (args.train_percentage / 100))
                if args.train_percentage > 0 and train_size == 0:
                    train_size = 1
                    logger.warning(f'Training percentage {args.train_percentage}% resulted in 0 examples for {task_type}. Using 1 example instead.')
                logger.info(f'Using {train_size} examples ({args.train_percentage}%) for training for task {task_type}')
            if train_size >= dataset_size:
                logger.warning(f'Training size {train_size} >= dataset size {dataset_size} for task {task_type}. Using same data for testing.')
                splits[task_type] = {'train': task_dataset, 'test': task_dataset}
                continue
            shuffled_data = task_dataset.shuffle(seed=args.seed)
            train_data = shuffled_data.select(range(train_size))
            test_data = shuffled_data.select(range(train_size, len(task_dataset)))
            splits[task_type] = {'train': train_data, 'test': test_data}
            logger.info(f'Task {task_type}: Calculated {train_size} training examples, actually got {len(train_data)}')
            logger.info(f'Task {task_type}: {len(train_data)} training examples, {len(test_data)} test examples')
    else:
        main_data = dataset
        task_types = set()
        for example in main_data:
            if 'task_type' in example and example['task_type']:
                task_types.add(example['task_type'])
        logger.info(f'Found task types in dataset: {task_types}')
        task_data = {}
        for task_type in task_types:
            task_data[task_type] = [ex for ex in main_data if ex.get('task_type') == task_type]
            logger.info(f'Task {task_type}: {len(task_data[task_type])} examples')
        for task_type, examples in task_data.items():
            dataset_size = len(examples)
            if args.use_all_data:
                logger.info(f'Using all {dataset_size} examples for both training and testing for task {task_type}')
                splits[task_type] = {'train': examples, 'test': examples}
                continue
            if args.train_count is not None:
                train_size = min(args.train_count, dataset_size)
                logger.info(f'Using fixed count of {train_size} examples for training for task {task_type}')
            else:
                train_size = int(dataset_size * (args.train_percentage / 100))
                if args.train_percentage > 0 and train_size == 0:
                    train_size = 1
                    logger.warning(f'Training percentage {args.train_percentage}% resulted in 0 examples for {task_type}. Using 1 example instead.')
                logger.info(f'Using {train_size} examples ({args.train_percentage}%) for training for task {task_type}')
            if train_size >= dataset_size:
                logger.warning(f'Training size {train_size} >= dataset size {dataset_size} for task {task_type}. Using same data for testing.')
                splits[task_type] = {'train': examples, 'test': examples}
                continue
            shuffled_examples = examples.copy() if hasattr(examples, 'copy') else list(examples)
            np.random.seed(args.seed)
            np.random.shuffle(shuffled_examples)
            if hasattr(shuffled_examples, 'select') and callable(getattr(shuffled_examples, 'select')):
                train_data = shuffled_examples.select(range(train_size))
                test_data = shuffled_examples.select(range(train_size, len(shuffled_examples)))
            else:
                train_data = shuffled_examples[:train_size]
                test_data = shuffled_examples[train_size:]
            splits[task_type] = {'train': train_data, 'test': test_data}
            logger.info(f'Task {task_type}: {len(train_data)} training examples, {len(test_data)} test examples')
    if not splits:
        logger.warning('No task types found in the dataset.')
    return splits

def prepare_examples(examples, task_type):
    """Prepare examples based on task type."""
    texts = []
    labels = []
    if hasattr(examples, '__iter__') and (not isinstance(examples, (dict, str))):
        iterator = examples
    else:
        iterator = [examples]
    for example in iterator:
        if hasattr(example, 'to_dict'):
            example = example[0] if isinstance(example, tuple) else example
            try:
                example = example.to_dict()
            except:
                example_dict = {}
                for key in example.keys():
                    example_dict[key] = example[key]
                example = example_dict
        elif not isinstance(example, dict):
            try:
                example = dict(example)
            except:
                logger.warning(f'Could not convert example to dictionary: {type(example)}')
                continue
        text = format_input_for_hallucination_detection(example, task_type)
        score = example.get('score', None)
        if not isinstance(score, (int, float)) and score is not None:
            try:
                score = float(score) if score else None
            except (ValueError, TypeError):
                score = None
        if score is not None:
            has_hallucination = score == 0
        else:
            logger.warning(f'No score found for example in task {task_type}, assuming no hallucination')
            has_hallucination = False
        texts.append(text)
        labels.append('HALLUCINATED' if has_hallucination else 'NOT_HALLUCINATED')
    hallucinated_count = sum((1 for label in labels if label == 'HALLUCINATED'))
    non_hallucinated_count = len(labels) - hallucinated_count
    logger.info(f'Prepared {len(texts)} examples: {hallucinated_count} HALLUCINATED, {non_hallucinated_count} NOT_HALLUCINATED')
    return (texts, labels)

def format_input_for_hallucination_detection(example, task_type):
    """Format the input for hallucination detection based on task type."""
    if not isinstance(example, dict):
        example = dict(example)
    source_info = example.get('source_info', '')
    prompt = example.get('prompt', '')
    response = example.get('response', '')
    if task_type == 'qa':
        return f'Context: {source_info}\nQuestion: {prompt}\nAnswer: {response}'
    elif task_type == 'data2text':
        return f'Context: {source_info}\nGenerated Text: {response}'
    elif task_type == 'summarization':
        return f'Article: {source_info}\nSummary: {response}'
    else:
        return f'Source: {source_info}\nPrompt: {prompt}\nResponse: {response}'

def train_and_evaluate(args):
    """Train and evaluate the hallucination detector."""
    logger.info(f'Loading dataset with training parameters:')
    if args.use_all_data:
        logger.info('  - Using all data for both training and testing')
    elif args.train_count is not None:
        logger.info(f'  - Using fixed count of {args.train_count} examples for training')
    else:
        logger.info(f'  - Using {args.train_percentage}% of data for training')
    splits = load_and_split_dataset(args)
    if not splits:
        logger.error('No data found in dataset. Please check the dataset structure.')
        return {}
    logger.info(f'Initializing adaptive classifier with {args.model_name}')
    config = {'max_length': 2048, 'batch_size': 4, 'max_examples_per_class': 100, 'prototype_update_frequency': 10, 'similarity_threshold': 0.6, 'learning_rate': 2e-05, 'epochs': 5, 'early_stopping_patience': 2, 'prototype_weight': 0.7, 'neural_weight': 0.3}
    classifier = AdaptiveClassifier(args.model_name, config=config)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    task_stats = {}
    for task_type in splits.keys():
        logger.info(f'Processing task: {task_type}')
        train_texts, train_labels = prepare_examples(splits[task_type]['train'], task_type)
        test_texts, test_labels = prepare_examples(splits[task_type]['test'], task_type)
        if len(train_texts) == 0 or len(test_texts) == 0:
            logger.warning(f'No data for task {task_type}. Skipping.')
            continue
        elif len(train_texts) < args.min_examples or len(test_texts) < args.min_examples:
            logger.warning(f'Not enough data for task {task_type}. Minimum required: {args.min_examples}. Found: {len(train_texts)} train, {len(test_texts)} test. Skipping.')
            continue
        train_positive = sum((1 for label in train_labels if label == 'HALLUCINATED'))
        train_negative = len(train_labels) - train_positive
        test_positive = sum((1 for label in test_labels if label == 'HALLUCINATED'))
        test_negative = len(test_labels) - test_positive
        logger.info(f'Train distribution for {task_type}: {train_positive} HALLUCINATED, {train_negative} NOT_HALLUCINATED')
        logger.info(f'Test distribution for {task_type}: {test_positive} HALLUCINATED, {test_negative} NOT_HALLUCINATED')
        logger.info(f'Training on {len(train_texts)} examples for task {task_type}')
        batch_size = args.batch_size
        for i in range(0, len(train_texts), batch_size):
            end_idx = min(i + batch_size, len(train_texts))
            batch_texts = train_texts[i:end_idx]
            batch_labels = train_labels[i:end_idx]
            classifier.add_examples(batch_texts, batch_labels)
            if i // batch_size % 10 == 0 and i > 0:
                logger.info(f'Added {i + len(batch_texts)} examples so far...')
        logger.info(f'Evaluating on {len(test_texts)} examples for task {task_type}')
        metrics = evaluate_classifier(classifier, test_texts, test_labels)
        true_positives = sum((1 for true, pred in zip(metrics['true_labels'], metrics['predictions']) if true == 'HALLUCINATED' and pred == 'HALLUCINATED'))
        false_positives = sum((1 for true, pred in zip(metrics['true_labels'], metrics['predictions']) if true == 'NOT_HALLUCINATED' and pred == 'HALLUCINATED'))
        true_negatives = sum((1 for true, pred in zip(metrics['true_labels'], metrics['predictions']) if true == 'NOT_HALLUCINATED' and pred == 'NOT_HALLUCINATED'))
        false_negatives = sum((1 for true, pred in zip(metrics['true_labels'], metrics['predictions']) if true == 'HALLUCINATED' and pred == 'NOT_HALLUCINATED'))
        confusion_matrix = {'true_positives': true_positives, 'false_positives': false_positives, 'true_negatives': true_negatives, 'false_negatives': false_negatives}
        task_stats[task_type] = {'precision': metrics['precision'], 'recall': metrics['recall'], 'f1': metrics['f1'], 'throughput': metrics['throughput'], 'num_train_examples': len(train_texts), 'num_test_examples': len(test_texts), 'confusion_matrix': confusion_matrix}
        logger.info(f'Task: {task_type} - Precision: {metrics['precision']:.2f}%, Recall: {metrics['recall']:.2f}%, F1: {metrics['f1']:.2f}%')
        logger.info(f'Throughput: {metrics['throughput']:.2f} examples/second')
        logger.info(f'Confusion Matrix: TP={true_positives}, FP={false_positives}, TN={true_negatives}, FN={false_negatives}')
        if 'predictions' in metrics:
            del metrics['predictions']
        if 'true_labels' in metrics:
            del metrics['true_labels']
    if not task_stats:
        logger.warning('No task statistics available. Skipping overall metrics calculation.')
        return {'metadata': {'train_percentage': args.train_percentage, 'model_name': args.model_name, 'batch_size': args.batch_size, 'seed': args.seed, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'error': 'No data processed successfully'}}
    total_test = sum((stats['num_test_examples'] for stats in task_stats.values()))
    if total_test > 0:
        overall_precision = sum((stats['precision'] * stats['num_test_examples'] / total_test for stats in task_stats.values()))
        overall_recall = sum((stats['recall'] * stats['num_test_examples'] / total_test for stats in task_stats.values()))
        overall_f1 = sum((stats['f1'] * stats['num_test_examples'] / total_test for stats in task_stats.values()))
        overall_throughput = sum((stats['throughput'] * stats['num_test_examples'] / total_test for stats in task_stats.values()))
        overall_tp = sum((stats['confusion_matrix']['true_positives'] for stats in task_stats.values()))
        overall_fp = sum((stats['confusion_matrix']['false_positives'] for stats in task_stats.values()))
        overall_tn = sum((stats['confusion_matrix']['true_negatives'] for stats in task_stats.values()))
        overall_fn = sum((stats['confusion_matrix']['false_negatives'] for stats in task_stats.values()))
        task_stats['overall'] = {'precision': overall_precision, 'recall': overall_recall, 'f1': overall_f1, 'throughput': overall_throughput, 'confusion_matrix': {'true_positives': overall_tp, 'false_positives': overall_fp, 'true_negatives': overall_tn, 'false_negatives': overall_fn}}
        logger.info(f'Overall - Precision: {overall_precision:.2f}%, Recall: {overall_recall:.2f}%, F1: {overall_f1:.2f}%')
        logger.info(f'Overall Throughput: {overall_throughput:.2f} examples/second')
        logger.info(f'Overall Confusion Matrix: TP={overall_tp}, FP={overall_fp}, TN={overall_tn}, FN={overall_fn}')
    task_stats['metadata'] = {'train_percentage': args.train_percentage, 'model_name': args.model_name, 'batch_size': args.batch_size, 'seed': args.seed, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    logger.info(f'Saving model to {save_dir}')
    classifier.save(save_dir)
    metrics_file = save_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(task_stats, f, indent=2)
    logger.info(f'Metrics saved to {metrics_file}')
    if args.push_to_hub:
        repo_id = f'adaptive-classifier/{args.push_to_hub}'
        logger.info(f'Pushing model to HuggingFace Hub: {repo_id}')
        try:
            classifier.push_to_hub(repo_id)
            logger.info(f'Successfully pushed to {repo_id}')
        except Exception as e:
            logger.error(f'Error pushing to HuggingFace Hub: {e}')
            logger.info('Trying alternative push method...')
            try:
                from huggingface_hub import upload_folder
                upload_folder(folder_path=str(save_dir), repo_id=repo_id, repo_type='model')
                logger.info(f'Successfully pushed to {repo_id} using upload_folder')
            except Exception as e:
                logger.error(f'Error using alternative push method: {e}')
    return task_stats

def main():
    args = parse_args()
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    set_seeds(args.seed)
    try:
        stats = train_and_evaluate(args)
        if not stats or not any((key not in ['metadata'] for key in stats.keys())):
            logger.warning('No statistics available. Cannot generate report.')
            return
        print('\nResults Summary:')
        print('=' * 80)
        print(f'{'Task':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Throughput':<12}')
        print('-' * 80)
        for task_type, metrics in stats.items():
            if task_type not in ['overall', 'metadata']:
                print(f'{task_type:<15} {metrics['precision']:<12.2f} {metrics['recall']:<12.2f} {metrics['f1']:<12.2f} {metrics['throughput']:<12.2f}')
        print('-' * 80)
        if 'overall' in stats:
            print(f'{'overall':<15} {stats['overall']['precision']:<12.2f} {stats['overall']['recall']:<12.2f} {stats['overall']['f1']:<12.2f} {stats['overall']['throughput']:<12.2f}')
        print('=' * 80)
        print_comparison_table(stats, args)
    except Exception as e:
        logger.error(f'Error during execution: {e}')
        raise

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and evaluate a hallucination detector using adaptive-classifier')
    parser.add_argument('--train-percentage', type=float, default=20.0, help='Percentage of dataset to use for training (default: 20%%)')
    parser.add_argument('--train-count', type=int, default=None, help='Fixed number of examples to use for training (overrides --train-percentage if specified)')
    parser.add_argument('--use-all-data', action='store_true', help='Use all available data for training and testing (overrides other training parameters)')
    parser.add_argument('--model-name', type=str, default='distilbert-base-uncased', help='Hugging Face model name to use as the base model')
    parser.add_argument('--save-dir', type=str, default='./hallucination-detector', help='Directory to save the trained model')
    parser.add_argument('--push-to-hub', type=str, help='Name to use when pushing to HuggingFace Hub under adaptive-classifier/')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--min-examples', type=int, default=0, help='Minimum number of examples required per task (default: 0, use all available data)')
    parser.add_argument('--verbose', action='store_true', help='Enable more detailed logging')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help='Set the logging level')
    return parser.parse_args()

def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_comparison_table(stats, args):
    """Print a comparison table similar to the one in the paper."""
    print('\nRESULTS COMPARISON WITH PAPER')
    print('=' * 80)
    print(f'{'TASK':<15} {'METHOD':<25} {'PRECISION':<10} {'RECALL':<10} {'F1':<10}')
    print('-' * 80)
    paper_baselines = {'qa': [('Luna (paper)', 37.8, 80.0, 51.3), ('LettuceDetect-large', 65.93, 75.0, 70.18)], 'data2txt': [('Luna (paper)', 64.9, 91.2, 75.9), ('LettuceDetect-large', 90.45, 86.7, 88.54)], 'summarization': [('Luna (paper)', 40.0, 76.5, 52.5), ('LettuceDetect-large', 64.0, 55.88, 59.69)]}
    overall_paper_results = [('Luna (paper)', 52.7, 86.1, 65.4), ('LettuceDetect-large', 80.44, 78.05, 79.22)]
    tasks = [task for task in stats.keys() if task not in ['overall', 'metadata']]
    for task in tasks:
        print(f'{task:<15} {'Our Model':<25} {stats[task]['precision']:<10.2f} {stats[task]['recall']:<10.2f} {stats[task]['f1']:<10.2f}')
        if task.lower() in paper_baselines:
            for method, precision, recall, f1 in paper_baselines[task.lower()]:
                print(f'{'':<15} {method:<25} {precision:<10.1f} {recall:<10.1f} {f1:<10.1f}')
        else:
            print(f'{'':<15} {'(No paper baseline)':<25} {'-':<10} {'-':<10} {'-':<10}')
    print('-' * 80)
    if 'overall' in stats:
        print(f'{'Overall':<15} {'Our Model':<25} {stats['overall']['precision']:<10.2f} {stats['overall']['recall']:<10.2f} {stats['overall']['f1']:<10.2f}')
    else:
        print(f'{'Overall':<15} {'Our Model':<25} {'-':<10} {'-':<10} {'-':<10}')
    for method, precision, recall, f1 in overall_paper_results:
        print(f'{'':<15} {method:<25} {precision:<10.1f} {recall:<10.1f} {f1:<10.1f}')
    print('=' * 80)
    print(f'Training with {args.train_percentage}% of data using model: {args.model_name}')
    if 'overall' in stats:
        print(f'Throughput: {stats['overall']['throughput']:.2f} examples/second')

