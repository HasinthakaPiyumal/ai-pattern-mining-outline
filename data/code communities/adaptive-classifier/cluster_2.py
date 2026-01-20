# Cluster 2

def save_results(classifier: AdaptiveClassifier, results: Dict[str, Any], args: argparse.Namespace):
    """Save evaluation results and optionally push to Hub."""
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'benchmark_results_{timestamp}.json'
    filepath = os.path.join(args.output_dir, filename)
    results['config'] = {'model': args.model, 'batch_size': args.batch_size, 'max_samples': args.max_samples, 'timestamp': timestamp}
    classifier.save(args.output_dir)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Results saved to {filepath}')
    if args.push_to_hub:
        if not args.hub_repo:
            raise ValueError('--hub-repo must be specified when using --push-to-hub')
        hub_url = push_to_hub(classifier, args.hub_repo, args.hub_token, metrics=results['metrics'])
        results['hub_url'] = hub_url
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    print('\nEvaluation Results:')
    print('-' * 50)
    print(f'Model: {args.model}')
    print(f'Accuracy: {results['metrics']['accuracy']:.4f}')
    print('\nPer-class metrics:')
    for label in ['HIGH', 'LOW']:
        metrics = results['metrics'][label]
        print(f'\n{label}:')
        print(f'  Precision: {metrics['precision']:.4f}')
        print(f'  Recall: {metrics['recall']:.4f}')
        print(f'  F1-score: {metrics['f1-score']:.4f}')
    print('\nConfusion Matrix:')
    print('            Predicted')
    print('             HIGH  LOW')
    print(f'Actual HIGH  {results['confusion_matrix'][0][0]:4d}  {results['confusion_matrix'][0][1]:4d}')
    print(f'      LOW   {results['confusion_matrix'][1][0]:4d}  {results['confusion_matrix'][1][1]:4d}')
    if args.push_to_hub:
        print(f'\nModel pushed to HuggingFace Hub: {results['hub_url']}')

def push_to_hub(classifier: AdaptiveClassifier, repo_id: str, token: str=None, metrics: Dict[str, Any]=None) -> str:
    """Push the classifier to HuggingFace Hub.
    
    Args:
        classifier: Trained classifier to push
        repo_id: HuggingFace Hub repository ID
        token: HuggingFace Hub token
        metrics: Optional evaluation metrics to add to model card
        
    Returns:
        URL of the model on the Hub
    """
    logger.info(f'Pushing model to HuggingFace Hub: {repo_id}')
    if token:
        HfFolder.save_token(token)
    try:
        url = classifier.push_to_hub(repo_id, commit_message='Upload from benchmark script')
        logger.info(f'Successfully pushed model to Hub: {url}')
        return url
    except Exception as e:
        logger.error(f'Error pushing to Hub: {str(e)}')
        raise

def main():
    """Main execution function."""
    args = setup_args()
    torch.manual_seed(42)
    train_dataset, val_dataset = load_dataset(args.max_samples)
    classifier = train_classifier(args.model, train_dataset, args.batch_size)
    results = evaluate_classifier(classifier, val_dataset, args.batch_size)
    save_results(classifier, results, args)

def setup_args() -> argparse.Namespace:
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmark Adaptive Classifier')
    parser.add_argument('--model', type=str, default='distilbert/distilbert-base-cased', help='Base transformer model to use')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--max-samples', type=int, default=1200, help='Maximum number of samples to use (for testing)')
    parser.add_argument('--output-dir', type=str, default='benchmark_results', help='Directory to save results')
    parser.add_argument('--push-to-hub', action='store_true', help='Push the trained model to HuggingFace Hub')
    parser.add_argument('--hub-repo', type=str, help='HuggingFace Hub repository ID (e.g. "username/model-name") for pushing the model')
    parser.add_argument('--hub-token', type=str, help='HuggingFace Hub token. If not provided, will look for the token in the environment')
    return parser.parse_args()

def load_dataset(max_samples: int=None) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """Load and preprocess the dataset."""
    logger.info('Loading routellm/gpt4_dataset...')
    dataset = datasets.load_dataset('routellm/gpt4_dataset')

    def preprocess_function(example: Dict[str, Any]) -> Dict[str, Any]:
        """Convert scores to binary labels."""
        score = example['mixtral_score']
        label = 'LOW' if score >= 4 else 'HIGH'
        return {'text': example['prompt'], 'label': label}
    train_dataset = dataset['train'].map(preprocess_function)
    val_dataset = dataset['validation'].map(preprocess_function)
    if max_samples:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(max_samples, len(val_dataset))))
    logger.info(f'Dataset loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}')
    return (train_dataset, val_dataset)

def train_classifier(model_name: str, train_dataset: datasets.Dataset, batch_size: int) -> AdaptiveClassifier:
    """Train the adaptive classifier with improved balancing and configuration."""
    logger.info(f'Initializing classifier with model: {model_name}')
    labels = train_dataset['label']
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    logger.info(f'Original class distribution: {label_counts}')
    total_samples = sum(label_counts.values())
    class_weights = {label: total_samples / (len(label_counts) * count) for label, count in label_counts.items()}
    logger.info(f'Class weights: {class_weights}')
    classifier = AdaptiveClassifier(model_name, device='cuda' if torch.cuda.is_available() else 'cpu', config={'batch_size': batch_size, 'max_examples_per_class': 500, 'prototype_update_frequency': 50, 'learning_rate': 0.0005, 'similarity_threshold': 0.7, 'prototype_weight': 0.8, 'neural_weight': 0.2})
    texts = train_dataset['text']
    examples_by_label = {label: [] for label in label_counts.keys()}
    for text, label in zip(texts, labels):
        examples_by_label[label].append(text)
    min_class_size = min((len(examples) for examples in examples_by_label.values()))
    balanced_texts = []
    balanced_labels = []
    for label, examples in examples_by_label.items():
        if len(examples) < min_class_size * 2:
            sampled_examples = random.choices(examples, k=min_class_size * 2)
        else:
            sampled_examples = random.sample(examples, min_class_size * 2)
        balanced_texts.extend(sampled_examples)
        balanced_labels.extend([label] * len(sampled_examples))
    combined = list(zip(balanced_texts, balanced_labels))
    random.Random(42).shuffle(combined)
    balanced_texts, balanced_labels = zip(*combined)
    total_batches = (len(balanced_texts) + batch_size - 1) // batch_size
    logger.info(f'Total batches: {total_batches}')
    for i in tqdm(range(0, len(balanced_texts), batch_size), total=total_batches):
        try:
            batch_texts = balanced_texts[i:i + batch_size]
            batch_labels = balanced_labels[i:i + batch_size]
            if i % (batch_size * 10) == 0:
                logger.debug(f'Batch {i // batch_size + 1}/{total_batches}')
                label_counts = {label: batch_labels.count(label) for label in set(batch_labels)}
                logger.debug(f'Batch class distribution: {label_counts}')
            classifier.add_examples(batch_texts, batch_labels)
        except Exception as e:
            logger.error(f'Error in batch {i // batch_size + 1}')
            logger.error(str(e))
            raise
    memory_stats = classifier.get_memory_stats()
    logger.info(f'Final memory stats: {memory_stats}')
    return classifier

def evaluate_dataset(config: RouterConfig, enable_adaptation: bool, output_file: str):
    """Evaluate the dataset using the LLM router."""
    router = LLMRouter(config, enable_adaptation=enable_adaptation)
    dataset = load_dataset('lmarena-ai/arena-hard-auto-v0.1')
    results = []
    for item in tqdm(dataset['train'], desc='Evaluating examples'):
        query = extract_first_turn_content(item['turns'])
        if not query:
            continue
        passed_rtc, evaluation_result = router.route_and_evaluate(query)
        results.append(evaluation_result)
        save_results(output_file, router, results)
    if enable_adaptation:
        router.save_classifier()
    print_summary(router)

def extract_first_turn_content(turns: List[Dict]) -> str:
    """Extract content from first turn in conversation."""
    if not turns or not isinstance(turns, list):
        return ''
    return turns[0].get('content', '')

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM router on arena-hard-auto dataset')
    parser.add_argument('--high-model', type=str, default='gpt-4o', help='Model to use for high-complexity queries')
    parser.add_argument('--low-model', type=str, default='gpt-4o-mini', help='Model to use for low-complexity queries')
    parser.add_argument('--without-adaptation', action='store_true', help='Disable adaptive learning during evaluation')
    parser.add_argument('--output', type=str, default='router_eval_results.json', help='Output file for results')
    parser.add_argument('--router-path', type=str, default='./adaptive_router', help='Path to load/save the adaptive router')
    args = parser.parse_args()
    os.makedirs('benchmark_results', exist_ok=True)
    output_file = os.path.join('benchmark_results', args.output)
    config = RouterConfig(high_model=args.high_model, low_model=args.low_model, adaptive_router_path=args.router_path)
    evaluate_dataset(config, enable_adaptation=not args.without_adaptation, output_file=output_file)

def load_adv_glue_dataset() -> Tuple[List[str], List[str]]:
    """Load the AI-Secure/adv_glue dataset (adv_sst2 subset, validation split).
    
    Returns:
        Tuple of (texts, labels) where labels are converted to string format
    """
    logger.info('Loading AI-Secure/adv_glue dataset (adv_sst2 subset)...')
    try:
        dataset = load_dataset('AI-Secure/adv_glue', 'adv_sst2', split='validation')
        texts = dataset['sentence']
        labels = dataset['label']
        label_map = {0: 'negative', 1: 'positive'}
        labels = [label_map[label] for label in labels]
        logger.info(f'Loaded {len(texts)} examples')
        logger.info(f'Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}')
        return (texts, labels)
    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        raise

class ConfigOptimizer:
    """Optimizer class to find best temperature configurations for queries."""

    def __init__(self, training_config: TrainingConfig):
        """Initialize the optimizer."""
        self.training_config = training_config
        self.classifier = AdaptiveClassifier('distilbert-base-uncased')
        self.evaluator = ResponseEvaluator()
        self.stats = {'total_queries': 0, 'successful_optimizations': 0, 'failed_optimizations': 0, 'avg_similarity_score': 0.0, 'class_distribution': {class_name: 0 for class_name in TemperatureConfig.class_ranges.keys()}, 'detailed_scores': []}

    def find_best_temperature_class(self, query: str) -> Tuple[Optional[str], float, Dict[str, float]]:
        """Find best temperature class for a query."""
        best_score = -1
        best_class = None
        best_metrics = {}
        configs_tested = 0
        for class_name in TemperatureConfig.class_ranges.keys():
            class_temps = TemperatureConfig.get_temperatures_for_class(class_name)
            for temp in class_temps:
                configs_tested += 1
                score, metrics = self._evaluate_temperature(query, temp)
                logger.debug(f'Temperature {temp:.1f} achieved score {score:.3f}')
                if score > best_score:
                    best_score = score
                    best_class = class_name
                    best_metrics = metrics
                if best_score >= 0.8:
                    break
            if best_score >= 0.8:
                break
        logger.info(f'Tested {configs_tested} configurations for query')
        return (best_class, best_score, best_metrics)

    def _evaluate_temperature(self, query: str, temperature: float) -> Tuple[float, Dict[str, float]]:
        """Evaluate a temperature setting using RTC."""
        config = {'temperature': temperature, 'top_p': 1.0}
        response_1 = self._get_llm_response(query, config)
        if not response_1:
            return (0.0, {})
        inverse_prompt = f'Given this query and response pair, generate a new query that would lead to a similar response:\n\nOriginal Query: {query}\nResponse: {response_1}\n\nGenerate a new query that would elicit a similar response:'
        alternate_query = self._get_llm_response(inverse_prompt, config)
        if not alternate_query:
            return (0.0, {})
        response_2 = self._get_llm_response(alternate_query, config)
        if not response_2:
            return (0.0, {})
        similarity_score, metrics = self.evaluator.evaluate_responses(response_1, response_2)
        metrics['temperature'] = temperature
        return (similarity_score, metrics)

    def _get_llm_response(self, prompt: str, config: Dict[str, float]) -> Optional[str]:
        """Get response from the LLM with improved error handling."""
        messages = [{'role': 'user', 'content': prompt}]
        for attempt in range(self.training_config.max_retries):
            try:
                response = client.chat.completions.create(model=self.training_config.model, messages=messages, max_tokens=4096, **config)
                if not response or not hasattr(response, 'choices') or (not response.choices):
                    logger.warning(f'Invalid response structure (attempt {attempt + 1})')
                    continue
                content = response.choices[0].message.content
                if not content or not isinstance(content, str):
                    logger.warning(f'Invalid content (attempt {attempt + 1})')
                    continue
                return content.strip()
            except Exception as e:
                logger.error(f'Error getting LLM response (attempt {attempt + 1}): {e}')
            if attempt < self.training_config.max_retries - 1:
                sleep_time = self.training_config.retry_delay * 2 ** attempt
                time.sleep(sleep_time)
        return None

    def optimize_and_train(self, save_path: str, push_to_hub: str):
        """Run optimization and training process."""
        try:
            dataset = load_dataset('lmarena-ai/arena-hard-auto-v0.1')
            logger.info('Successfully loaded dataset')
        except Exception as e:
            logger.error(f'Error loading dataset: {e}')
            return
        logger.info(f'Starting optimization for model: {self.training_config.model}')
        successful_examples = []
        for i in tqdm(range(0, min(len(dataset['train']), self.training_config.max_examples), self.training_config.batch_size)):
            batch = dataset['train'][i:i + self.training_config.batch_size]
            for item in batch:
                query = item['text'] if isinstance(item, dict) else str(item)
                self.stats['total_queries'] += 1
                best_class, score, metrics = self.find_best_temperature_class(query)
                if best_class and score >= self.training_config.similarity_threshold:
                    successful_examples.append((query, best_class))
                    self.stats['successful_optimizations'] += 1
                    self.stats['avg_similarity_score'] = (self.stats['avg_similarity_score'] * (len(successful_examples) - 1) + score) / len(successful_examples)
                    self.stats['class_distribution'][best_class] += 1
                    self.stats['detailed_scores'].append({'query': query, 'class': best_class, 'score': score, 'metrics': metrics})
                else:
                    self.stats['failed_optimizations'] += 1
                if self.stats['total_queries'] % 50 == 0:
                    self._print_intermediate_stats()
            if successful_examples:
                queries, labels = zip(*successful_examples)
                self.classifier.add_examples(list(queries), list(labels))
                successful_examples = []
        self._save_results(save_path)
        if push_to_hub:
            repo_id = f'adaptive-classifier/{push_to_hub}'
            logger.info(f'\nPushing to HuggingFace Hub: {repo_id}')
            try:
                self.classifier.push_to_hub(repo_id)
                logger.info('Successfully pushed to HuggingFace Hub')
            except Exception as e:
                logger.error(f'Error pushing to HuggingFace Hub: {e}')
        self._print_final_stats()

    def _print_intermediate_stats(self):
        """Print intermediate statistics."""
        logger.info('\nIntermediate Statistics:')
        logger.info(f'Processed queries: {self.stats['total_queries']}')
        logger.info(f'Successful optimizations: {self.stats['successful_optimizations']}')
        success_rate = self.stats['successful_optimizations'] / self.stats['total_queries'] * 100
        logger.info(f'Current success rate: {success_rate:.2f}%')
        logger.info(f'Average similarity score: {self.stats['avg_similarity_score']:.3f}')
        logger.info('\nClass distribution:')
        for class_name, count in self.stats['class_distribution'].items():
            if count > 0:
                percentage = count / self.stats['successful_optimizations'] * 100
                logger.info(f'{class_name}: {count} ({percentage:.1f}%)')

    def _print_final_stats(self):
        """Print final detailed statistics."""
        logger.info('\nFinal Statistics:')
        logger.info(f'Total queries processed: {self.stats['total_queries']}')
        logger.info(f'Successful optimizations: {self.stats['successful_optimizations']}')
        logger.info(f'Failed optimizations: {self.stats['failed_optimizations']}')
        if self.stats['successful_optimizations'] > 0:
            success_rate = self.stats['successful_optimizations'] / self.stats['total_queries'] * 100
            logger.info(f'Success rate: {success_rate:.2f}%')
            logger.info(f'Average similarity score: {self.stats['avg_similarity_score']:.3f}')
            logger.info('\nTemperature Class Distribution:')
            for class_name, count in sorted(self.stats['class_distribution'].items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    percentage = count / self.stats['successful_optimizations'] * 100
                    logger.info(f'{class_name}: {count} ({percentage:.1f}%)')
            logger.info('\nAverage Scores by Class:')
            class_scores = {}
            for result in self.stats['detailed_scores']:
                class_name = result['class']
                if class_name not in class_scores:
                    class_scores[class_name] = []
                class_scores[class_name].append(result['score'])
            for class_name, scores in class_scores.items():
                avg_score = sum(scores) / len(scores)
                logger.info(f'{class_name}: {avg_score:.3f}')

    def _save_results(self, save_path: str):
        """Save classifier and statistics."""
        self.classifier.save(save_path)
        stats_path = Path(save_path) / 'optimization_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f'\nResults saved to {save_path}')

