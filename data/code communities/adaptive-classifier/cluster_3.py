# Cluster 3

def evaluate_classifier(classifier: AdaptiveClassifier, val_dataset: datasets.Dataset, batch_size: int) -> Dict[str, Any]:
    """Evaluate the classifier."""
    logger.info('Starting evaluation...')
    predictions = []
    true_labels = val_dataset['label']
    texts = val_dataset['text']
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(texts), batch_size), total=total_batches):
        batch_texts = texts[i:i + batch_size]
        batch_predictions = classifier.predict_batch(batch_texts, k=1)
        predictions.extend([pred[0][0] for pred in batch_predictions])
    report = classification_report(true_labels, predictions, output_dict=True)
    conf_matrix = confusion_matrix(true_labels, predictions).tolist()
    memory_stats = classifier.get_memory_stats()
    example_stats = classifier.get_example_statistics()
    results = {'metrics': report, 'confusion_matrix': conf_matrix, 'memory_stats': memory_stats, 'example_stats': example_stats}
    return results

def convert_to_serializable(obj):
    """Convert numpy/torch types to JSON serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def evaluate_strategic_robustness(strategic_classifier: AdaptiveClassifier, test_texts: List[str], test_labels: List[str]) -> Dict[str, Any]:
    """Evaluate strategic robustness at different gaming levels.
    
    Args:
        strategic_classifier: Classifier with strategic capabilities
        test_texts: Test texts  
        test_labels: Test labels
        
    Returns:
        Dictionary of robustness metrics
    """
    if not strategic_classifier.strategic_mode:
        logger.error('Strategic mode not enabled - cannot evaluate robustness')
        raise RuntimeError('Strategic mode not enabled - cannot evaluate robustness')
    logger.info('Evaluating strategic robustness...')
    gaming_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    robustness_results = {}
    for level in gaming_levels:
        logger.info(f'Testing gaming level: {level}')
        if level == 0.0:
            results = evaluate_classifier(strategic_classifier, test_texts, test_labels, mode='regular')
        elif level == 1.0:
            results = evaluate_classifier(strategic_classifier, test_texts, test_labels, mode='strategic')
        else:
            results = evaluate_classifier(strategic_classifier, test_texts, test_labels, mode='dual')
        robustness_results[f'gaming_level_{level}'] = {'accuracy': results['accuracy'], 'f1_score': results['f1_score'], 'avg_confidence': results['avg_confidence']}
    baseline_accuracy = robustness_results['gaming_level_0.0']['accuracy']
    strategic_accuracy = robustness_results['gaming_level_1.0']['accuracy']
    robustness_score = baseline_accuracy - strategic_accuracy
    relative_robustness = strategic_accuracy / baseline_accuracy if baseline_accuracy > 0 else 0.0
    robustness_results['summary'] = {'baseline_accuracy': baseline_accuracy, 'strategic_accuracy': strategic_accuracy, 'robustness_score': robustness_score, 'relative_robustness': relative_robustness}
    logger.info(f'Robustness Score: {robustness_score:.4f}')
    logger.info(f'Relative Robustness: {relative_robustness:.4f}')
    return robustness_results

def main():
    parser = argparse.ArgumentParser(description='Evaluate Strategic Classifier on AI-Secure/adv_glue dataset')
    parser.add_argument('--model', type=str, default='answerdotai/ModernBERT-base', help="HuggingFace model name to use (default: answerdotai/ModernBERT-base). The script automatically adapts to any model's embedding dimension.")
    parser.add_argument('--cost-strategy', type=str, default='balanced', choices=['balanced', 'sparse_low', 'uniform_low', 'minimal', 'sparse_high'], help="Strategic cost function strategy. Options: 'balanced' (50%% manipulable dims, cost 0.3), 'sparse_low' (20%% manipulable dims, cost 0.4), 'uniform_low' (all dims, cost 0.15), 'minimal' (all dims, cost 0.05 - for debugging), 'sparse_high' (legacy - adjusted to be less restrictive)")
    parser.add_argument('--output', type=str, default='strategic_classifier_evaluation_results.json', help='Output file for results (default: strategic_classifier_evaluation_results.json)')
    parser.add_argument('--test-size', type=float, default=0.3, help='Fraction of data to use for testing (default: 0.3)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()
    set_seed(args.seed)
    logger.info('Starting Strategic Classifier Evaluation')
    logger.info(f'Model: {args.model}')
    logger.info(f'Cost strategy: {args.cost_strategy}')
    logger.info(f'Output file: {args.output}')
    logger.info(f'Test size: {args.test_size}')
    logger.info(f'Random seed: {args.seed}')
    start_time = time.time()
    try:
        texts, labels = load_adv_glue_dataset()
        train_texts, test_texts, train_labels, test_labels = split_dataset(texts, labels, test_size=args.test_size, random_state=args.seed)
        logger.info('=' * 60)
        logger.info('TRAINING REGULAR CLASSIFIER')
        logger.info('=' * 60)
        regular_classifier = train_classifier(args.model, train_texts, train_labels, config=None)
        logger.info('=' * 60)
        logger.info('TRAINING STRATEGIC CLASSIFIER')
        logger.info('=' * 60)
        strategic_config = create_strategic_config(args.model, args.cost_strategy)
        strategic_classifier = train_classifier(args.model, train_texts, train_labels, config=strategic_config)
        if not strategic_classifier.strategic_mode:
            raise RuntimeError('Strategic mode failed to initialize. Cannot proceed with strategic evaluation.')
        logger.info('=' * 60)
        logger.info('EVALUATION PHASE')
        logger.info('=' * 60)
        regular_results = evaluate_classifier(regular_classifier, test_texts, test_labels, mode='regular')
        strategic_dual_results = evaluate_classifier(strategic_classifier, test_texts, test_labels, mode='dual')
        strategic_only_results = evaluate_classifier(strategic_classifier, test_texts, test_labels, mode='strategic')
        robust_results = evaluate_classifier(strategic_classifier, test_texts, test_labels, mode='robust')
        comparison_results = evaluate_comparison_on_manipulated_data(regular_classifier, strategic_classifier, test_texts, test_labels)
        robustness_results = evaluate_strategic_robustness(strategic_classifier, test_texts, test_labels)
        end_time = time.time()
        total_time = end_time - start_time
        final_results = {'metadata': {'model_name': args.model, 'dataset': 'AI-Secure/adv_glue (adv_sst2)', 'evaluation_date': datetime.now().isoformat(), 'total_examples': len(texts), 'train_examples': len(train_texts), 'test_examples': len(test_texts), 'test_size': args.test_size, 'random_seed': args.seed, 'total_evaluation_time': total_time}, 'dataset_info': {'train_label_distribution': dict(zip(*np.unique(train_labels, return_counts=True))), 'test_label_distribution': dict(zip(*np.unique(test_labels, return_counts=True)))}, 'regular_classifier': regular_results, 'strategic_classifier': {'dual_mode': strategic_dual_results, 'strategic_only_mode': strategic_only_results, 'robust_mode': robust_results}, 'strategic_robustness': robustness_results, 'fair_comparison_on_manipulated_data': comparison_results, 'comparison': {'accuracy_improvement': strategic_dual_results['accuracy'] - regular_results['accuracy'], 'f1_improvement': strategic_dual_results['f1_score'] - regular_results['f1_score'], 'relative_accuracy_improvement': (strategic_dual_results['accuracy'] - regular_results['accuracy']) / regular_results['accuracy'] if regular_results['accuracy'] > 0 else 0.0}, 'config': {'strategic_config': strategic_config, 'cost_strategy': args.cost_strategy}}
        output_path = Path(args.output)
        serializable_results = convert_to_serializable(final_results)
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, sort_keys=True)
        logger.info('=' * 60)
        logger.info('EVALUATION SUMMARY')
        logger.info('=' * 60)
        logger.info(f'Regular Classifier Accuracy: {regular_results['accuracy']:.4f}')
        logger.info(f'Strategic Classifier (Dual) Accuracy: {strategic_dual_results['accuracy']:.4f}')
        logger.info(f'Strategic Classifier (Strategic-only) Accuracy: {strategic_only_results['accuracy']:.4f}')
        logger.info(f'Strategic Classifier (Robust) Accuracy: {robust_results['accuracy']:.4f}')
        logger.info('')
        logger.info(f'Accuracy Improvement: {final_results['comparison']['accuracy_improvement']:.4f}')
        logger.info(f'F1-score Improvement: {final_results['comparison']['f1_improvement']:.4f}')
        logger.info(f'Relative Accuracy Improvement: {final_results['comparison']['relative_accuracy_improvement']:.4f}')
        if robustness_results and 'summary' in robustness_results:
            logger.info('')
            logger.info(f'Strategic Robustness Score: {robustness_results['summary']['robustness_score']:.4f}')
            logger.info(f'Relative Robustness: {robustness_results['summary']['relative_robustness']:.4f}')
        if comparison_results and 'comparison' in comparison_results:
            logger.info('')
            logger.info('COMPARISON ON MANIPULATED DATA:')
            logger.info(f'Regular Classifier on Manipulated Data: {comparison_results['regular_on_manipulated']['accuracy']:.4f}')
            logger.info(f'Strategic Classifier on Manipulated Data: {comparison_results['strategic_on_manipulated']['accuracy']:.4f}')
            logger.info(f'Accuracy Improvement: {comparison_results['comparison']['accuracy_improvement']:.4f}')
            logger.info(f'F1-score Improvement: {comparison_results['comparison']['f1_improvement']:.4f}')
        logger.info('')
        logger.info(f'Total Evaluation Time: {total_time:.2f} seconds')
        logger.info(f'Results saved to: {output_path}')
        logger.info('=' * 60)
        logger.info('EVALUATION COMPLETED SUCCESSFULLY')
        logger.info('=' * 60)
    except Exception as e:
        logger.error(f'Evaluation failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

def set_seed(seed: int=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def split_dataset(texts: List[str], labels: List[str], test_size: float=0.3, random_state: int=42) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Split dataset into train and test sets.
    
    Args:
        texts: List of text samples
        labels: List of corresponding labels
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_texts, test_texts, train_labels, test_labels)
    """
    logger.info(f'Splitting dataset with test_size={test_size}')
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=test_size, random_state=random_state, stratify=labels)
    logger.info(f'Train set: {len(train_texts)} examples')
    logger.info(f'Test set: {len(test_texts)} examples')
    return (train_texts, test_texts, train_labels, test_labels)

def create_strategic_config(model_name: str, cost_strategy: str='balanced') -> Dict[str, Any]:
    """Create configuration for strategic classification with balanced cost functions.
    
    Args:
        model_name: Name of the HuggingFace model to get embedding dimension from
        cost_strategy: Cost function strategy ('balanced', 'sparse_low', 'uniform_low', 'minimal')
    
    Returns:
        Configuration dictionary with strategic settings
    """
    try:
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(model_name)
        embedding_dim = model_config.hidden_size
        logger.info(f'Model {model_name} embedding dimension: {embedding_dim}')
    except Exception as e:
        logger.error(f'Failed to get embedding dimension for model {model_name}: {e}')
        raise RuntimeError(f'Could not determine embedding dimension for model {model_name}. Please ensure the model exists and is accessible.')
    if cost_strategy == 'balanced':
        manipulable_dims = int(embedding_dim * 0.5)
        cost_coefficients = [0.0] * embedding_dim
        import random
        random.seed(42)
        manipulable_indices = random.sample(range(embedding_dim), manipulable_dims)
        for idx in manipulable_indices:
            cost_coefficients[idx] = 0.3
        logger.info(f'Balanced cost function: {manipulable_dims} manipulable dimensions with cost 0.3')
    elif cost_strategy == 'sparse_low':
        manipulable_dims = int(embedding_dim * 0.2)
        cost_coefficients = [0.0] * embedding_dim
        import random
        random.seed(42)
        manipulable_indices = random.sample(range(embedding_dim), manipulable_dims)
        for idx in manipulable_indices:
            cost_coefficients[idx] = 0.4
        logger.info(f'Sparse low cost function: {manipulable_dims} manipulable dimensions with cost 0.4')
    elif cost_strategy == 'uniform_low':
        cost_coefficients = [0.15] * embedding_dim
        logger.info(f'Uniform low cost function: 0.15 across all {embedding_dim} dimensions')
    elif cost_strategy == 'minimal':
        cost_coefficients = [0.05] * embedding_dim
        logger.info(f'Minimal cost function: 0.05 across all {embedding_dim} dimensions')
    elif cost_strategy == 'sparse_high':
        manipulable_dims = int(embedding_dim * 0.3)
        cost_coefficients = [0.0] * embedding_dim
        import random
        random.seed(42)
        manipulable_indices = random.sample(range(embedding_dim), manipulable_dims)
        for idx in manipulable_indices:
            cost_coefficients[idx] = 0.4
        logger.info(f'Sparse high (adjusted) cost function: {manipulable_dims} manipulable dimensions with cost 0.4')
    else:
        raise ValueError(f'Unknown cost strategy: {cost_strategy}')
    return {'enable_strategic_mode': True, 'cost_function_type': 'linear', 'cost_coefficients': cost_coefficients, 'strategic_lambda': 0.05, 'strategic_training_frequency': 10, 'strategic_blend_regular_weight': 0.7, 'strategic_blend_strategic_weight': 0.3, 'strategic_robust_proto_weight': 0.8, 'strategic_robust_head_weight': 0.2, 'strategic_prediction_proto_weight': 0.5, 'strategic_prediction_head_weight': 0.5}

