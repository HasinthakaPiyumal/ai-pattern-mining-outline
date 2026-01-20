# Cluster 4

def evaluate_comparison_on_manipulated_data(regular_classifier: AdaptiveClassifier, strategic_classifier: AdaptiveClassifier, test_texts: List[str], test_labels: List[str]) -> Dict[str, Any]:
    """Perform comparison by evaluating both classifiers on manipulated data.
    
    Args:
        regular_classifier: Regular classifier (no strategic training)
        strategic_classifier: Strategic classifier
        test_texts: Original test texts
        test_labels: Test labels
        
    Returns:
        Dictionary with comparison results
    """
    logger.info('=' * 60)
    logger.info('EVALUATION ON MANIPULATED DATA')
    logger.info('=' * 60)
    manipulated_embeddings = generate_manipulated_data(strategic_classifier, test_texts, manipulation_level=1.0)
    logger.info('Evaluating regular classifier on manipulated data...')
    regular_on_manipulated = evaluate_classifier_on_embeddings(regular_classifier, manipulated_embeddings, test_labels, mode='regular')
    logger.info('Evaluating strategic classifier on manipulated data...')
    strategic_on_manipulated = evaluate_classifier_on_embeddings(strategic_classifier, manipulated_embeddings, test_labels, mode='dual')
    accuracy_improvement = strategic_on_manipulated['accuracy'] - regular_on_manipulated['accuracy']
    f1_improvement = strategic_on_manipulated['f1_score'] - regular_on_manipulated['f1_score']
    return {'regular_on_manipulated': regular_on_manipulated, 'strategic_on_manipulated': strategic_on_manipulated, 'comparison': {'accuracy_improvement': accuracy_improvement, 'f1_improvement': f1_improvement, 'relative_accuracy_improvement': accuracy_improvement / regular_on_manipulated['accuracy'] if regular_on_manipulated['accuracy'] > 0 else 0.0}}

def generate_manipulated_data(strategic_classifier: AdaptiveClassifier, test_texts: List[str], manipulation_level: float=1.0) -> List[torch.Tensor]:
    """Generate strategically manipulated versions of test data.
    
    Args:
        strategic_classifier: Classifier with strategic capabilities
        test_texts: Original test texts
        manipulation_level: Level of manipulation (0.0 = no manipulation, 1.0 = full manipulation)
        
    Returns:
        List of manipulated embeddings
    """
    if not strategic_classifier.strategic_mode:
        logger.warning('Strategic mode not enabled - returning original embeddings')
        return strategic_classifier._get_embeddings(test_texts)
    logger.info(f'Generating manipulated data with manipulation level: {manipulation_level}')
    manipulated_embeddings = []
    original_embeddings = strategic_classifier._get_embeddings(test_texts)

    def classifier_func(x):
        with torch.no_grad():
            if strategic_classifier.adaptive_head is not None:
                strategic_classifier.adaptive_head.eval()
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                logits = strategic_classifier.adaptive_head(x.to(strategic_classifier.device))
                return F.softmax(logits, dim=-1)
            else:
                num_classes = len(strategic_classifier.label_to_id) if strategic_classifier.label_to_id else 1
                return torch.ones(1, num_classes) / num_classes
    for i, original_embedding in enumerate(original_embeddings):
        if torch.rand(1).item() < manipulation_level:
            try:
                manipulated_embedding = strategic_classifier.strategic_cost_function.compute_best_response(original_embedding, classifier_func)
                manipulated_embeddings.append(manipulated_embedding)
            except Exception as e:
                logger.warning(f'Strategic manipulation failed for example {i}: {e}')
                manipulated_embeddings.append(original_embedding)
        else:
            manipulated_embeddings.append(original_embedding)
        if (i + 1) % 10 == 0:
            logger.info(f'Generated {i + 1} / {len(test_texts)} manipulated examples')
    return manipulated_embeddings

def evaluate_classifier_on_embeddings(classifier: AdaptiveClassifier, embeddings: List[torch.Tensor], test_labels: List[str], mode: str='regular') -> Dict[str, Any]:
    """Evaluate a classifier on pre-computed embeddings.
    
    Args:
        classifier: Trained classifier
        embeddings: List of embeddings to evaluate on
        test_labels: True labels
        mode: Evaluation mode (for strategic classifiers)
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f'Evaluating classifier on manipulated data in {mode} mode...')
    predictions = []
    prediction_probs = []
    prediction_times = []
    for i, embedding in enumerate(embeddings):
        start_time = time.time()
        try:
            if mode == 'regular' or not classifier.strategic_mode:
                pred_results = classifier._predict_from_embedding(embedding, k=2)
            elif mode == 'strategic':
                pred_results = classifier._predict_from_embedding(embedding, k=2, strategic=True)
            elif mode == 'robust':
                pred_results = classifier._predict_from_embedding(embedding, k=2, robust=True)
            elif mode == 'dual':
                pred_results = classifier._predict_from_embedding(embedding, k=2)
            else:
                raise ValueError(f'Unknown evaluation mode: {mode}')
        except Exception as e:
            logger.warning(f'Prediction failed for embedding {i}: {e}')
            pred_results = classifier._predict_from_embedding(embedding, k=2)
        end_time = time.time()
        if pred_results:
            top_pred, top_prob = pred_results[0]
            predictions.append(top_pred)
            prediction_probs.append(top_prob)
        else:
            predictions.append('negative')
            prediction_probs.append(0.5)
        prediction_times.append(end_time - start_time)
        if (i + 1) % 10 == 0:
            logger.info(f'Evaluated {i + 1} / {len(embeddings)} examples')
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='weighted')
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(test_labels, predictions, average=None, labels=['negative', 'positive'])
    cm = confusion_matrix(test_labels, predictions, labels=['negative', 'positive'])
    avg_confidence = np.mean(prediction_probs)
    std_confidence = np.std(prediction_probs)
    avg_prediction_time = np.mean(prediction_times)
    results = {'mode': mode, 'accuracy': float(accuracy), 'precision': float(precision), 'recall': float(recall), 'f1_score': float(f1), 'per_class_metrics': {'negative': {'precision': float(precision_per_class[0]), 'recall': float(recall_per_class[0]), 'f1_score': float(f1_per_class[0]), 'support': int(support_per_class[0])}, 'positive': {'precision': float(precision_per_class[1]), 'recall': float(recall_per_class[1]), 'f1_score': float(f1_per_class[1]), 'support': int(support_per_class[1])}}, 'confusion_matrix': cm.tolist(), 'avg_confidence': float(avg_confidence), 'std_confidence': float(std_confidence), 'avg_prediction_time': float(avg_prediction_time), 'total_predictions': len(predictions)}
    logger.info(f'{mode.capitalize()} mode results on manipulated data:')
    logger.info(f'  Accuracy: {accuracy:.4f}')
    logger.info(f'  F1-score: {f1:.4f}')
    logger.info(f'  Avg confidence: {avg_confidence:.4f}')
    logger.info(f'  Avg prediction time: {avg_prediction_time:.4f}s')
    return results

