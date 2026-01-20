# Cluster 1

class TokenExporter:
    """
    Exporter for pivotal tokens to various formats.
    """

    def __init__(self, token_storage: Optional[TokenStorage]=None, searcher=None):
        """
        Initialize the token exporter.
        
        Args:
            token_storage: Optional TokenStorage instance to export from
            searcher: Optional PivotalTokenSearcher instance for finding rejected tokens
        """
        self.token_storage = token_storage or TokenStorage()
        self.searcher = searcher

    def export_dpo_dataset(self, output_path: str, filter_criteria: Optional[Dict[str, Any]]=None, min_prob_delta: float=0.1, balance_positive_negative: bool=True, max_pairs: Optional[int]=None, seed: int=42, model_name: Optional[str]=None, save_tokens: bool=True, tokens_output_path: Optional[str]=None, num_candidates: int=10, find_rejected_tokens: bool=True, hf_push: bool=False, hf_repo_id: Optional[str]=None, private: bool=False):
        """
        Export pivotal tokens as DPO training pairs.
        
        Args:
            output_path: Path to save the DPO pairs
            filter_criteria: Criteria to filter tokens
            min_prob_delta: Minimum probability delta for inclusion
            balance_positive_negative: Whether to balance positive and negative examples
            max_pairs: Maximum number of pairs to export
            seed: Random seed for shuffling
            model_name: Model name for finding rejected tokens
            save_tokens: Whether to save updated tokens
            tokens_output_path: Path to save updated tokens
            num_candidates: Number of candidate tokens to consider
            find_rejected_tokens: Whether to find rejected tokens
            hf_push: Whether to push to Hugging Face
            hf_repo_id: Hugging Face repository ID
            private: Whether to make the repository private
        """
        filtered_storage = self.token_storage
        if filter_criteria or min_prob_delta:
            filtered_storage = self.token_storage.filter(criteria=filter_criteria, min_prob_delta=min_prob_delta)
        positive_tokens = [t for t in filtered_storage if t.get('prob_delta', 0) > 0]
        negative_tokens = [t for t in filtered_storage if t.get('prob_delta', 0) < 0]
        logger.info(f'Found {len(positive_tokens)} positive and {len(negative_tokens)} negative tokens')
        if balance_positive_negative:
            min_count = min(len(positive_tokens), len(negative_tokens))
            random.seed(seed)
            if len(positive_tokens) > min_count:
                positive_tokens = random.sample(positive_tokens, min_count)
            if len(negative_tokens) > min_count:
                negative_tokens = random.sample(negative_tokens, min_count)
            logger.info(f'Balanced to {len(positive_tokens)} pairs of each type')
        all_tokens = positive_tokens + negative_tokens
        random.seed(seed)
        random.shuffle(all_tokens)
        if max_pairs and len(all_tokens) > max_pairs:
            all_tokens = all_tokens[:max_pairs]
        if find_rejected_tokens and (not self.searcher) and model_name:
            try:
                from .core import PivotalTokenSearcher
                from .oracle import DummyOracle
                logger.info(f'Creating searcher with model {model_name} for finding rejected tokens')
                self.searcher = PivotalTokenSearcher(model_name=model_name, oracle=DummyOracle(), prob_threshold=min_prob_delta)
                logger.info(f'Searcher created successfully')
            except Exception as e:
                logger.error(f'Error creating searcher: {e}')
                logger.warning('Will continue without finding rejected tokens')
        pairs = []
        modified_tokens = []
        for token in all_tokens:
            query = token.get('query', '')
            pivot_context = token.get('pivot_context', '')
            pivot_token = token.get('pivot_token', '')
            rejected_token = token.get('rejected_token')
            rejected_token_id = token.get('rejected_token_id')
            prob_delta = token.get('prob_delta', 0)
            token_id = token.get('pivot_token_id')
            if find_rejected_tokens and self.searcher and (prob_delta > 0 and (not rejected_token) or prob_delta < 0):
                try:
                    if prob_delta > 0:
                        from .core import PivotalToken
                        pivot_token_obj = PivotalToken(query=query, pivot_context=pivot_context, pivot_token=pivot_token, pivot_token_id=token_id, prob_before=token.get('prob_before', 0), prob_after=token.get('prob_after', 0), prob_delta=prob_delta, model_id=token.get('model_id', model_name or 'unknown'), task_type=token.get('task_type', 'unknown'))
                        result = self.searcher.find_rejected_tokens(pivot_token_obj, num_candidates=num_candidates)
                        if result:
                            rejected_token, rejected_token_id, _ = result
                            if rejected_token == pivot_token:
                                logger.warning(f"Skipping as rejected token '{rejected_token}' is the same as pivot token '{pivot_token}'")
                                continue
                            logger.info(f"Found rejected token '{rejected_token}' for positive token '{pivot_token}'")
                            token['rejected_token'] = rejected_token
                            token['rejected_token_id'] = rejected_token_id
                            modified_tokens.append(token)
                    else:
                        with torch.no_grad():
                            context_ids = self.searcher.tokenizer.encode(pivot_context, return_tensors='pt').to(self.searcher.device)
                            outputs = self.searcher.model(context_ids)
                            logits = outputs.logits[0, -1, :]
                            top_tokens = torch.topk(torch.softmax(logits, dim=0), k=5)
                            for i in range(5):
                                next_token_id = top_tokens.indices[i].item()
                                next_token_str = self.searcher.tokenizer.decode([next_token_id])
                                if next_token_str == pivot_token:
                                    continue
                                rejected_token = next_token_str
                                rejected_token_id = next_token_id
                                logger.info(f"Using token '{rejected_token}' as alternative for negative token '{pivot_token}'")
                                token['rejected_token'] = rejected_token
                                token['rejected_token_id'] = rejected_token_id
                                modified_tokens.append(token)
                                break
                except Exception as e:
                    logger.error(f'Error finding rejected token: {e}')
            if prob_delta > 0 and (not rejected_token):
                logger.warning(f'No rejected token found for token: {pivot_token}, skipping')
                continue
            if prob_delta > 0:
                if rejected_token and rejected_token != pivot_token:
                    pair = {'prompt': pivot_context, 'chosen': pivot_token, 'rejected': rejected_token, 'metadata': {'original_query': query, 'prob_delta': prob_delta, 'task_type': token.get('task_type', 'unknown')}}
                    pairs.append(pair)
            elif rejected_token and rejected_token != pivot_token:
                pair = {'prompt': pivot_context, 'chosen': rejected_token, 'rejected': pivot_token, 'metadata': {'original_query': query, 'prob_delta': abs(prob_delta), 'task_type': token.get('task_type', 'unknown')}}
                pairs.append(pair)
        with open(output_path, 'w') as f:
            for pair in pairs:
                f.write(json.dumps(pair) + '\n')
        logger.info(f'Exported {len(pairs)} DPO pairs to {output_path}')
        if save_tokens and modified_tokens:
            tokens_path = tokens_output_path or self.token_storage.filepath
            if tokens_path:
                logger.info(f'Saving {len(modified_tokens)} updated tokens to {tokens_path}')
                updated_storage = TokenStorage()
                for token in self.token_storage.tokens:
                    modified = next((m for m in modified_tokens if m.get('pivot_context') == token.get('pivot_context') and m.get('pivot_token') == token.get('pivot_token')), None)
                    updated_storage.add_token(modified or token)
                updated_storage.save(tokens_path)
                logger.info(f'Saved {len(updated_storage)} tokens to {tokens_path}')
        if hf_push and hf_repo_id:
            try:
                from huggingface_hub import create_repo, upload_file
                create_repo(hf_repo_id, private=private, repo_type='dataset', exist_ok=True)
                upload_file(path_or_fileobj=output_path, path_in_repo='dpo_dataset.jsonl', repo_id=hf_repo_id, repo_type='dataset')
                logger.info(f'Pushed DPO dataset to Hugging Face: {hf_repo_id}')
                readme_content = generate_readme_content(file_type='dpo', model_name=model_name, dataset_info=f'- **Dataset size:** {len(pairs)} pairs\n- **Minimum probability delta:** {min_prob_delta}')
                readme_path = 'README.md.tmp'
                with open(readme_path, 'w') as f:
                    f.write(readme_content)
                upload_file(path_or_fileobj=readme_path, path_in_repo='README.md', repo_id=hf_repo_id, repo_type='dataset')
                os.remove(readme_path)
            except Exception as e:
                logger.error(f'Error pushing to Hugging Face: {e}')

    def export_steering_vectors(self, output_path: str, model_name: str, layer_nums: List[int]=[19, 23, 27], num_clusters: int=10, pca_components: int=50, batch_size: int=4, filter_criteria: Optional[Dict[str, Any]]=None, min_prob_delta: float=0.2, select_layer: Optional[int]=None, hf_push: bool=False, hf_repo_id: Optional[str]=None, private: bool=False):
        """
        Export steering vectors for activation-based steering.
        
        Args:
            output_path: Path to save the steering vectors
            model_name: Model name for extracting activations
            layer_nums: List of layer numbers to extract activations from
            num_clusters: Number of clusters for K-means
            pca_components: Number of PCA components for dimensionality reduction
            batch_size: Batch size for processing
            filter_criteria: Criteria to filter tokens
            min_prob_delta: Minimum probability delta for inclusion
            select_layer: Layer to use for steering vectors (defaults to first layer in layer_nums)
            hf_push: Whether to push to Hugging Face
            hf_repo_id: Hugging Face repository ID
            private: Whether to make the repository private
        """
        filtered_storage = self.token_storage
        if filter_criteria or min_prob_delta:
            filtered_storage = self.token_storage.filter(criteria=filter_criteria, min_prob_delta=min_prob_delta)
        if len(filtered_storage) == 0:
            logger.warning('No tokens to export')
            return
        logger.info(f'Loading model {model_name} for extracting activations...')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        contexts = []
        token_data = []
        for token in filtered_storage:
            contexts.append(token.get('pivot_context', ''))
            token_data.append(token)
        logger.info(f'Extracting activations for {len(contexts)} contexts...')
        activations = {}
        hooks = {}

        def get_activation(layer_num):

            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                if layer_num not in activations:
                    activations[layer_num] = []
                activations[layer_num].append(hidden_states.detach().cpu())
            return hook
        for layer_num in layer_nums:
            if hasattr(model, 'transformer'):
                module = model.transformer.h[layer_num]
            elif hasattr(model, 'model'):
                if hasattr(model.model, 'layers'):
                    module = model.model.layers[layer_num]
                else:
                    module = model.model.decoder.layers[layer_num]
            else:
                module = model.layers[layer_num]
            hooks[layer_num] = module.register_forward_hook(get_activation(layer_num))
        all_activations = {layer: [] for layer in layer_nums}
        for i in tqdm(range(0, len(contexts), batch_size), desc='Extracting activations'):
            batch_contexts = contexts[i:i + batch_size]
            inputs = tokenizer(batch_contexts, return_tensors='pt', padding=True).to(device)
            for layer in layer_nums:
                activations[layer] = []
            with torch.no_grad():
                model(**inputs)
            for layer in layer_nums:
                for batch_idx, seq_len in enumerate(inputs['attention_mask'].sum(dim=1)):
                    final_token_idx = seq_len - 1
                    activation = activations[layer][0][batch_idx, final_token_idx, :]
                    all_activations[layer].append(activation)
        for hook in hooks.values():
            hook.remove()
        for layer in layer_nums:
            all_activations[layer] = torch.stack(all_activations[layer])
        select_layer = select_layer if select_layer in layer_nums else layer_nums[0]
        layer_activations = all_activations[select_layer]
        if len(layer_activations) > pca_components:
            logger.info(f'Applying PCA to reduce dimensions from {layer_activations.shape[1]} to {pca_components}')
            pca = PCA(n_components=pca_components)
            activations_np = layer_activations.numpy()
            reduced_activations = pca.fit_transform(activations_np)
        else:
            reduced_activations = layer_activations.numpy()
        logger.info(f'Clustering into {num_clusters} clusters...')
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(reduced_activations)
        cluster_indices = defaultdict(list)
        for i, cluster in enumerate(clusters):
            cluster_indices[int(cluster)].append(i)
        cluster_vectors = {}
        for cluster, indices in cluster_indices.items():
            cluster_mean = layer_activations[indices].mean(dim=0)
            cluster_tokens = [token_data[i] for i in indices]
            positive_count = sum((1 for token in cluster_tokens if token.get('prob_delta', 0) > 0))
            positive_ratio = positive_count / len(cluster_tokens) if cluster_tokens else 0
            cluster_vectors[cluster] = {'vector': cluster_mean.tolist(), 'indices': indices, 'tokens': [t.get('pivot_token', '') for t in cluster_tokens], 'positive_ratio': positive_ratio, 'size': len(indices)}
        reasoning_patterns = {'depth_and_thoroughness': ['therefore', 'alternatively', 'however', 'wait', "let's", 'so', 'think', 'analyze', 'additional', 'furthermore', 'moreover', 'deeper', 'detailed', 'comprehensive', 'examine', 'investigate', 'explore', 'consider', 'important', 'significant', 'critical', 'careful', 'precise', 'nuanced', 'full', 'complete', 'exhaustive', 'rigorous'], 'numerical_accuracy': ['calculate', 'compute', 'equation', 'correct', 'check', 'verify', 'math', 'number', 'calculation', 'result', 'answer', 'precision', 'formula', 'computation', 'sum', 'total', 'value', 'exact', 'accurate', 'integer', 'decimal', 'fraction', 'multiply', 'divide'], 'self_correction': ['mistake', 'incorrect', 'wrong', 'error', 'let me reconsider', 'actually', 'revise', 'correction', 'revising', 'mistaken', 'fix', 'adjust', 'rectify', 'amend', 'correct', 'misunderstood', 'misinterpreted', 'miscalculated'], 'exploration': ['alternative', 'approach', 'method', 'strategy', 'consider', 'explore', 'possibility', 'different', 'solution', 'examine', 'investigate', 'option', 'alternatives', 'pathway', 'direction', 'route', 'perspective', 'viewpoint'], 'organization': ['first', 'second', 'next', 'finally', 'step', 'organize', 'list', 'sequence', 'order', 'structure', 'outline', 'categorize', 'classify', 'group', 'arrange', 'prioritize', 'rank', 'sort', 'divide', 'section', 'segment']}
        pattern_scores = {}
        for cluster_id, cluster_info in cluster_vectors.items():
            pattern_scores[cluster_id] = {}
            all_tokens = ' '.join(cluster_info['tokens']).lower()
            for pattern, keywords in reasoning_patterns.items():
                score = sum((all_tokens.count(keyword.lower()) for keyword in keywords))
                pattern_scores[cluster_id][pattern] = score
        cluster_to_pattern = {}
        for cluster_id in cluster_vectors.keys():
            scores_for_cluster = pattern_scores[cluster_id]
            if scores_for_cluster:
                best_pattern = max(scores_for_cluster, key=scores_for_cluster.get)
                best_score = scores_for_cluster[best_pattern]
                if best_score > 0:
                    cluster_to_pattern[cluster_id] = best_pattern
                else:
                    patterns = list(reasoning_patterns.keys())
                    fallback_pattern = patterns[int(cluster_id) % len(patterns)]
                    cluster_to_pattern[cluster_id] = fallback_pattern
            else:
                patterns = list(reasoning_patterns.keys())
                fallback_pattern = patterns[int(cluster_id) % len(patterns)]
                cluster_to_pattern[cluster_id] = fallback_pattern
        steering_tokens = []
        for i, token in enumerate(token_data):
            steering_token = dict(token)
            for cluster_id, cluster_info in cluster_vectors.items():
                if i in cluster_info['indices']:
                    reasoning_pattern = cluster_to_pattern.get(cluster_id, 'unknown')
                    steering_token = {'query': token.get('query', ''), 'pivot_context': token.get('pivot_context', ''), 'pivot_token': token.get('pivot_token', ''), 'pivot_token_id': token.get('pivot_token_id', 0), 'prob_before': token.get('prob_before', 0), 'prob_after': token.get('prob_after', 0), 'prob_delta': token.get('prob_delta', 0), 'model_id': token.get('model_id', ''), 'task_type': token.get('task_type', 'unknown'), 'steering_vector': layer_activations[i].tolist(), 'cluster_id': int(cluster_id), 'reasoning_pattern': reasoning_pattern, 'cluster_vector': cluster_vectors[cluster_id]['vector'], 'steering_layer': select_layer}
                    break
            steering_tokens.append(steering_token)
        logger.info(f'Saving {len(steering_tokens)} steering tokens to {output_path}')
        with open(output_path, 'w') as f:
            for token in steering_tokens:
                f.write(json.dumps(token) + '\n')
        metadata_summary = f'- **Layer used:** {select_layer}\n'
        metadata_summary += f'- **Number of clusters:** {num_clusters}\n'
        metadata_summary += f'- **Number of vectors:** {len(steering_tokens)}\n'
        metadata_summary += '- **Clusters:**\n'
        for cluster_id, pattern_name in cluster_to_pattern.items():
            cluster_size = len(cluster_vectors[cluster_id]['indices'])
            positive_ratio = cluster_vectors[cluster_id]['positive_ratio']
            metadata_summary += f'  - Cluster {cluster_id}: {pattern_name} ({cluster_size} vectors, {positive_ratio:.2f} positive ratio)\n'
        if hf_push and hf_repo_id:
            try:
                from huggingface_hub import create_repo, upload_file
                create_repo(hf_repo_id, private=private, repo_type='dataset', exist_ok=True)
                upload_file(path_or_fileobj=output_path, path_in_repo='steering_vectors.jsonl', repo_id=hf_repo_id, repo_type='dataset')
                logger.info(f'Pushed steering vectors to Hugging Face: {hf_repo_id}')
                readme_content = generate_readme_content(file_type='steering', model_name=model_name, dataset_info=metadata_summary)
                readme_path = 'README.md.tmp'
                with open(readme_path, 'w') as f:
                    f.write(readme_content)
                upload_file(path_or_fileobj=readme_path, path_in_repo='README.md', repo_id=hf_repo_id, repo_type='dataset')
                os.remove(readme_path)
            except Exception as e:
                logger.error(f'Error pushing to Hugging Face: {e}')

    def export_thought_anchors(self, output_path: str, filter_criteria: Optional[Dict[str, Any]]=None, min_importance_score: float=0.1, max_anchors: Optional[int]=None, sort_by_importance: bool=True, include_alternatives: bool=True, hf_push: bool=False, hf_repo_id: Optional[str]=None, private: bool=False, model_name: Optional[str]=None):
        """
        Export thought anchors for use in inference systems.
        
        Args:
            output_path: Path to save the thought anchors dataset
            filter_criteria: Criteria to filter anchors
            min_importance_score: Minimum importance score for inclusion
            max_anchors: Maximum number of anchors to export
            sort_by_importance: Whether to sort by importance score
            include_alternatives: Whether to include alternative sentences tested
            hf_push: Whether to push to Hugging Face
            hf_repo_id: Hugging Face repository ID
            private: Whether to make the repository private
            model_name: Model name for metadata
        """
        logger.info(f'Exporting thought anchors to {output_path}')
        if hasattr(self.token_storage, 'tokens'):
            anchors = self.token_storage.tokens
        else:
            logger.error('No thought anchors found in storage')
            return
        if not anchors:
            logger.warning('No thought anchors to export')
            return
        logger.info(f'Found {len(anchors)} thought anchors in storage')
        filtered_anchors = []
        for anchor in anchors:
            importance_score = anchor.get('importance_score', 0)
            if importance_score < min_importance_score:
                continue
            if filter_criteria:
                skip_anchor = False
                for key, value in filter_criteria.items():
                    if key not in anchor or anchor[key] != value:
                        skip_anchor = True
                        break
                if skip_anchor:
                    continue
            filtered_anchors.append(anchor)
        logger.info(f'Filtered to {len(filtered_anchors)} thought anchors')
        if sort_by_importance:
            filtered_anchors.sort(key=lambda x: x.get('importance_score', 0), reverse=True)
        if max_anchors:
            filtered_anchors = filtered_anchors[:max_anchors]
            logger.info(f'Limited to top {len(filtered_anchors)} thought anchors')
        export_data = []
        for anchor in filtered_anchors:
            export_item = {'anchor_id': f'{anchor.get('dataset_id', 'unknown')}_{anchor.get('dataset_item_id', 'unknown')}_{anchor.get('sentence_id', 0)}', 'query': anchor.get('query', ''), 'sentence': anchor.get('sentence', ''), 'sentence_id': anchor.get('sentence_id', 0), 'prefix_context': anchor.get('prefix_context', ''), 'sentence_category': anchor.get('sentence_category', 'unknown'), 'importance_score': anchor.get('importance_score', 0), 'prob_delta': anchor.get('prob_delta', 0), 'prob_with_sentence': anchor.get('prob_with_sentence', 0), 'prob_without_sentence': anchor.get('prob_without_sentence', 0), 'is_positive': anchor.get('is_positive', False), 'task_type': anchor.get('task_type', 'generic'), 'model_id': anchor.get('model_id', model_name or 'unknown'), 'timestamp': anchor.get('timestamp', ''), 'reasoning_pattern': self._classify_reasoning_pattern(anchor)}
            if include_alternatives and 'alternatives_tested' in anchor:
                export_item['alternatives_tested'] = anchor['alternatives_tested']
                export_item['num_alternatives'] = len(anchor.get('alternatives_tested', []))
            export_data.append(export_item)
        summary_stats = self._generate_anchor_summary(export_data)
        with open(output_path, 'w') as f:
            for item in export_data:
                f.write(json.dumps(item) + '\n')
        logger.info(f'Exported {len(export_data)} thought anchors to {output_path}')
        logger.info(f'Summary: {summary_stats}')
        if hf_push and hf_repo_id:
            try:
                from huggingface_hub import create_repo, upload_file
                create_repo(hf_repo_id, private=private, repo_type='dataset', exist_ok=True)
                filename = os.path.basename(output_path)
                upload_file(path_or_fileobj=output_path, path_in_repo=filename, repo_id=hf_repo_id, repo_type='dataset')
                logger.info(f'Pushed thought anchors to Hugging Face: {hf_repo_id}')
                readme_content = generate_readme_content(file_type='thought_anchors', model_name=model_name, dataset_info=summary_stats)
                readme_path = 'README.md.tmp'
                with open(readme_path, 'w') as f:
                    f.write(readme_content)
                upload_file(path_or_fileobj=readme_path, path_in_repo='README.md', repo_id=hf_repo_id, repo_type='dataset')
                os.remove(readme_path)
            except Exception as e:
                logger.error(f'Error pushing to Hugging Face: {e}')

    def _classify_reasoning_pattern(self, anchor: Dict[str, Any]) -> str:
        """Classify the reasoning pattern of a thought anchor."""
        sentence = anchor.get('sentence', '').lower()
        category = anchor.get('sentence_category', '')
        if category == 'plan_generation':
            return 'planning'
        elif category == 'uncertainty_management':
            if any((word in sentence for word in ['wait', 'actually', 'hmm', 'reconsider'])):
                return 'backtracking'
            else:
                return 'uncertainty_management'
        elif category == 'self_checking':
            return 'verification'
        elif category == 'active_computation':
            return 'computation'
        elif category == 'fact_retrieval':
            return 'knowledge_access'
        elif category == 'result_consolidation':
            return 'consolidation'
        elif category == 'final_answer_emission':
            return 'answer_emission'
        elif category == 'problem_setup':
            return 'problem_understanding'
        else:
            return 'other'

    def _generate_anchor_summary(self, anchors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for exported thought anchors."""
        if not anchors:
            return {'total_anchors': 0}
        category_counts = {}
        pattern_counts = {}
        importance_scores = []
        for anchor in anchors:
            category = anchor.get('sentence_category', 'unknown')
            pattern = anchor.get('reasoning_pattern', 'unknown')
            importance = anchor.get('importance_score', 0)
            category_counts[category] = category_counts.get(category, 0) + 1
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            importance_scores.append(importance)
        return {'total_anchors': len(anchors), 'category_distribution': category_counts, 'reasoning_pattern_distribution': pattern_counts, 'average_importance': sum(importance_scores) / len(importance_scores) if importance_scores else 0, 'max_importance': max(importance_scores) if importance_scores else 0, 'min_importance': min(importance_scores) if importance_scores else 0, 'positive_anchors': sum((1 for a in anchors if a.get('is_positive', False))), 'negative_anchors': sum((1 for a in anchors if not a.get('is_positive', False)))}

def generate_readme_content(file_type, model_name=None, dataset_info=None):
    """
    Generate README content for Hugging Face repositories.
    
    Args:
        file_type: Type of file ('dpo', 'steering', or 'tokens')
        model_name: Name of the model used
        dataset_info: Additional dataset information
        
    Returns:
        README content as a string
    """
    if file_type == 'steering':
        content = f'# PTS Steering Vectors Dataset\n\nA dataset of activation-based steering vectors created using the Pivotal Token Search (PTS) technique.\n\n## Details\n\n- **Source:** Generated using the [PTS](https://github.com/codelion/pts) tool\n- **Model:** {model_name or 'Unknown'}\n\n## Dataset Structure\n\nThis dataset contains:\n- `steering_vectors.jsonl`: The main file with token-level steering vectors\n\n## Usage\n\nThese steering vectors can be used for activation-based steering during inference to guide language models toward particular reasoning patterns.\n\n### OptiLLM Integration\n\nYou can use these steering vectors with the open-source [OptiLLM](https://github.com/codelion/optillm) library for optimizing inference via the `autothink` approach. OptiLLM provides a proxy server that can apply steering techniques to improve model performance.\n\n### Example Python Code\n\n```python\nimport json\nimport torch\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n\n# Load model\nmodel = AutoModelForCausalLM.from_pretrained("{model_name or 'MODEL_NAME'}")\ntokenizer = AutoTokenizer.from_pretrained("{model_name or 'MODEL_NAME'}")\n\n# Load steering vectors directly from Hugging Face\nfrom datasets import load_dataset\ndataset = load_dataset("USERNAME/REPO_NAME")\nvectors = [json.loads(example) for example in dataset["train"]]\n\n# Define a hook to apply steering\ndef steering_hook(module, input, output):\n    # Add steering vector to activation\n    # Implementation depends on your specific use case\n    return output\n\n# Register hook on appropriate layer\nmodel.transformer.h[LAYER_NUM].register_forward_hook(steering_hook)\n\n# Generate text with steering\ninput_text = "Your prompt here"\ninput_ids = tokenizer.encode(input_text, return_tensors="pt")\noutput = model.generate(input_ids, max_length=100)\nresult = tokenizer.decode(output[0])\nprint(result)\n```\n'
    elif file_type == 'thought_anchors':
        content = f"""---\nlanguage:\n- en\ntags:\n- pts\n- thought-anchors\n- reasoning\n- llm-analysis\n- sentence-level-analysis\n- pivotal-token-search\nsize_categories:\n- n<1K\ntask_categories:\n- other\npretty_name: "PTS Thought Anchors Dataset"\ndataset_info:\n  config_name: default\n  features:\n  - name: model_id\n    dtype: string\n  - name: query\n    dtype: string\n  - name: sentence\n    dtype: string\n  - name: sentence_id\n    dtype: int64\n  - name: prefix_context\n    dtype: string\n  - name: prob_with_sentence\n    dtype: float64\n  - name: prob_without_sentence\n    dtype: float64\n  - name: prob_delta\n    dtype: float64\n  - name: task_type\n    dtype: string\n  - name: suffix_context\n    dtype: string\n  - name: full_reasoning_trace\n    dtype: string\n  - name: sentence_embedding\n    sequence: float64\n  - name: alternatives_embeddings\n    sequence:\n      sequence: float64\n  - name: causal_dependencies\n    sequence: int64\n  - name: causal_dependents\n    sequence: int64\n  - name: logical_relationship\n    dtype: string\n  - name: failure_mode\n    dtype: string\n  - name: error_type\n    dtype: string\n  - name: correction_suggestion\n    dtype: string\n  - name: importance_score\n    dtype: float64\n  - name: is_positive\n    dtype: bool\n  - name: sentence_category\n    dtype: string\n  - name: alternatives_tested\n    sequence: string\n  - name: dependency_sentences\n    sequence: int64\n  - name: dataset_id\n    dtype: string\n  - name: dataset_item_id\n    dtype: string\n  - name: timestamp\n    dtype: timestamp[s]\n---\n\n# PTS Thought Anchors Dataset\n\nA dataset of thought anchors - critical reasoning steps - identified using the Thought Anchors technique from the PTS tool.\n\n## Details\n\n- **Source:** Generated using the [PTS](https://github.com/codelion/pts) tool\n- **Model:** {model_name or 'Unknown'}\n- **Tags:** `pts`, `thought-anchors`, `reasoning`, `llm-analysis`\n\n## Dataset Structure\n\nThis dataset contains thought anchors identified from reasoning traces. Each anchor represents a sentence that significantly impacts the success probability of the reasoning process.\n\n## Fields\n\nEach thought anchor contains:\n\n### Core Fields\n- `model_id`: The model used to generate the reasoning trace\n- `query`: The original problem/question that was processed\n- `sentence`: The actual sentence that serves as a thought anchor\n- `sentence_id`: Position of the sentence in the reasoning trace\n- `prob_with_sentence`: Success probability when this sentence is included\n- `prob_without_sentence`: Success probability when this sentence is replaced/removed\n- `prob_delta`: Change in success probability (with - without)\n- `importance_score`: Absolute impact score of this anchor\n- `is_positive`: Whether this sentence helps (true) or hurts (false) success\n\n### Context Fields\n- `prefix_context`: All sentences that come before this one\n- `suffix_context`: All sentences that come after this one\n- `full_reasoning_trace`: Complete reasoning trace for context\n\n### Semantic Analysis\n- `sentence_embedding`: 384-dimensional vector representation of the sentence\n- `alternatives_embeddings`: Vector representations of alternative sentences tested\n- `alternatives_tested`: List of alternative sentences that were tested\n\n### Dependency Analysis\n- `causal_dependencies`: Sentence IDs this sentence logically depends on\n- `causal_dependents`: Sentence IDs that depend on this sentence\n- `logical_relationship`: Type of logical relationship ("premise", "conclusion", etc.)\n\n### Failure Analysis (for negative anchors)\n- `failure_mode`: Type of failure ("logical_error", "computational_mistake", etc.)\n- `error_type`: More specific error classification\n- `correction_suggestion`: How to improve the sentence\n\n### Classification\n- `sentence_category`: Type of reasoning step ("plan_generation", "active_computation", etc.)\n- `task_type`: Type of task being solved\n- `dataset_id`: Source dataset identifier\n- `dataset_item_id`: Specific item ID from the dataset\n- `timestamp`: When the anchor was generated\n\n## Usage\n\nThese thought anchors can be used for:\n- Understanding which reasoning steps matter most\n- Guiding inference systems to focus on critical reasoning steps\n- Analyzing reasoning patterns in language models\n- Building better reasoning evaluation metrics\n\n### Example Usage\n\n```python\nfrom datasets import load_dataset\n\n# Load thought anchors from Hugging Face\ndataset = load_dataset("codelion/Qwen3-0.6B-pts-thought-anchors")\nanchors = dataset['train']\n\n# Find high-impact positive anchors\npositive_anchors = anchors.filter(\n    lambda x: x["is_positive"] and x["importance_score"] > 0.3\n)\n\n# Find planning-related sentences\nplanning_anchors = anchors.filter(\n    lambda x: x["sentence_category"] == "plan_generation"\n)\n\n# Analyze failure modes for negative anchors\nfailure_analysis = {{}}\nnegative_anchors = anchors.filter(lambda x: not x["is_positive"] and x["failure_mode"])\nfor anchor in negative_anchors:\n    mode = anchor["failure_mode"]\n    failure_analysis[mode] = failure_analysis.get(mode, 0) + 1\n\nprint("Failure modes:", failure_analysis)\nprint(f"Found {{len(positive_anchors)}} positive anchors")\nprint(f"Found {{len(planning_anchors)}} planning anchors")\n\n# Example: Access embeddings for similarity search\nsample_anchor = anchors[0]\nembedding = sample_anchor["sentence_embedding"]  # 384-dim vector\nprint(f"Embedding dimension: {{len(embedding)}}")\n```\n\n### Integration with Inference Systems\n\nThought anchors can be used to:\n1. **Focus attention** on critical reasoning steps during generation\n2. **Validate reasoning** by checking for presence of important anchor patterns\n3. **Guide beam search** to prefer paths containing high-value anchor sentences\n4. **Improve CoT prompting** by incorporating successful anchor patterns\n\n### OptiLLM Integration\n\nYou can use these thought anchors with the open-source [OptiLLM](https://github.com/codelion/optillm) library for optimizing inference via the autothink approach. OptiLLM provides a proxy server that can apply thought anchor techniques to improve model reasoning performance by:\n\n- **Guided reasoning**: Using positive anchors as templates for better reasoning paths\n- **Quality monitoring**: Detecting negative anchor patterns to avoid poor reasoning\n- **Adaptive prompting**: Incorporating successful anchor patterns into prompts\n- **Real-time optimization**: Applying anchor insights during model inference\n\n"""
    elif file_type == 'dpo':
        content = f'# PTS DPO Dataset\n\nA Direct Preference Optimization (DPO) dataset created using the Pivotal Token Search (PTS) technique.\n\n## Details\n\n- **Source:** Generated using the [PTS](https://github.com/codelion/pts) tool\n- **Model:** {model_name or 'Unknown'}\n\n## Format\n\nEach example in the dataset consists of:\n- `prompt`: The context leading up to the pivotal token\n- `chosen`: The preferred token that increases success probability\n- `rejected`: The alternative token that decreases success probability\n- `metadata`: Additional information about the example\n\n## Usage\n\nThis dataset can be used for fine-tuning language models with Direct Preference Optimization (DPO).\n\nFor a quick start, you can use our Google Colab notebook to fine-tune a model using this DPO dataset:\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FggA9EQ1eFBjE0Qbsl0-EFzyWIxpdhlH?usp=sharing)\n\n```python\nfrom datasets import load_dataset\n\n# Load the dataset from Hugging Face\ndataset = load_dataset("USERNAME/REPO_NAME")\n\n# Use with your favorite DPO implementation\n# Example with TRL library:\nfrom trl import DPOTrainer\n\ntrainer = DPOTrainer(\n    model=model,\n    args=training_args,\n    beta=0.1,\n    train_dataset=dataset,\n    tokenizer=tokenizer,\n    # ... other parameters\n)\n\ntrainer.train()\n```\n'
    else:
        content = f'# PTS Pivotal Tokens Dataset\n\nA dataset of pivotal tokens discovered using the Pivotal Token Search (PTS) technique.\n\n## Details\n\n- **Source:** Generated using the [PTS](https://github.com/codelion/pts) tool\n- **Model:** {model_name or 'Unknown'}\n\n## Format\n\nEach token in the dataset includes:\n- `query`: The original query that was processed\n- `pivot_context`: The context leading up to the pivotal token\n- `pivot_token`: The actual token that impacts success probability\n- `prob_delta`: The change in success probability caused by the token\n- Other metadata about the token\n\n## Usage\n\nThese pivotal tokens can be used for creating DPO datasets or extracting steering vectors:\n\n```bash\n# Export to DPO format\npts export --input-path="pivotal_tokens.jsonl" --format="dpo" --output-path="dpo_dataset.jsonl" --model="MODEL_NAME" --find-rejected-tokens\n\n# Extract steering vectors\npts export --input-path="pivotal_tokens.jsonl" --format="steering" --output-path="steering_vectors.jsonl" --model="MODEL_NAME"\n```\n'
    if dataset_info:
        content += f'\n## Additional Information\n\n{dataset_info}\n'
    return content

