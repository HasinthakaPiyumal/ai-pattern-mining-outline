# Cluster 3

def setup_logging(log_level: str='INFO'):
    """Set up logging with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler('pts.log')])

def export_tokens(args):
    """Export pivotal tokens to different formats."""
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info(f'Exporting tokens from {args.input_path} to {args.output_path}')
    storage = TokenStorage(filepath=args.input_path)
    exporter = TokenExporter(token_storage=storage)
    if args.format == 'dpo':
        logger.info(f'Exporting to DPO format')
        exporter.export_dpo_dataset(output_path=args.output_path, min_prob_delta=args.min_prob_delta, balance_positive_negative=args.balance, max_pairs=args.max_pairs, seed=args.seed, model_name=args.model, save_tokens=args.save_tokens, tokens_output_path=args.tokens_output_path, num_candidates=args.num_candidates, find_rejected_tokens=args.find_rejected_tokens, hf_push=args.hf_push, hf_repo_id=args.hf_repo_id, private=args.private)
    elif args.format == 'steering':
        logger.info(f'Exporting to steering vectors format')
        exporter.export_steering_vectors(output_path=args.output_path, model_name=args.model, layer_nums=args.layer_nums, num_clusters=args.num_clusters, pca_components=args.pca_components, batch_size=args.batch_size, min_prob_delta=args.min_prob_delta, select_layer=args.select_layer, hf_push=args.hf_push, hf_repo_id=args.hf_repo_id, private=args.private)
    elif args.format == 'thought_anchors':
        logger.info(f'Exporting to thought anchors format')
        exporter.export_thought_anchors(output_path=args.output_path, min_importance_score=getattr(args, 'min_prob_delta', 0.1), max_anchors=args.max_pairs, sort_by_importance=True, include_alternatives=True, hf_push=args.hf_push, hf_repo_id=args.hf_repo_id, private=args.private, model_name=args.model)
    else:
        logger.error(f'Unsupported export format: {args.format}')
        sys.exit(1)
    logger.info(f'Export completed successfully')

def push_to_hf(args):
    """Push a file to Hugging Face."""
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info(f'Pushing {args.input_path} to Hugging Face: {args.hf_repo_id}')
    try:
        from huggingface_hub import create_repo, upload_file
        create_repo(args.hf_repo_id, private=args.private, repo_type='dataset', exist_ok=True)
        filename = os.path.basename(args.input_path)
        upload_file(path_or_fileobj=args.input_path, path_in_repo=filename, repo_id=args.hf_repo_id, repo_type='dataset')
        logger.info(f'Pushed {args.input_path} to Hugging Face: {args.hf_repo_id}')
        if not args.no_readme:
            file_type = 'tokens'
            if filename.endswith('_vectors.jsonl') or 'steering' in filename:
                file_type = 'steering'
            elif 'dpo' in filename:
                file_type = 'dpo'
            elif 'thought_anchor' in filename or 'anchor' in filename:
                file_type = 'thought_anchors'
            else:
                try:
                    with open(args.input_path, 'r') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            import json
                            data = json.loads(first_line)
                            if 'sentence_embedding' in data or 'prob_with_sentence' in data or 'suffix_context' in data:
                                file_type = 'thought_anchors'
                            elif 'chosen' in data and 'rejected' in data:
                                file_type = 'dpo'
                            elif 'steering_vector' in data or 'activation_vector' in data:
                                file_type = 'steering'
                except Exception:
                    pass
            from .exporters import generate_readme_content
            readme_content = generate_readme_content(file_type=file_type, model_name=args.model)
            readme_path = 'README.md.tmp'
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            upload_file(path_or_fileobj=readme_path, path_in_repo='README.md', repo_id=args.hf_repo_id, repo_type='dataset')
            os.remove(readme_path)
            logger.info(f'Created README for {args.hf_repo_id}')
        else:
            logger.info(f'Skipped README creation (--no-readme flag used)')
    except Exception as e:
        logger.error(f'Error pushing to Hugging Face: {e}')
        sys.exit(1)

