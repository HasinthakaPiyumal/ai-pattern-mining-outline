# Cluster 8

def create_benchmark_dataset() -> Dataset:
    """Create the complete benchmark dataset"""
    all_examples = []
    for category, config in tqdm(SOURCES.items(), desc='Processing datasets'):
        print(f'\nProcessing {category} dataset...')
        dataset = load_source_dataset(config)
        if not dataset:
            continue
        try:
            examples = select_challenging_examples(dataset, category, config['samples'], config['field_map'])
            print(f'Selected {len(examples)} examples from {category}')
            all_examples.extend(examples)
        except Exception as e:
            print(f'Error selecting examples from {category}: {str(e)}')
            continue
    random.shuffle(all_examples)
    num_train = int(len(all_examples) * SPLIT_RATIO['train'])
    train_examples = all_examples[:num_train]
    test_examples = all_examples[num_train:]
    dataset_dict = DatasetDict({'train': Dataset.from_list(train_examples), 'test': Dataset.from_list(test_examples)})
    return dataset_dict

def load_source_dataset(config: Dict[str, Any]) -> datasets.Dataset:
    """Load a source dataset with error handling"""
    try:
        dataset = datasets.load_dataset(config['name'], config.get('subset'))
        return dataset
    except Exception as e:
        print(f'Error loading dataset {config['name']}: {str(e)}')
        return None

def select_challenging_examples(dataset: datasets.Dataset, category: str, num_samples: int, field_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """Select challenging examples from the dataset"""
    examples = []
    all_examples = dataset['train'] if 'train' in dataset else dataset['validation']
    shuffled_indices = list(range(len(all_examples)))
    random.shuffle(shuffled_indices)
    for idx in shuffled_indices:
        example = all_examples[idx]
        try:
            if category == 'gsm8k':
                question = str(example[field_map['question']])
                answer = str(example[field_map['answer']])
                if answer.count('=') < 3:
                    continue
            elif category == 'boolq':
                passage = str(example[field_map['passage']])
                q = str(example[field_map['question']])
                question = f'Context: {passage}\nQuestion: {q}'
                answer = 'Yes' if example[field_map['answer']] else 'No'
            elif category == 'mmlu_math':
                question = str(example[field_map['question']])
                choices = example[field_map['choices']]
                answer_index = int(example[field_map['answer']])
                if 0 <= answer_index < len(choices):
                    answer = choices[answer_index]
                else:
                    print(f"Warning: Answer index '{answer_index}' is out of range for choices: {choices}")
                    continue
                choices_text = '\n'.join([f'{i}. {choice}' for i, choice in enumerate(choices)])
                question = f'{question}\nChoices:\n{choices_text}'
            elif category == 'aqua_rat':
                question = str(example[field_map['question']])
                answer = str(example[field_map['answer']])
                if len(question.split()) < 12:
                    continue
            if len(question.split()) < 10:
                continue
            examples.append(format_question(category, question, answer))
            if len(examples) >= num_samples:
                break
        except Exception as e:
            print(f'Error processing example from {category}: {str(e)}')
            continue
    return examples

def main():
    """Main execution function"""
    print('Starting OptILM Bench dataset generation...')
    dataset = create_benchmark_dataset()
    print('\nDataset Statistics:')
    for split in dataset:
        print(f'\n{split} split:')
        print(f'Number of examples: {len(dataset[split])}')
        categories = dataset[split].unique('category')
        for category in categories:
            count = len([ex for ex in dataset[split] if ex['category'] == category])
            print(f'- {category}: {count} examples')
    print('\nPushing dataset to HuggingFace Hub...')
    push_to_hub(dataset, DATASET_NAME)

def push_to_hub(dataset: DatasetDict, repo_id: str):
    """Push the dataset to HuggingFace Hub"""
    try:
        readme_content = f"""# OptiLLMBench Dataset\n\nA benchmark dataset for evaluating test-time optimization and scaling capabilities of language models.\n\n## Dataset Description\n\nOptiLLMBench contains {NUM_SAMPLES} carefully selected challenging problems across multiple domains:\n- Mathematical reasoning (from competition_math)\n- Code generation (from HumanEval)\n- Word problems (from GSM8K)\n- Multiple choice reasoning (from MMLU)\n- Logical deduction (from BBH)\n\nEach example is chosen to benefit from test-time optimization techniques like:\n- Increased context length\n- Chain-of-thought reasoning\n- Self-consistency\n- Multiple solution attempts\n- And other scaling approaches\n\n## Usage\n\n```python\nfrom datasets import load_dataset\n\ndataset = load_dataset("codelion/optillmbench")\n\n# Access examples\nfor example in dataset["train"]:\n    print(f"Category: {{example['category']}}")\n    print(f"Question: {{example['question']}}")\n    print(f"Answer: {{example['answer']}}")\n    print(f"Metadata: {{example['metadata']}}")\n```\n\n## Citation\n\nIf you use this dataset in your research, please cite:\n\n```bibtex\n@software{{optillm,\n  title = {{Optillm: Optimizing inference proxy for LLMs}},\n  author = {{Asankhaya Sharma}},\n  year = {{2024}},\n  publisher = {{GitHub}},\n  url = {{https://github.com/codelion/optillm}}\n}}\n```\n"""
        dataset.push_to_hub(repo_id, private=False, embed_external_files=True)
        api = HfApi()
        api.upload_file(path_or_fileobj=readme_content.encode(), path_in_repo='README.md', repo_id=repo_id, repo_type='dataset')
        print(f'Successfully pushed dataset to {repo_id}')
    except Exception as e:
        print(f'Error pushing to hub: {str(e)}')

