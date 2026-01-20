# Cluster 6

def prepare_dataset_splits() -> Dict[str, pd.DataFrame]:
    """Load all JSONL files and prepare them as dataset splits."""
    splits = {}
    jsonl_files = list(DATASET_DIR.glob('*.jsonl'))
    print(f'Found {len(jsonl_files)} JSONL files')
    for jsonl_file in jsonl_files:
        split_name = get_split_name(jsonl_file.name)
        print(f"Processing {jsonl_file.name} as split '{split_name}'")
        data = load_jsonl(jsonl_file)
        df = pd.DataFrame(data)
        required_columns = ['question', 'answer', 'image']
        for col in required_columns:
            if col not in df.columns:
                print(f"Warning: Missing column '{col}' in {jsonl_file.name}")
        splits[split_name] = df
        print(f'  Loaded {len(df)} examples')
    return splits

def get_split_name(filename: str) -> str:
    """Extract a clean split name from the filename."""
    name = filename.replace('.jsonl', '')
    if '_2025-' in name:
        name = name.split('_2025-')[0]
    elif '_mc_2025-' in name:
        name = name.split('_mc_2025-')[0] + '_mc'
    return name

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def create_huggingface_dataset(splits: Dict[str, pd.DataFrame]) -> DatasetDict:
    """Create a HuggingFace DatasetDict from the splits."""
    dataset_dict = {}
    features = Features({'question': Value('string'), 'answer': Value('string'), 'image': HFImage(), 'image_id': Value('string'), 'question_type': Value('string'), 'metadata': Value('string')})
    for split_name, df in splits.items():
        print(f'\nProcessing split: {split_name}')
        split_data = []
        for idx, row in df.iterrows():
            image_path = resolve_image_path(row['image'])
            if image_path is None:
                print(f'Skipping example {idx} due to missing image')
                continue
            example = {'question': row['question'], 'answer': str(row['answer']), 'image': image_path, 'image_id': row['image'], 'question_type': row.get('question_type', 'unknown'), 'metadata': json.dumps(row.get('metadata', {}))}
            split_data.append(example)
        if split_data:
            dataset = Dataset.from_list(split_data, features=features)
            dataset_dict[split_name] = dataset
            print(f'  Created dataset with {len(dataset)} examples')
        else:
            print(f'  Warning: No valid examples for split {split_name}')
    return DatasetDict(dataset_dict)

def resolve_image_path(image_id: str) -> str:
    """Resolve the full path to an image given its ID."""
    full_path = IMAGE_BASE_DIR / image_id
    if not full_path.exists():
        print(f'Warning: Image not found at {full_path}')
        return None
    return str(full_path)

def main():
    """Main function to prepare and upload the dataset."""
    print('=== Factorio VQA Dataset Upload ===\n')
    if not DATASET_DIR.exists():
        print(f'Error: Dataset directory {DATASET_DIR} not found')
        return
    if not IMAGE_BASE_DIR.exists():
        print(f'Error: Image directory {IMAGE_BASE_DIR} not found')
        return
    splits = prepare_dataset_splits()
    if not splits:
        print('Error: No valid splits found')
        return
    print(f'\nTotal splits: {len(splits)}')
    print('Splits:', list(splits.keys()))
    dataset_dict = create_huggingface_dataset(splits)
    print('\n=== Dataset Summary ===')
    for split_name, dataset in dataset_dict.items():
        print(f'{split_name}: {len(dataset)} examples')
    print('\nSaving dataset locally for preview...')
    dataset_dict.save_to_disk('./factorio_vqa_dataset')
    print('Dataset saved to ./factorio_vqa_dataset')
    readme_path = DATASET_DIR / 'README.md'
    if readme_path.exists():
        print('\nDataset card found at', readme_path)
        print('Remember to upload this README.md to your HuggingFace dataset repository')

def generate_dataset_card(dataset_dir: Path) -> str:
    """Generate a comprehensive dataset card from JSONL files."""
    jsonl_files = list(dataset_dir.glob('*.jsonl'))
    stats = {'total_samples': 0, 'splits': {}, 'task_types': defaultdict(int), 'question_types': defaultdict(int)}
    examples = {}
    for jsonl_file in sorted(jsonl_files):
        split_name = jsonl_file.stem
        data = load_jsonl(jsonl_file)
        if not data:
            continue
        stats['total_samples'] += len(data)
        stats['splits'][split_name] = len(data)
        task_type = get_task_type(jsonl_file.name)
        stats['task_types'][task_type] += len(data)
        for item in data:
            q_type = item.get('question_type', 'unknown')
            stats['question_types'][q_type] += 1
        if data:
            examples[split_name] = data[0]
    card = f'---\nlicense: mit\ntask_categories:\n- visual-question-answering\n- image-to-text\nlanguage:\n- en\ntags:\n- factorio\n- game\n- vqa\n- spatial-reasoning\n- factory-simulation\npretty_name: Factorio Visual Question Answering Dataset\nsize_categories:\n- 1K<n<10K\n---\n\n# Factorio Visual Question Answering (VQA) Dataset\n\n## Dataset Description\n\nThis dataset contains visual question-answering pairs for the Factorio Learning Environment (FLE). It is designed to train and evaluate vision-language models on understanding Factorio game elements, spatial relationships, and factory designs.\n\n### Dataset Summary\n\n- **Total Samples**: {stats['total_samples']:,}\n- **Number of Splits**: {len(stats['splits'])}\n- **Task Categories**: {len(stats['task_types'])}\n- **Languages**: English\n- **License**: MIT\n- **Created**: {datetime.now().strftime('%Y-%m-%d')}\n\n### Task Distribution\n\n| Task Category | Samples |\n|--------------|---------|\n'
    for task_type, count in sorted(stats['task_types'].items()):
        card += f'| {task_type.capitalize()} | {count:,} |\n'
    card += '\n### Question Types\n\n| Type | Count |\n|------|-------|\n'
    for q_type, count in sorted(stats['question_types'].items()):
        card += f'| {q_type} | {count:,} |\n'
    card += '\n## Dataset Structure\n\n### Data Splits\n\nEach JSONL file represents a different split focused on specific task types:\n\n| Split Name | Samples | Description |\n|------------|---------|-------------|\n'
    task_descriptions = {'terrain_nearest_entity': 'Find nearest entities in terrain views', 'terrain_nearest_resource': 'Find nearest resources in terrain views', 'factory_nearest_entity': 'Find nearest entities in factory setups', 'factory_entity_status': 'Identify entity statuses in factories', 'entity_name': 'Identify entity names from blueprints', 'position_finding': 'Find entity positions in blueprints', 'entity_counting': 'Count entities in blueprints', 'entity_direction': 'Determine entity facing directions', 'denoising': 'Identify missing entities (denoising)', 'contrastive_alignment_title': 'Match blueprints to titles', 'contrastive_alignment_purpose': 'Match blueprints to purposes'}
    for split_name, count in sorted(stats['splits'].items()):
        base_name = split_name.split('_2025-')[0].replace('_mc', '')
        desc = task_descriptions.get(base_name, 'Visual question answering task')
        if '_mc' in split_name:
            desc += ' (multiple choice)'
        card += f'| {split_name} | {count:,} | {desc} |\n'
    card += '\n### Data Fields\n\nAll entries contain these common fields:\n- `question` (string): The question text\n- `answer` (string): The answer\n- `image` (string): Path to the associated image\n- `question_type` (string): Type of question (open_ended, multiple_choice, etc.)\n- `metadata` (dict): Additional task-specific metadata\n\n### Data Examples\n\nHere are examples from different task types:\n\n'
    example_splits = ['terrain_task', 'terrain_task_mc', 'factory_task', 'factory_task_mc', 'position_task', 'position_task_mc', 'entity_name_task', 'entity_name_task_mc', 'contrastive_alignment_title', 'counting_task', 'counting_task_mc', 'direction_task', 'simple_denoising_blueprint_task', 'entity_counting', 'denoising_mc', 'contrastive_alignment_purpose']
    for split in example_splits:
        split_match = None
        for split_name in examples:
            if split in split_name:
                split_match = split_name
                break
        if split_match and split_match in examples:
            example = examples[split_match]
            card += f'#### {split}\n```json\n{{\n  "question": "{example['question']}",\n  "answer": "{example['answer']}",\n  "image": "/blueprints/{{id}}.png"",\n  "question_type": "{example.get('question_type', 'unknown')}"\n}}\n```\n\n'
    card += '## Dataset Creation\n\n### Generation Process\n\nThe dataset was generated using the Factorio Learning Environment (FLE) with the following approach:\n\n1. **Terrain Tasks**: Generated by spawning at random coordinates and querying about nearby entities/resources\n2. **Factory Tasks**: Created by placing random entities and generating spatial/status questions\n3. **Blueprint Tasks**: Used pre-existing blueprint files to generate various question types\n4. **Denoising Tasks**: Modified blueprints by removing entities and asking about missing components\n5. **Contrastive Tasks**: Paired blueprints with titles/purposes for multiple-choice selection\n\n### Image Information\n\nImages are organized in three directories:\n- `blueprints/`: Rendered blueprint images\n- `terrain/`: Terrain view captures\n- `factory/`: Factory setup images\n\nAll images are saved as PNG files for lossless quality.\n\n## Usage\n\n### Loading the Dataset\n\n```python\nfrom datasets import load_dataset\n\n# Load all splits\ndataset = load_dataset("Noddybear/fle_vqa")\n\n# Load specific split\nterrain_data = load_dataset("Noddybear/fle_vqa", split="terrain_nearest_entity_mc")\n```\n\n### Answer Formats\n\n- **Open-ended position answers**: `"Position(x=X, y=Y)"`\n- **Multiple choice answers**: Single letter `"a"`, `"b"`, `"c"`, or `"d"`\n- **Entity names**: Lowercase with hyphens (e.g., `"transport-belt"`)\n- **Directions**: Compass directions (e.g., `"north"`, `"east"`)\n- **Counts**: Integer strings (e.g., `"5"`)\n\n## Considerations\n\n- Questions are designed to be answerable from visual information alone\n- Multiple choice questions include plausible distractors\n- Positions are given in integer game coordinates\n- Some images may contain multiple valid entities for "nearest" questions\n\n## Citation\n\nIf you use this dataset, please cite:\n\n```bibtex\n@dataset{factorio_vqa_2025,\n  title={Factorio Visual Question Answering Dataset},\n  author={FLE Contributors},\n  year={2025},\n  publisher={HuggingFace}\n}\n```\n'
    return card

def get_task_type(filename: str) -> str:
    """Extract task type from filename."""
    name = filename.replace('.jsonl', '')
    if '_2025-' in name:
        name = name.split('_2025-')[0]
    elif '_mc_2025-' in name:
        name = name.split('_mc_2025-')[0]
    if 'terrain' in name:
        return 'terrain'
    elif 'factory' in name:
        return 'factory'
    elif 'blueprints' in name or name in ['entity_name', 'position_finding', 'entity_counting', 'entity_direction', 'denoising', 'contrastive_alignment_title', 'contrastive_alignment_purpose']:
        return 'blueprints'
    else:
        return 'other'

def main():
    """Generate dataset card and upload to HuggingFace."""
    dataset_dir = Path('/Users/jackhopkins/PycharmProjects/PaperclipMaximiser/data/vqa/dataset')
    print('Generating dataset card...')
    dataset_card = generate_dataset_card(dataset_dir)
    readme_path = dataset_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(dataset_card)
    print(f'Dataset card saved to {readme_path}')
    print('\nUploading to HuggingFace...')
    api = HfApi(token=os.getenv('HF_TOKEN'))
    api.upload_large_folder(folder_path=str(dataset_dir), repo_id='Noddybear/fle_vqa', repo_type='dataset')
    print('Upload complete!')

