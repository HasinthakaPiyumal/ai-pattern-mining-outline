# Cluster 26

def get_blueprints_from_blueprint_book(data: Dict, source_file: str, index: BlueprintIndex) -> int:
    """Process blueprints from blueprint book and return count of processed blueprints"""
    processed = 0
    if 'blueprints' in data:
        for bp in data['blueprints']:
            if 'blueprint' in bp:
                if index.add_blueprint(bp_data=bp['blueprint'], source_file=source_file, label=bp['blueprint'].get('label', 'Unnamed Blueprint')):
                    processed += 1
    if 'blueprint_book' in data and 'blueprints' in data['blueprint_book']:
        for bp in data['blueprint_book']['blueprints']:
            if 'blueprint_book' in bp:
                processed += get_blueprints_from_blueprint_book(bp, source_file, index)
            elif 'blueprint' in bp:
                if index.add_blueprint(bp_data=bp['blueprint'], source_file=source_file, label=bp['blueprint'].get('label', 'Unnamed Blueprint')):
                    processed += 1
    return processed

def main():
    prototype_names = set((proto.value[0] for proto in Prototype))
    index = BlueprintIndex(valid_entities=prototype_names)
    total_blueprints = 0
    valid_blueprints = 0
    for file, data in load_json_files().items():
        total_blueprints += 1
        valid_blueprints += get_blueprints_from_blueprint_book(data, str(file), index)
    plot_histograms(index.blueprints)
    processed_names = get_processed_filenames('blueprints')
    all_blueprints = index.find_all_blueprints()
    processed_blueprints = set()
    for bp in all_blueprints:
        processed_blueprints.add(bp)
    misc_blueprints = [bp for bp in index.blueprints if bp not in processed_blueprints and bp.label not in processed_names]
    analyze_missing_prototypes(index.blueprints, prototype_names)
    belt_only_blueprints = []
    non_belt_blueprints = []
    for bp in misc_blueprints:
        if all(('-belt' in entity['name'] or 'splitter' in entity['name'] for entity in bp.data['entities'])):
            belt_only_blueprints.append(bp)
        else:
            non_belt_blueprints.append(bp)
    write_blueprints_to_folder(belt_only_blueprints, './blueprints/balancing')
    write_blueprints_to_folder(non_belt_blueprints, './blueprints/other')
    print(f'Total blueprints: {len(index.blueprints)}')
    print(f'Processed blueprints: {len(processed_blueprints)}')
    print(f'Misc blueprints: {len(misc_blueprints)}')
    pass

def load_json_files() -> dict:
    data = {}
    for file in Path('blueprints/decoded').rglob('*.json'):
        with open(file, 'r') as f:
            data[str(file)] = json.load(f)
    return data

def plot_histograms(blueprints: List[Blueprint]):
    """Plot histograms of blueprint statistics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    unique_lengths = [len(bp.names) for bp in blueprints]
    ax1.hist(unique_lengths, bins=30, edgecolor='black')
    ax1.set_title('Unique Entity Types per Blueprint')
    ax1.set_xlabel('Number of Unique Entity Types')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    mean_unique = sum(unique_lengths) / len(unique_lengths)
    median_unique = sorted(unique_lengths)[len(unique_lengths) // 2]
    ax1.axvline(mean_unique, color='red', linestyle='--', label=f'Mean: {mean_unique:.1f}')
    ax1.axvline(median_unique, color='green', linestyle='--', label=f'Median: {median_unique}')
    ax1.legend()
    total_lengths = list(filter(lambda x: x < 500, [bp.entity_count for bp in blueprints]))
    ax2.hist(total_lengths, bins=30, edgecolor='black')
    ax2.set_title('Total Entities per Blueprint')
    ax2.set_xlabel('Number of Entities')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    mean_total = sum(total_lengths) / len(total_lengths)
    median_total = sorted(total_lengths)[len(total_lengths) // 2]
    ax2.axvline(mean_total, color='red', linestyle='--', label=f'Mean: {mean_total:.1f}')
    ax2.axvline(median_total, color='green', linestyle='--', label=f'Median: {median_total}')
    ax2.legend()
    plt.tight_layout()
    plt.show()
    print('\nBlueprint Statistics:')
    print('Unique Entity Types:')
    print(f'  Mean: {mean_unique:.1f}')
    print(f'  Median: {median_unique}')
    print(f'  Min: {min(unique_lengths)}')
    print(f'  Max: {max(unique_lengths)}')
    print('\nTotal Entities:')
    print(f'  Mean: {mean_total:.1f}')
    print(f'  Median: {median_total}')
    print(f'  Min: {min(total_lengths)}')
    print(f'  Max: {max(total_lengths)}')

def get_processed_filenames(blueprints_dir: str) -> Set[str]:
    """Get all filenames from subdirectories of blueprints folder"""
    processed = set()
    for subdir in Path(blueprints_dir).iterdir():
        if subdir.is_dir() and subdir.name != 'decoded' and (subdir.name != 'misc'):
            for file in subdir.glob('*.json'):
                processed.add(file.stem)
    return processed

def analyze_missing_prototypes(blueprints: List[Blueprint], prototype_names: Set[str]):
    """Analyze entities that appear in blueprints but not in Prototype enum"""
    all_names = set()
    name_counts = Counter()
    for bp in blueprints:
        all_names.update(bp.names)
        name_counts.update(bp.names)
    missing_names = all_names - prototype_names
    missing_counts = {name: name_counts[name] for name in missing_names}
    sorted_missing = sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)
    print('\nEntities not in Prototype enum:')
    print(f'Total missing entities: {len(missing_names)}')
    print('\nTop 20 most common missing entities:')
    for name, count in sorted_missing[:20]:
        print(f'{name}: {count} appearances')
    print('\nEnum format for missing entities:')
    for name, _ in sorted_missing:
        sanitized_name = name.replace('-', '_').upper()
        print(f'{sanitized_name} = ("{name}",)')

def write_blueprints_to_folder(blueprints: List[Blueprint], folder: str):
    """Write blueprints to individual files in the specified folder"""
    Path(folder).mkdir(parents=True, exist_ok=True)
    for bp in blueprints:
        label = bp.label.replace('/', '').replace('[', '').replace(']', '')
        with open(f'{folder}/{label}.json', 'w') as f:
            json.dump(bp.data, f, indent=2)

