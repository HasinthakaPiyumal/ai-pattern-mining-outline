# Cluster 29

def main():
    load_dotenv()
    exec_dir = Path(__file__).parent
    builds_dir = exec_dir / 'full'
    results = analyze_factorio_builds(builds_dir)
    for file_path, analysis in results.items():
        print(f'\nFile: {file_path}')
        print(f'Objective: {analysis['objective']}')
        print(f'Docstring:\n{analysis['docstring']}')
        print(f'Inventory: {analysis['inventory']}')

def analyze_factorio_builds(builds_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Analyzes all Factorio build files in the specified directory and its subdirectories.
    Returns a dictionary of analysis results keyed by file path.
    """
    analyzer = BlueprintMetadataGenerator()
    results = {}
    for root, _, files in os.walk(builds_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                try:
                    results[str(file_path)] = analyzer.analyze_file(file_path)
                except Exception as e:
                    print(f'Error analyzing {file_path}: {str(e)}')
    return results

