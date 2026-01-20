# Cluster 69

def _get_tools_base_path() -> Path:
    """Get the base path to the tools directory"""
    return importlib.resources.files('fle') / 'env' / 'tools'

def _get_repo_base_path() -> Path:
    """Get the base path to the repository directory"""
    return importlib.resources.files('fle')

def search_dir(dir_path):
    try:
        for entry in dir_path.iterdir():
            if entry.is_file() and entry.suffix in ['.py', '.lua', '.md', '.txt']:
                search_file(entry)
            elif entry.is_dir() and recursive:
                search_dir(entry)
    except Exception as e:
        results.append(f'Error accessing {dir_path}: {str(e)}')

def search_file(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        file_matches = []
        for i, line in enumerate(lines, 1):
            if regex.search(line):
                if line_numbers:
                    file_matches.append(f'{i}: {line.rstrip()}')
                else:
                    file_matches.append(line.rstrip())
        if file_matches:
            rel_path = str(file_path.relative_to(base_path))
            results.append(f'File: {rel_path}')
            results.append('```')
            results.extend(file_matches[:20])
            if len(file_matches) > 20:
                results.append(f'... and {len(file_matches) - 20} more matches')
            results.append('```')
            results.append('')
    except Exception:
        pass

def add_tree(current_path, prefix='', depth=0):
    if depth >= max_depth:
        return
    try:
        all_entries = list(current_path.iterdir())
        filtered_entries = [e for e in all_entries if not should_exclude(e)]
        entries = sorted(filtered_entries, key=lambda x: (not x.is_dir(), x.name))
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = '└── ' if is_last else '├── '
            icon = '📁' if entry.is_dir() else '📄'
            result.append(f'{prefix}{connector}{icon} {entry.name}')
            if entry.is_dir():
                new_prefix = prefix + ('    ' if is_last else '│   ')
                add_tree(entry, new_prefix, depth + 1)
    except PermissionError:
        result.append(f'{prefix}└── ⚠️ Permission denied')
    except Exception as e:
        result.append(f'{prefix}└── ⚠️ Error: {str(e)}')

def should_exclude(entry):
    if not show_hidden:
        name = entry.name
        return any((pattern in name for pattern in excluded_patterns))
    return False

