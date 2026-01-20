# Cluster 4

def pattern_match(path: str, pattern: str) -> bool:
    """
    Improved pattern matching that better handles **/ patterns and different path separators.
    """
    pattern = normalize_pattern(pattern)
    path = path.replace('\\', '/')
    if pattern.startswith('**/'):
        base_pattern = pattern[3:]
        return fnmatch.fnmatch(path, base_pattern) or fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, f'**/{base_pattern}')
    elif pattern.startswith('*') and (not pattern.startswith('**/')):
        return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, f'**/{pattern}')
    return fnmatch.fnmatch(path, pattern)

def normalize_pattern(pattern: str) -> str:
    """
    Normalize a pattern by removing unnecessary whitespace.
    """
    return pattern.strip()

def matches_any_pattern(path: Path, patterns: List[str]) -> bool:
    """Check if path matches any of the given patterns."""
    if not patterns:
        return False
    str_path = str(path).replace('\\', '/')
    return any((pattern_match(str_path, p) for p in patterns))

def should_force_include(path: Path, append_patterns: List[str]) -> bool:
    """Check if path should be force-included via -a patterns."""
    if not append_patterns:
        return False
    str_path = str(path).replace('\\', '/')
    if matches_any_pattern(path, append_patterns):
        return True
    for pattern in append_patterns:
        if pattern.endswith('/**'):
            if path_in_directory(path, pattern):
                return True
    if path.is_dir():
        path_parts = str_path.split('/')
        for pattern in append_patterns:
            if pattern.endswith('/**') and '/**' in pattern:
                pattern_parts = pattern[:-3].split('/')
                if pattern.startswith('**/'):
                    pattern_parts = pattern_parts[1:]
                for i in range(min(len(path_parts), len(pattern_parts))):
                    if i == len(pattern_parts) - 1:
                        if fnmatch.fnmatch(path_parts[i], pattern_parts[i]):
                            return True
    return False

def path_in_directory(path: Path, dir_pattern: str) -> bool:
    """
    Check if path is inside a directory that matches the pattern.
    For patterns like "**/examples/workflow_mcp_server/**", only match that specific directory.
    """
    if not dir_pattern.endswith('/**'):
        return False
    base_dir = dir_pattern[:-3]
    has_prefix = base_dir.startswith('**/')
    if has_prefix:
        base_dir = base_dir[3:]
    str_path = str(path).replace('\\', '/')
    if '/' in base_dir:
        if has_prefix:
            return base_dir in str_path and (str_path.endswith(f'/{base_dir}') or f'/{base_dir}/' in str_path)
        else:
            return str_path.startswith(f'{base_dir}/') or str_path == base_dir
    parts = str_path.split('/')
    for i in range(len(parts)):
        prefix = '/'.join(parts[:i + 1])
        if fnmatch.fnmatch(prefix, base_dir):
            return True
    return False

def should_include_by_pattern(path: Path, include_patterns: List[str]) -> bool:
    """Check if path should be included based on -i patterns."""
    if not include_patterns:
        return True
    str_path = str(path).replace('\\', '/')
    if path.is_dir():
        if matches_any_pattern(path, include_patterns):
            return True
        for pattern in include_patterns:
            if pattern.endswith('/**') and path_in_directory(path, pattern):
                return True
        dir_path = str_path + '/'
        for pattern in include_patterns:
            pattern = normalize_pattern(pattern)
            if pattern.startswith('**/'):
                return True
            if fnmatch.fnmatch(dir_path + 'anyfile', pattern):
                return True
        return False
    return matches_any_pattern(path, include_patterns)

def should_ignore(path: Path, ignore_patterns: List[str], gitignore_patterns: List[str]) -> bool:
    """Check if path should be ignored based on -x patterns and gitignore."""
    return matches_any_pattern(path, ignore_patterns) or matches_any_pattern(path, gitignore_patterns)

def should_process_path(path: Path, include_patterns: List[str], append_patterns: List[str], ignore_patterns: List[str], gitignore_patterns: List[str]) -> bool:
    """
    Determine if a path should be processed based on precedence rules:
    1. If matches -a patterns → include
    2. If matches -i patterns → include
    3. If matches -x or gitignore patterns → exclude (unless forced by -a)
    4. If no -i patterns provided → include by default
    5. If -i patterns provided → exclude by default (only include what matches)
    """
    if should_force_include(path, append_patterns):
        return True
    if include_patterns and should_include_by_pattern(path, include_patterns):
        return True
    if should_ignore(path, ignore_patterns, gitignore_patterns):
        return False
    return not bool(include_patterns)

def has_includable_content(directory: Path, include_patterns: List[str], append_patterns: List[str], ignore_patterns: List[str], gitignore_patterns: List[str], visited_dirs=None) -> bool:
    """
    Check if a directory contains any files that should be included.
    Uses a visited_dirs set to prevent infinite recursion with symlinks.
    """
    if visited_dirs is None:
        visited_dirs = set()
    dir_path = directory.resolve()
    if dir_path in visited_dirs:
        return False
    visited_dirs.add(dir_path)
    try:
        for item in directory.iterdir():
            if any((pattern.endswith('/**') for pattern in append_patterns)):
                if should_force_include(item, append_patterns):
                    return True
            if should_process_path(item, include_patterns, append_patterns, ignore_patterns, gitignore_patterns):
                if item.is_file():
                    return True
                elif item.is_dir() and has_includable_content(item, include_patterns, append_patterns, ignore_patterns, gitignore_patterns, visited_dirs):
                    return True
    except (PermissionError, OSError):
        return False
    return False

def add_to_tree(current_path: Path, tree: Tree):
    try:
        items = sorted(current_path.iterdir(), key=lambda p: (p.is_file(), p.name))
    except (PermissionError, OSError):
        tree.add('[red]Error: Cannot access directory[/red]')
        return
    for item in items:
        if not should_process_path(item, include_patterns, append_patterns, ignore_patterns, gitignore_patterns):
            continue
        if item.is_file():
            tree.add(f'📄 {item.name}')
        elif item.is_dir():
            if should_ignore(item, ignore_patterns, gitignore_patterns) and (not should_force_include(item, append_patterns)):
                if not has_includable_content(item, include_patterns, append_patterns, ignore_patterns, gitignore_patterns):
                    continue
            branch = tree.add(f'📁 {item.name}')
            add_to_tree(item, branch)

def create_tree_structure(path: Path, include_patterns: List[str], append_patterns: List[str], ignore_patterns: List[str], gitignore_patterns: List[str]) -> Tree:
    """Create a rich Tree representation of the directory structure."""
    tree = Tree(f'📁 {path.name}')

    def add_to_tree(current_path: Path, tree: Tree):
        try:
            items = sorted(current_path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except (PermissionError, OSError):
            tree.add('[red]Error: Cannot access directory[/red]')
            return
        for item in items:
            if not should_process_path(item, include_patterns, append_patterns, ignore_patterns, gitignore_patterns):
                continue
            if item.is_file():
                tree.add(f'📄 {item.name}')
            elif item.is_dir():
                if should_ignore(item, ignore_patterns, gitignore_patterns) and (not should_force_include(item, append_patterns)):
                    if not has_includable_content(item, include_patterns, append_patterns, ignore_patterns, gitignore_patterns):
                        continue
                branch = tree.add(f'📁 {item.name}')
                add_to_tree(item, branch)
    add_to_tree(path, tree)
    return tree

def package_project(path: Path, output_file: Path, include_patterns: List[str], append_patterns: List[str], ignore_patterns: List[str], gitignore_patterns: List[str]) -> None:
    """Package project files into a single markdown file."""
    include_patterns = [normalize_pattern(p) for p in include_patterns]
    append_patterns = [normalize_pattern(p) for p in append_patterns]
    ignore_patterns = [normalize_pattern(p) for p in ignore_patterns]
    gitignore_patterns = [normalize_pattern(p) for p in gitignore_patterns]
    print(f'Include patterns: {include_patterns}')
    print(f'Append patterns: {append_patterns}')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f'# Project: {path.name}\n\n')
        f.write('## Directory Structure\n\n')
        f.write('```\n')
        console = Console(file=None)
        with console.capture() as capture:
            console.print(create_tree_structure(path, include_patterns, append_patterns, ignore_patterns, gitignore_patterns))
        f.write(capture.get())
        f.write('```\n\n')
        f.write('## File Contents\n\n')

        def write_files(current_path: Path):
            try:
                items = sorted(current_path.iterdir(), key=lambda p: (p.is_file(), p.name))
            except (PermissionError, OSError):
                f.write(f'### Error accessing {current_path.relative_to(path)}\n\n')
                f.write('```\nPermission denied or I/O error\n```\n\n')
                return
            for item in items:
                if not should_process_path(item, include_patterns, append_patterns, ignore_patterns, gitignore_patterns):
                    continue
                if item.is_file():
                    try:
                        with open(item, 'r', encoding='utf-8') as source_file:
                            content = source_file.read()
                            f.write(f'### {item.relative_to(path)}\n\n')
                            f.write('```')
                            if item.suffix:
                                f.write(item.suffix[1:])
                            f.write('\n')
                            f.write(content)
                            f.write('\n```\n\n')
                    except UnicodeDecodeError:
                        f.write(f'### {item.relative_to(path)}\n\n')
                        f.write('```\nBinary file not included\n```\n\n')
                    except (PermissionError, OSError):
                        f.write(f'### {item.relative_to(path)}\n\n')
                        f.write('```\nError: Cannot read file\n```\n\n')
                elif item.is_dir():
                    if should_ignore(item, ignore_patterns, gitignore_patterns) and (not should_force_include(item, append_patterns)):
                        if not has_includable_content(item, include_patterns, append_patterns, ignore_patterns, gitignore_patterns):
                            continue
                    write_files(item)
        write_files(path)

def write_files(current_path: Path):
    try:
        items = sorted(current_path.iterdir(), key=lambda p: (p.is_file(), p.name))
    except (PermissionError, OSError):
        f.write(f'### Error accessing {current_path.relative_to(path)}\n\n')
        f.write('```\nPermission denied or I/O error\n```\n\n')
        return
    for item in items:
        if not should_process_path(item, include_patterns, append_patterns, ignore_patterns, gitignore_patterns):
            continue
        if item.is_file():
            try:
                with open(item, 'r', encoding='utf-8') as source_file:
                    content = source_file.read()
                    f.write(f'### {item.relative_to(path)}\n\n')
                    f.write('```')
                    if item.suffix:
                        f.write(item.suffix[1:])
                    f.write('\n')
                    f.write(content)
                    f.write('\n```\n\n')
            except UnicodeDecodeError:
                f.write(f'### {item.relative_to(path)}\n\n')
                f.write('```\nBinary file not included\n```\n\n')
            except (PermissionError, OSError):
                f.write(f'### {item.relative_to(path)}\n\n')
                f.write('```\nError: Cannot read file\n```\n\n')
        elif item.is_dir():
            if should_ignore(item, ignore_patterns, gitignore_patterns) and (not should_force_include(item, append_patterns)):
                if not has_includable_content(item, include_patterns, append_patterns, ignore_patterns, gitignore_patterns):
                    continue
            write_files(item)

def main(path: str=typer.Argument('.', help='Path to the project directory'), output: str=typer.Option('prompt.md', '--output', '-o', help='Output file path'), include: Optional[List[str]]=typer.Option(None, '--include', '-i', help="Patterns to ONLY include (e.g. '*.py')"), append_include: Optional[List[str]]=typer.Option(None, '--append-include', '-a', help='Additional patterns to include (has precedence over -i and -x)'), ignore: Optional[List[str]]=typer.Option(None, '--ignore', '-x', help='Patterns to ignore'), skip_gitignore: bool=typer.Option(False, '--skip-gitignore', help='Skip reading .gitignore patterns')):
    """
    Package project files into a single markdown file with directory structure.

    Precedence rules:
    1. -a (--append-include): Always include these patterns
    2. -i (--include): Include ONLY these patterns (unless -a is also specified)
    3. -x (--ignore): Ignore these patterns (unless they match -i or -a)
    """
    project_path = Path(path).resolve()
    output_path = Path(output).resolve()
    if not project_path.exists():
        typer.echo(f"Error: Project path '{path}' does not exist")
        raise typer.Exit(1)
    gitignore_patterns = [] if skip_gitignore else parse_gitignore(project_path)
    include_patterns = include or []
    ignore_patterns = ignore or []
    append_include_patterns = append_include or []
    default_ignores = ['**/__pycache__/**', '**/*.pyc', '**/.coverage', '**/.pytest_cache/**', '**/.ruff_cache/**', '**/.git/**', '**/.github/**', '**/.idea/**', '**/.vscode/**', '**/.venv/**', '**/venv/**', '**/env/**', '**/uv.lock', '**/.pre-commit-config.yaml', '**/.python-version', '**/.gitignore', '**/data/**', '**/dist/**', '**/examples/**', '**/htmlcov/**', '**/schema/**', '**/scripts/**', '**/tests/**', '**/LICENSE', '**/CONTRIBUTING.md', '**/CLAUDE.md', '**/README.md', '**/LLMS.txt', '**/Makefile', '**/pyproject.toml', '**/requirements.txt', '**/mcp_agent.config.yaml', '**/mcp_agent.secrets.yaml', '**/mcp_agent.config.yaml.example', '**/prompt.md', '**/.DS_Store', '**/py.typed']
    ignore_patterns.extend(default_ignores)
    typer.echo(f'Packaging project from: {project_path}')
    typer.echo(f'Output file: {output_path}')
    if include_patterns:
        typer.echo(f'Include ONLY patterns: {include_patterns}')
    if append_include_patterns:
        typer.echo(f'Additional include patterns: {append_include_patterns}')
    typer.echo(f'Ignoring {len(ignore_patterns)} patterns (default + custom)')
    if not skip_gitignore and gitignore_patterns:
        typer.echo(f'Using .gitignore with {len(gitignore_patterns)} patterns')
    try:
        package_project(project_path, output_path, include_patterns, append_include_patterns, ignore_patterns, gitignore_patterns)
        typer.echo(f'Successfully packaged project to {output_path}')
    except Exception as e:
        typer.echo(f'Error packaging project: {str(e)}')
        raise typer.Exit(1)

def parse_gitignore(path: Path) -> List[str]:
    """Parse .gitignore file and return list of patterns."""
    gitigore_path = path / '.gitignore'
    if not gitigore_path.exists():
        return []
    with open(file=gitigore_path, mode='r', encoding='utf-8') as f:
        patterns = [line.strip() for line in f if line.strip() and (not line.startswith('#'))]
    return patterns

