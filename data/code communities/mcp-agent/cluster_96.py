# Cluster 96

def test_create_pathspec_from_gitignore(tmp_path):
    """`create_pathspec_from_gitignore` should parse patterns into a matcher.

    Writes a temporary ignore file, loads it into a `PathSpec`, and asserts the
    resulting matcher includes and excludes representative paths.
    """
    ignore_path = tmp_path / '.mcpacignore'
    ignore_path.write_text('*.log\nbuild/\n')
    spec = create_pathspec_from_gitignore(ignore_path)
    assert spec is not None
    assert spec.match_file('debug.log')
    assert spec.match_file('build/output.txt')
    assert not spec.match_file('main.py')

def create_pathspec_from_gitignore(ignore_file_path: Path) -> Optional[pathspec.PathSpec]:
    """Create and return a `PathSpec` from an ignore file.

    The file is parsed using the `gitwildmatch` (gitignore) syntax. If the file
    does not exist, `None` is returned so callers can fall back to default
    behavior.

    Args:
        ignore_file_path: Path to the ignore file (e.g., `.mcpacignore`).

    Returns:
        A `PathSpec` that can match file/directory paths, or `None`.
    """
    if not ignore_file_path.exists():
        return None
    with open(ignore_file_path, 'r', encoding='utf-8') as f:
        spec = pathspec.PathSpec.from_lines('gitwildmatch', f)
    return spec

def test_create_pathspec_from_gitignore_missing_file(tmp_path):
    """Missing ignore files must return `None`.

    Ensures callers can detect the absence of an ignore file and fall back to
    default behaviour without raising.
    """
    missing_path = tmp_path / '.doesnotexist'
    assert create_pathspec_from_gitignore(missing_path) is None

