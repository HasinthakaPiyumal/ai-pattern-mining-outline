# Cluster 95

def test_should_ignore_by_gitignore():
    """Exercise ignore matching for mixed files and directories.

    Builds a `PathSpec` with file globs and directory suffixes and verifies the
    adapter returns only the names that match those patterns, covering the
    core filtering logic used during bundle copies.
    """
    gitignore_content = '*.log\n*.pyc\nnode_modules/\ntemp/\nbuild/\n'
    spec = pathspec.PathSpec.from_lines('gitwildmatch', gitignore_content.splitlines())
    project_dir = Path('/fake/project')
    current_path = str(project_dir)
    names = ['test.log', 'main.py', 'node_modules', 'config.yaml', 'test.pyc']
    original_is_dir = Path.is_dir
    Path.is_dir = lambda self: self.name in ['node_modules', 'temp', 'build']
    try:
        ignored = should_ignore_by_gitignore(current_path, names, project_dir, spec)
    finally:
        Path.is_dir = original_is_dir
    assert 'test.log' in ignored
    assert 'test.pyc' in ignored
    assert 'node_modules' in ignored
    assert 'main.py' not in ignored
    assert 'config.yaml' not in ignored

def should_ignore_by_gitignore(path_str: str, names: list, project_dir: Path, spec: Optional[pathspec.PathSpec]) -> Set[str]:
    """Return the subset of `names` to ignore for `shutil.copytree`.

    This function is designed to be passed as the `ignore` callback to
    `shutil.copytree`. For each entry in the current directory (`path_str`), it
    computes the path relative to the `project_dir` root and checks it against
    the provided `spec` (a `PathSpec` created from an ignore file).

    Notes:
    - If `spec` is `None`, this returns an empty set (no additional ignores).
    - For directories, we also check the relative path with a trailing slash
      (a common gitignore convention).
    """
    if spec is None:
        return set()
    ignored: Set[str] = set()
    current_path = Path(path_str)
    for name in names:
        full_path = current_path / name
        try:
            rel_path = full_path.relative_to(project_dir)
        except ValueError:
            continue
        rel_path_str = rel_path.as_posix()
        if spec.match_file(rel_path_str):
            ignored.add(name)
        elif full_path.is_dir() and spec.match_file(rel_path_str + '/'):
            ignored.add(name)
    return ignored

def test_should_ignore_by_gitignore_without_spec(tmp_path):
    """When no spec is provided the adapter should ignore nothing.

    Verifies the helper returns an empty set so the copy operation only applies
    the hard-coded exclusions.
    """
    project_dir = tmp_path
    (project_dir / 'data.txt').write_text('data')
    ignored = should_ignore_by_gitignore(str(project_dir), ['data.txt'], project_dir, spec=None)
    assert ignored == set()

def test_should_ignore_by_gitignore_matches_directories(tmp_path):
    """Directory patterns like `build/` must match folder names.

    Confirms the helper rewrites directory paths with a trailing slash when
    checking patterns so gitignore-style directory globs are honoured.
    """
    project_dir = tmp_path
    (project_dir / 'build').mkdir()
    spec = pathspec.PathSpec.from_lines('gitwildmatch', ['build/'])
    ignored = should_ignore_by_gitignore(str(project_dir), ['build'], project_dir, spec)
    assert 'build' in ignored

def test_should_ignore_by_gitignore_handles_nested_paths(tmp_path):
    """Nested patterns should be evaluated relative to the project root.

    Demonstrates that patterns such as `assets/*.txt` apply to files in a
    subdirectory while sparing siblings that do not match.
    """
    project_dir = tmp_path
    nested = project_dir / 'assets'
    nested.mkdir()
    (nested / 'notes.txt').write_text('notes')
    (nested / 'keep.md').write_text('keep')
    spec = pathspec.PathSpec.from_lines('gitwildmatch', ['assets/*.txt'])
    ignored = should_ignore_by_gitignore(str(nested), ['notes.txt', 'keep.md'], project_dir, spec)
    assert 'notes.txt' in ignored
    assert 'keep.md' not in ignored

