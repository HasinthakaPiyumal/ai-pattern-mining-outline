# Cluster 13

def _validate_and_resolve_paths(allowed_paths: List[Path], source_path: str, destination_path: str) -> tuple[Path, Path]:
    """
    Validate source and destination paths for copy operations.

    Args:
        allowed_paths: List of allowed base paths for validation
        source_path: Source file/directory path
        destination_path: Destination path in workspace

    Returns:
        Tuple of (resolved_source, resolved_destination)

    Raises:
        ValueError: If paths are invalid
    """
    try:
        source = Path(source_path).resolve()
        if not source.exists():
            raise ValueError(f'Source path does not exist: {source}')
        _validate_path_access(source, allowed_paths)
        if Path(destination_path).is_absolute():
            destination = Path(destination_path).resolve()
        else:
            destination = (Path.cwd() / destination_path).resolve()
        _validate_path_access(destination, allowed_paths)
        return (source, destination)
    except Exception as e:
        raise ValueError(f'Path validation failed: {e}')

def _validate_path_access(path: Path, allowed_paths: List[Path]) -> None:
    """
    Validate that a path is within allowed directories.

    Args:
        path: Path to validate
        allowed_paths: List of allowed base paths

    Raises:
        ValueError: If path is not within allowed directories
    """
    if not allowed_paths:
        return
    for allowed_path in allowed_paths:
        try:
            path.relative_to(allowed_path)
            return
        except ValueError:
            continue
    raise ValueError(f'Path not in allowed directories: {path}')

@mcp.tool()
def copy_file(source_path: str, destination_path: str, overwrite: bool=False) -> Dict[str, Any]:
    """
        Copy a file or directory from any accessible path to the agent's workspace.

        This is the primary tool for copying files from temp workspaces, context paths,
        or any other accessible location to the current agent's workspace.

        Args:
            source_path: Path to source file/directory (must be absolute path)
            destination_path: Destination path - can be:
                - Relative path: Resolved relative to your workspace (e.g., "output/file.txt")
                - Absolute path: Must be within allowed directories for security
            overwrite: Whether to overwrite existing files/directories (default: False)

        Returns:
            Dictionary with copy operation results
        """
    source, destination = _validate_and_resolve_paths(mcp.allowed_paths, source_path, destination_path)
    result = _perform_copy(source, destination, overwrite)
    return {'success': True, 'operation': 'copy_file', 'details': result}

def _perform_copy(source: Path, destination: Path, overwrite: bool=False) -> Dict[str, Any]:
    """
    Perform the actual copy operation.

    Args:
        source: Source path
        destination: Destination path
        overwrite: Whether to overwrite existing files

    Returns:
        Dict with operation results
    """
    try:
        if destination.exists() and (not overwrite):
            raise ValueError(f'Destination already exists (use overwrite=true): {destination}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            shutil.copy2(source, destination)
            return {'type': 'file', 'source': str(source), 'destination': str(destination), 'size': destination.stat().st_size}
        elif source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
            file_count = len([f for f in destination.rglob('*') if f.is_file()])
            return {'type': 'directory', 'source': str(source), 'destination': str(destination), 'file_count': file_count}
        else:
            raise ValueError(f'Source is neither file nor directory: {source}')
    except Exception as e:
        raise ValueError(f'Copy operation failed: {e}')

@mcp.tool()
def copy_files_batch(source_base_path: str, destination_base_path: str='', include_patterns: Optional[List[str]]=None, exclude_patterns: Optional[List[str]]=None, overwrite: bool=False) -> Dict[str, Any]:
    """
        Copy multiple files with pattern matching and exclusions.

        This advanced tool allows copying multiple files at once with glob-style patterns
        for inclusion and exclusion, useful for copying entire directory structures
        while filtering out unwanted files.

        Args:
            source_base_path: Base path to copy from (must be absolute path)
            destination_base_path: Base destination path - can be:
                - Relative path: Resolved relative to your workspace (e.g., "project/output")
                - Absolute path: Must be within allowed directories for security
                - Empty string: Copy to workspace root
            include_patterns: List of glob patterns for files to include (default: ["*"])
            exclude_patterns: List of glob patterns for files to exclude (default: [])
            overwrite: Whether to overwrite existing files (default: False)

        Returns:
            Dictionary with batch copy operation results
        """
    if include_patterns is None:
        include_patterns = ['*']
    if exclude_patterns is None:
        exclude_patterns = []
    try:
        copied_files = []
        skipped_files = []
        errors = []
        file_pairs = get_copy_file_pairs(mcp.allowed_paths, source_base_path, destination_base_path, include_patterns, exclude_patterns)
        for source_file, dest_file in file_pairs:
            rel_path_str = str(source_file.relative_to(Path(source_base_path).resolve()))
            try:
                if dest_file.exists() and (not overwrite):
                    skipped_files.append({'path': rel_path_str, 'reason': 'destination exists (overwrite=false)'})
                    continue
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, dest_file)
                copied_files.append({'source': str(source_file), 'destination': str(dest_file), 'relative_path': rel_path_str, 'size': dest_file.stat().st_size})
            except Exception as e:
                errors.append({'path': rel_path_str, 'error': str(e)})
        return {'success': True, 'operation': 'copy_files_batch', 'summary': {'copied': len(copied_files), 'skipped': len(skipped_files), 'errors': len(errors)}, 'details': {'copied_files': copied_files, 'skipped_files': skipped_files, 'errors': errors}}
    except Exception as e:
        return {'success': False, 'operation': 'copy_files_batch', 'error': str(e)}

def get_copy_file_pairs(allowed_paths: List[Path], source_base_path: str, destination_base_path: str='', include_patterns: Optional[List[str]]=None, exclude_patterns: Optional[List[str]]=None) -> List[Tuple[Path, Path]]:
    """
    Get all source->destination file pairs that would be copied by copy_files_batch.

    This function can be imported by the filesystem manager for permission validation.

    Args:
        allowed_paths: List of allowed base paths for validation
        source_base_path: Base path to copy from
        destination_base_path: Base path in workspace to copy to
        include_patterns: List of glob patterns for files to include
        exclude_patterns: List of glob patterns for files to exclude

    Returns:
        List of (source_path, destination_path) tuples

    Raises:
        ValueError: If paths are invalid
    """
    if include_patterns is None:
        include_patterns = ['*']
    if exclude_patterns is None:
        exclude_patterns = []
    source_base = Path(source_base_path).resolve()
    if not source_base.exists():
        raise ValueError(f'Source base path does not exist: {source_base}')
    _validate_path_access(source_base, allowed_paths)
    if destination_base_path:
        if Path(destination_base_path).is_absolute():
            dest_base = Path(destination_base_path).resolve()
        else:
            dest_base = (Path.cwd() / destination_base_path).resolve()
    else:
        raise ValueError('destination_base_path is required for copy_files_batch')
    _validate_path_access(dest_base, allowed_paths)
    file_pairs = []
    for item in source_base.rglob('*'):
        if not item.is_file():
            continue
        rel_path = item.relative_to(source_base)
        rel_path_str = str(rel_path)
        included = any((fnmatch.fnmatch(rel_path_str, pattern) for pattern in include_patterns))
        if not included:
            continue
        excluded = any((fnmatch.fnmatch(rel_path_str, pattern) for pattern in exclude_patterns))
        if excluded:
            continue
        dest_file = (dest_base / rel_path).resolve()
        _validate_path_access(dest_file, allowed_paths)
        file_pairs.append((item, dest_file))
    return file_pairs

@mcp.tool()
def delete_file(path: str, recursive: bool=False) -> Dict[str, Any]:
    """
        Delete a file or directory from the workspace.

        This tool allows agents to clean up outdated files or directories, helping maintain
        a clean workspace without cluttering it with old versions.

        Args:
            path: Path to file/directory to delete - can be:
                - Relative path: Resolved relative to your workspace (e.g., "old_file.txt")
                - Absolute path: Must be within allowed directories for security
            recursive: Whether to delete directories and their contents (default: False)
                      Required for non-empty directories

        Returns:
            Dictionary with deletion operation results

        Security:
            - Requires WRITE permission on path (validated by PathPermissionManager hook)
            - Must be within allowed directories
            - System files (.git, .env, etc.) cannot be deleted
            - Permission path roots themselves cannot be deleted
            - Protected paths specified in config are immune from deletion
        """
    try:
        if Path(path).is_absolute():
            target_path = Path(path).resolve()
        else:
            target_path = (Path.cwd() / path).resolve()
        _validate_path_access(target_path, mcp.allowed_paths)
        if not target_path.exists():
            return {'success': False, 'operation': 'delete_file', 'error': f'Path does not exist: {target_path}'}
        if _is_critical_path(target_path, mcp.allowed_paths):
            return {'success': False, 'operation': 'delete_file', 'error': f'Cannot delete critical system path: {target_path}'}
        if _is_permission_path_root(target_path, mcp.allowed_paths):
            return {'success': False, 'operation': 'delete_file', 'error': f'Cannot delete permission path root: {target_path}. You can delete files/directories within it, but not the root itself.'}
        if target_path.is_file():
            size = target_path.stat().st_size
            target_path.unlink()
            return {'success': True, 'operation': 'delete_file', 'details': {'type': 'file', 'path': str(target_path), 'size': size}}
        elif target_path.is_dir():
            if not recursive:
                if any(target_path.iterdir()):
                    return {'success': False, 'operation': 'delete_file', 'error': f'Directory not empty (use recursive=true): {target_path}'}
                target_path.rmdir()
            else:
                file_count = len([f for f in target_path.rglob('*') if f.is_file()])
                shutil.rmtree(target_path)
                return {'success': True, 'operation': 'delete_file', 'details': {'type': 'directory', 'path': str(target_path), 'file_count': file_count}}
            return {'success': True, 'operation': 'delete_file', 'details': {'type': 'directory', 'path': str(target_path)}}
        else:
            return {'success': False, 'operation': 'delete_file', 'error': f'Path is neither file nor directory: {target_path}'}
    except Exception as e:
        return {'success': False, 'operation': 'delete_file', 'error': str(e)}

def _is_critical_path(path: Path, allowed_paths: List[Path]=None) -> bool:
    """
    Check if a path is a critical system file that should not be deleted.

    Critical paths include:
    - .git directories (version control)
    - .env files (environment variables)
    - .massgen directories (MassGen metadata) - UNLESS within an allowed workspace
    - node_modules (package dependencies)
    - venv/.venv (Python virtual environments)
    - __pycache__ (Python cache)
    - massgen_logs (logging)

    Args:
        path: Path to check
        allowed_paths: List of allowed base paths (workspaces). If provided and path
                      is within an allowed path, only check for critical patterns
                      within that workspace (not in parent paths).

    Returns:
        True if path is critical and should not be deleted

    Examples:
        # Outside workspace - blocks any .massgen in path
        _is_critical_path(Path("/home/.massgen/config"))  → True (blocked)

        # Inside workspace - allows user files even if parent has .massgen
        workspace = Path("/home/.massgen/workspaces/workspace1")
        _is_critical_path(Path("/home/.massgen/workspaces/workspace1/user_dir"), [workspace])  → False (allowed)
        _is_critical_path(Path("/home/.massgen/workspaces/workspace1/.git"), [workspace])  → True (blocked)
    """
    CRITICAL_PATTERNS = ['.git', '.env', '.massgen', 'node_modules', '__pycache__', '.venv', 'venv', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'massgen_logs']
    resolved_path = path.resolve()
    if allowed_paths:
        for allowed_path in allowed_paths:
            try:
                rel_path = resolved_path.relative_to(allowed_path.resolve())
                for part in rel_path.parts:
                    if part in CRITICAL_PATTERNS:
                        return True
                if resolved_path.name in CRITICAL_PATTERNS:
                    return True
                return False
            except ValueError:
                continue
    parts = resolved_path.parts
    for part in parts:
        if part in CRITICAL_PATTERNS:
            return True
    if resolved_path.name in CRITICAL_PATTERNS:
        return True
    return False

def _is_permission_path_root(path: Path, allowed_paths: List[Path]) -> bool:
    """
    Check if a path is exactly one of the permission path roots.

    This prevents deletion of workspace directories, context path roots, etc.,
    while still allowing deletion of files and subdirectories within them.

    Args:
        path: Path to check
        allowed_paths: List of allowed base paths (permission path roots)

    Returns:
        True if path is exactly a permission path root

    Examples (Unix/macOS):
        allowed_paths = [Path("/workspace1"), Path("/context")]
        _is_permission_path_root(Path("/workspace1"))              → True  (blocked)
        _is_permission_path_root(Path("/workspace1/file.txt"))    → False (allowed)
        _is_permission_path_root(Path("/workspace1/subdir"))      → False (allowed)
        _is_permission_path_root(Path("/context"))                → True  (blocked)
        _is_permission_path_root(Path("/context/config.yaml"))    → False (allowed)

    Examples (Windows):
        allowed_paths = [Path("C:\\workspace1"), Path("D:\\context")]
        _is_permission_path_root(Path("C:\\workspace1"))           → True  (blocked)
        _is_permission_path_root(Path("C:\\workspace1\\file.txt")) → False (allowed)
        _is_permission_path_root(Path("D:\\context"))             → True  (blocked)
        _is_permission_path_root(Path("D:\\context\\data.json"))  → False (allowed)
    """
    resolved_path = path.resolve()
    for allowed_path in allowed_paths:
        if resolved_path == allowed_path.resolve():
            return True
    return False

@mcp.tool()
def delete_files_batch(base_path: str, include_patterns: Optional[List[str]]=None, exclude_patterns: Optional[List[str]]=None) -> Dict[str, Any]:
    """
        Delete multiple files matching patterns.

        This advanced tool allows deleting multiple files at once with glob-style patterns
        for inclusion and exclusion, useful for cleaning up entire directory structures
        while preserving specific files.

        Args:
            base_path: Base directory to search in - can be:
                - Relative path: Resolved relative to your workspace (e.g., "build")
                - Absolute path: Must be within allowed directories for security
            include_patterns: List of glob patterns for files to include (default: ["*"])
            exclude_patterns: List of glob patterns for files to exclude (default: [])

        Returns:
            Dictionary with batch deletion results including:
            - deleted: List of deleted files
            - skipped: List of skipped files (read-only or system files)
            - errors: List of errors encountered

        Security:
            - Requires WRITE permission on each file
            - Must be within allowed directories
            - System files (.git, .env, etc.) cannot be deleted
        """
    if include_patterns is None:
        include_patterns = ['*']
    if exclude_patterns is None:
        exclude_patterns = []
    try:
        deleted_files = []
        skipped_files = []
        errors = []
        if Path(base_path).is_absolute():
            base = Path(base_path).resolve()
        else:
            base = (Path.cwd() / base_path).resolve()
        if not base.exists():
            return {'success': False, 'operation': 'delete_files_batch', 'error': f'Base path does not exist: {base}'}
        _validate_path_access(base, mcp.allowed_paths)
        for item in base.rglob('*'):
            if not item.is_file():
                continue
            rel_path = item.relative_to(base)
            rel_path_str = str(rel_path)
            included = any((fnmatch.fnmatch(rel_path_str, pattern) for pattern in include_patterns))
            if not included:
                continue
            excluded = any((fnmatch.fnmatch(rel_path_str, pattern) for pattern in exclude_patterns))
            if excluded:
                continue
            try:
                if _is_critical_path(item, mcp.allowed_paths):
                    skipped_files.append({'path': rel_path_str, 'reason': 'system file (protected)'})
                    continue
                if _is_permission_path_root(item, mcp.allowed_paths):
                    skipped_files.append({'path': rel_path_str, 'reason': 'permission path root (protected)'})
                    continue
                _validate_path_access(item, mcp.allowed_paths)
                size = item.stat().st_size
                item.unlink()
                deleted_files.append({'path': str(item), 'relative_path': rel_path_str, 'size': size})
            except Exception as e:
                errors.append({'path': rel_path_str, 'error': str(e)})
        return {'success': True, 'operation': 'delete_files_batch', 'summary': {'deleted': len(deleted_files), 'skipped': len(skipped_files), 'errors': len(errors)}, 'details': {'deleted_files': deleted_files, 'skipped_files': skipped_files, 'errors': errors}}
    except Exception as e:
        return {'success': False, 'operation': 'delete_files_batch', 'error': str(e)}

@mcp.tool()
def compare_directories(dir1: str, dir2: str, show_content_diff: bool=False) -> Dict[str, Any]:
    """
        Compare two directories and show differences.

        This tool helps understand what changed between two workspaces or directory states,
        making it easier to review changes before deployment or understand agent modifications.

        Args:
            dir1: First directory path (absolute or relative to workspace)
            dir2: Second directory path (absolute or relative to workspace)
            show_content_diff: Whether to include unified diffs of different files (default: False)

        Returns:
            Dictionary with comparison results:
            - only_in_dir1: Files only in first directory
            - only_in_dir2: Files only in second directory
            - different: Files that exist in both but have different content
            - identical: Files that are identical
            - content_diffs: Optional unified diffs (if show_content_diff=True)

        Security:
            - Read-only operation, never modifies files
            - Both paths must be within allowed directories
        """
    try:
        path1 = Path(dir1).resolve() if Path(dir1).is_absolute() else (Path.cwd() / dir1).resolve()
        path2 = Path(dir2).resolve() if Path(dir2).is_absolute() else (Path.cwd() / dir2).resolve()
        _validate_path_access(path1, mcp.allowed_paths)
        _validate_path_access(path2, mcp.allowed_paths)
        if not path1.exists() or not path1.is_dir():
            return {'success': False, 'operation': 'compare_directories', 'error': f'First path is not a directory: {path1}'}
        if not path2.exists() or not path2.is_dir():
            return {'success': False, 'operation': 'compare_directories', 'error': f'Second path is not a directory: {path2}'}
        dcmp = filecmp.dircmp(str(path1), str(path2))
        result = {'success': True, 'operation': 'compare_directories', 'details': {'only_in_dir1': list(dcmp.left_only), 'only_in_dir2': list(dcmp.right_only), 'different': list(dcmp.diff_files), 'identical': list(dcmp.same_files)}}
        if show_content_diff and dcmp.diff_files:
            content_diffs = {}
            for filename in dcmp.diff_files:
                file1 = path1 / filename
                file2 = path2 / filename
                try:
                    if _is_text_file(file1) and _is_text_file(file2):
                        with open(file1) as f1, open(file2) as f2:
                            lines1 = f1.readlines()
                            lines2 = f2.readlines()
                        diff = list(difflib.unified_diff(lines1, lines2, fromfile=f'dir1/{filename}', tofile=f'dir2/{filename}', lineterm=''))
                        content_diffs[filename] = '\n'.join(diff[:100])
                except Exception as e:
                    content_diffs[filename] = f'Error generating diff: {e}'
            result['details']['content_diffs'] = content_diffs
        return result
    except Exception as e:
        return {'success': False, 'operation': 'compare_directories', 'error': str(e)}

def _is_text_file(path: Path) -> bool:
    """
    Check if a file is likely a text file (not binary).

    Uses simple heuristic: try to read as text and check for null bytes.

    TODO: Handle multi-modal files once implemented.

    Args:
        path: Path to check

    Returns:
        True if file appears to be text
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            chunk = f.read(8192)
            if '\x00' in chunk:
                return False
        return True
    except (UnicodeDecodeError, OSError):
        return False

@mcp.tool()
def compare_files(file1: str, file2: str, context_lines: int=3) -> Dict[str, Any]:
    """
        Compare two text files and show unified diff.

        This tool provides detailed line-by-line comparison of two files,
        making it easy to see exactly what changed between versions.

        Args:
            file1: First file path (absolute or relative to workspace)
            file2: Second file path (absolute or relative to workspace)
            context_lines: Number of context lines around changes (default: 3)

        Returns:
            Dictionary with comparison results:
            - identical: Boolean indicating if files are identical
            - diff: Unified diff output
            - stats: Statistics (lines added/removed/changed)

        Security:
            - Read-only operation, never modifies files
            - Both paths must be within allowed directories
            - Works best with text files
        """
    try:
        path1 = Path(file1).resolve() if Path(file1).is_absolute() else (Path.cwd() / file1).resolve()
        path2 = Path(file2).resolve() if Path(file2).is_absolute() else (Path.cwd() / file2).resolve()
        _validate_path_access(path1, mcp.allowed_paths)
        _validate_path_access(path2, mcp.allowed_paths)
        if not path1.exists() or not path1.is_file():
            return {'success': False, 'operation': 'compare_files', 'error': f'First path is not a file: {path1}'}
        if not path2.exists() or not path2.is_file():
            return {'success': False, 'operation': 'compare_files', 'error': f'Second path is not a file: {path2}'}
        try:
            with open(path1) as f1:
                lines1 = f1.readlines()
            with open(path2) as f2:
                lines2 = f2.readlines()
        except UnicodeDecodeError:
            return {'success': False, 'operation': 'compare_files', 'error': 'Files appear to be binary, not text'}
        diff = list(difflib.unified_diff(lines1, lines2, fromfile=str(path1), tofile=str(path2), lineterm='', n=context_lines))
        added = sum((1 for line in diff if line.startswith('+') and (not line.startswith('+++'))))
        removed = sum((1 for line in diff if line.startswith('-') and (not line.startswith('---'))))
        return {'success': True, 'operation': 'compare_files', 'details': {'identical': len(diff) == 0, 'diff': '\n'.join(diff[:500]), 'stats': {'added': added, 'removed': removed, 'changed': min(added, removed)}}}
    except Exception as e:
        return {'success': False, 'operation': 'compare_files', 'error': str(e)}

@mcp.tool()
def generate_and_store_image_with_input_images(base_image_paths: List[str], prompt: str='Create a variation of the provided images', model: str='gpt-4.1', n: int=1, storage_path: Optional[str]=None) -> Dict[str, Any]:
    """
        Create variations based on multiple input images using OpenAI's gpt-4.1 API.

        This tool generates image variations based on multiple base images using OpenAI's gpt-4.1 API
        and saves them to the workspace with automatic organization.

        Args:
            base_image_paths: List of paths to base images (PNG/JPEG files, less than 4MB)
                        - Relative path: Resolved relative to workspace
                        - Absolute path: Must be within allowed directories
            prompt: Text description for the variation (default: "Create a variation of the provided images")
            model: Model to use (default: "gpt-4.1")
            n: Number of variations to generate (default: 1)
            storage_path: Directory path where to save variations (optional)
                         - Relative path: Resolved relative to workspace
                         - Absolute path: Must be within allowed directories
                         - None/empty: Saves to workspace root

        Returns:
            Dictionary containing:
            - success: Whether operation succeeded
            - operation: "generate_and_store_image_with_input_images"
            - note: Note about usage
            - images: List of generated images with file paths and metadata
            - model: Model used for generation
            - prompt: The prompt used
            - total_images: Total number of images generated

        Examples:
            generate_and_store_image_with_input_images(["cat.png", "dog.png"], "Combine these animals")
            → Generates a variation combining both images

            generate_and_store_image_with_input_images(["art/logo.png", "art/icon.png"], "Create a unified design")
            → Generates variations based on both images

        Security:
            - Requires valid OpenAI API key
            - Input images must be valid image files less than 4MB
            - Files are saved to specified path within workspace
        """
    from datetime import datetime
    try:
        script_dir = Path(__file__).parent.parent.parent
        env_path = script_dir / '.env'
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            return {'success': False, 'operation': 'generate_and_store_image_with_input_images', 'error': 'OpenAI API key not found. Please set OPENAI_API_KEY in .env file or environment variable.'}
        client = OpenAI(api_key=openai_api_key)
        content = [{'type': 'input_text', 'text': prompt}]
        validated_paths = []
        for image_path_str in base_image_paths:
            if Path(image_path_str).is_absolute():
                image_path = Path(image_path_str).resolve()
            else:
                image_path = (Path.cwd() / image_path_str).resolve()
            _validate_path_access(image_path, mcp.allowed_paths)
            if not image_path.exists():
                return {'success': False, 'operation': 'generate_and_store_image_with_input_images', 'error': f'Image file does not exist: {image_path}'}
            if image_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                return {'success': False, 'operation': 'generate_and_store_image_with_input_images', 'error': f'Image must be PNG or JPEG format: {image_path}'}
            file_size = image_path.stat().st_size
            if file_size > 4 * 1024 * 1024:
                return {'success': False, 'operation': 'generate_and_store_image_with_input_images', 'error': f'Image file too large (must be < 4MB): {image_path} is {file_size / (1024 * 1024):.2f}MB'}
            validated_paths.append(image_path)
            with open(image_path, 'rb') as f:
                image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            mime_type = 'image/jpeg' if image_path.suffix.lower() in ['.jpg', '.jpeg'] else 'image/png'
            content.append({'type': 'input_image', 'image_url': f'data:{mime_type};base64,{image_base64}'})
        if storage_path:
            if Path(storage_path).is_absolute():
                storage_dir = Path(storage_path).resolve()
            else:
                storage_dir = (Path.cwd() / storage_path).resolve()
        else:
            storage_dir = Path.cwd()
        _validate_path_access(storage_dir, mcp.allowed_paths)
        storage_dir.mkdir(parents=True, exist_ok=True)
        try:
            response = client.responses.create(model=model, input=[{'role': 'user', 'content': content}], tools=[{'type': 'image_generation'}])
            image_generation_calls = [output for output in response.output if output.type == 'image_generation_call']
            all_variations = []
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            for idx, output in enumerate(image_generation_calls):
                if hasattr(output, 'result'):
                    image_base64 = output.result
                    image_bytes = base64.b64decode(image_base64)
                    if len(image_generation_calls) > 1:
                        filename = f'variation_{idx + 1}_{timestamp}.png'
                    else:
                        filename = f'variation_{timestamp}.png'
                    file_path = storage_dir / filename
                    file_path.write_bytes(image_bytes)
                    all_variations.append({'source_images': [str(p) for p in validated_paths], 'file_path': str(file_path), 'filename': filename, 'size': len(image_bytes), 'index': idx})
            if not all_variations:
                text_outputs = [output.content for output in response.output if hasattr(output, 'content')]
                if text_outputs:
                    return {'success': False, 'operation': 'generate_and_store_image_with_input_images', 'error': f'No images generated. Response: {' '.join(text_outputs)}'}
        except Exception as api_error:
            return {'success': False, 'operation': 'generate_and_store_image_with_input_images', 'error': f'OpenAI API error: {str(api_error)}'}
        return {'success': True, 'operation': 'generate_and_store_image_with_input_images', 'note': 'If no input images were provided, you must use generate_and_store_image_no_input_images tool.', 'images': all_variations, 'model': model, 'prompt': prompt, 'total_images': len(all_variations)}
    except Exception as e:
        return {'success': False, 'operation': 'generate_and_store_image_with_input_images', 'error': f'Failed to generate variations: {str(e)}'}

@mcp.tool()
def generate_and_store_audio_no_input_audios(prompt: str, model: str='gpt-4o-audio-preview', voice: str='alloy', audio_format: str='wav', storage_path: Optional[str]=None) -> Dict[str, Any]:
    """
        Generate audio from text using OpenAI's gpt-4o-audio-preview model and store it in the workspace.

        This tool generates audio speech from text prompts using OpenAI's audio generation API
        and saves the audio files to the workspace with automatic organization.

        Args:
            prompt: Text content to convert to audio speech
            model: Model to use for generation (default: "gpt-4o-audio-preview")
            voice: Voice to use for audio generation (default: "alloy")
                   Options: "alloy", "echo", "fable", "onyx", "nova", "shimmer"
            audio_format: Audio format for output (default: "wav")
                         Options: "wav", "mp3", "opus", "aac", "flac"
            storage_path: Directory path where to save the audio (optional)
                         - Relative path: Resolved relative to workspace (e.g., "audio/generated")
                         - Absolute path: Must be within allowed directories
                         - None/empty: Saves to workspace root

        Returns:
            Dictionary containing:
            - success: Whether operation succeeded
            - operation: "generate_and_store_audio_no_input_audios"
            - audio_file: Generated audio file with path and metadata
            - model: Model used for generation
            - prompt: The prompt used for generation
            - voice: Voice used for generation
            - format: Audio format used

        Examples:
            generate_and_store_audio_no_input_audios("Is a golden retriever a good family dog?")
            → Generates and saves to: 20240115_143022_audio.wav

            generate_and_store_audio_no_input_audios("Hello world", voice="nova", audio_format="mp3")
            → Generates with nova voice and saves as: 20240115_143022_audio.mp3

        Security:
            - Requires valid OpenAI API key (automatically detected from .env or environment)
            - Files are saved to specified path within workspace
            - Path must be within allowed directories
        """
    from datetime import datetime
    try:
        script_dir = Path(__file__).parent.parent.parent
        env_path = script_dir / '.env'
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            return {'success': False, 'operation': 'generate_and_store_audio_no_input_audios', 'error': 'OpenAI API key not found. Please set OPENAI_API_KEY in .env file or environment variable.'}
        client = OpenAI(api_key=openai_api_key)
        if storage_path:
            if Path(storage_path).is_absolute():
                storage_dir = Path(storage_path).resolve()
            else:
                storage_dir = (Path.cwd() / storage_path).resolve()
        else:
            storage_dir = Path.cwd()
        _validate_path_access(storage_dir, mcp.allowed_paths)
        storage_dir.mkdir(parents=True, exist_ok=True)
        try:
            completion = client.chat.completions.create(model=model, modalities=['text', 'audio'], audio={'voice': voice, 'format': audio_format}, messages=[{'role': 'user', 'content': prompt}])
            if not completion.choices[0].message.audio or not completion.choices[0].message.audio.data:
                return {'success': False, 'operation': 'generate_and_store_audio_no_input_audios', 'error': 'No audio data received from API'}
            audio_bytes = base64.b64decode(completion.choices[0].message.audio.data)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            clean_prompt = ''.join((c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_'))).strip()
            clean_prompt = clean_prompt.replace(' ', '_')
            filename = f'{timestamp}_{clean_prompt}.{audio_format}'
            file_path = storage_dir / filename
            file_path.write_bytes(audio_bytes)
            file_size = len(audio_bytes)
            text_response = completion.choices[0].message.content if completion.choices[0].message.content else None
            return {'success': True, 'operation': 'generate_and_store_audio_no_input_audios', 'audio_file': {'file_path': str(file_path), 'filename': filename, 'size': file_size, 'format': audio_format}, 'model': model, 'prompt': prompt, 'voice': voice, 'format': audio_format, 'text_response': text_response}
        except Exception as api_error:
            return {'success': False, 'operation': 'generate_and_store_audio_no_input_audios', 'error': f'OpenAI API error: {str(api_error)}'}
    except Exception as e:
        return {'success': False, 'operation': 'generate_and_store_audio_no_input_audios', 'error': f'Failed to generate or save audio: {str(e)}'}

@mcp.tool()
def generate_and_store_image_no_input_images(prompt: str, model: str='gpt-4.1', storage_path: Optional[str]=None) -> Dict[str, Any]:
    """
        Generate image using OpenAI's response with gpt-4.1 **WITHOUT ANY INPUT IMAGES** and store it in the workspace.

        This tool Generate image using OpenAI's response with gpt-4.1 **WITHOUT ANY INPUT IMAGES** and store it in the workspace.

        Args:
            prompt: Text description of the image to generate
            model: Model to use for generation (default: "gpt-4.1")
                   Options: "gpt-4.1"
            n: Number of images to generate (default: 1)
               - gpt-4.1: only 1
            storage_path: Directory path where to save the image (optional)
                         - Relative path: Resolved relative to workspace (e.g., "images/generated")
                         - Absolute path: Must be within allowed directories
                         - None/empty: Saves to workspace root

        Returns:
            Dictionary containing:
            - success: Whether operation succeeded
            - operation: "generate_and_store_image_no_input_images"
            - note: Note about operation
            - images: List of generated images with file paths and metadata
            - model: Model used for generation
            - prompt: The prompt used for generation
            - total_images: Total number of images generated and saved
            - images: List of generated images with file paths and metadata

        Examples:
            generate_and_store_image_no_input_images("a cat in space")
            → Generates and saves to: 20240115_143022_a_cat_in_space.png

            generate_and_store_image_no_input_images("sunset over mountains", storage_path="art/landscapes")
            → Generates and saves to: art/landscapes/20240115_143022_sunset_over_mountains.png

        Security:
            - Requires valid OpenAI API key (automatically detected from .env or environment)
            - Files are saved to specified path within workspace
            - Path must be within allowed directories

        Note:
            API key is automatically detected in this order:
            1. First checks .env file in current directory or parent directories
            2. Then checks environment variables
        """
    from datetime import datetime
    try:
        script_dir = Path(__file__).parent.parent.parent
        env_path = script_dir / '.env'
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            return {'success': False, 'operation': 'generate_and_store_image', 'error': 'OpenAI API key not found. Please set OPENAI_API_KEY in .env file or environment variable.'}
        client = OpenAI(api_key=openai_api_key)
        if storage_path:
            if Path(storage_path).is_absolute():
                storage_dir = Path(storage_path).resolve()
            else:
                storage_dir = (Path.cwd() / storage_path).resolve()
        else:
            storage_dir = Path.cwd()
        _validate_path_access(storage_dir, mcp.allowed_paths)
        storage_dir.mkdir(parents=True, exist_ok=True)
        try:
            response = client.responses.create(model=model, input=prompt, tools=[{'type': 'image_generation'}])
            image_data = [output.result for output in response.output if output.type == 'image_generation_call']
            saved_images = []
            if image_data:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                clean_prompt = ''.join((c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_'))).strip()
                clean_prompt = clean_prompt.replace(' ', '_')
                for idx, image_base64 in enumerate(image_data):
                    image_bytes = base64.b64decode(image_base64)
                    if len(image_data) > 1:
                        filename = f'{timestamp}_{clean_prompt}_{idx + 1}.png'
                    else:
                        filename = f'{timestamp}_{clean_prompt}.png'
                    file_path = storage_dir / filename
                    file_path.write_bytes(image_bytes)
                    file_size = len(image_bytes)
                    saved_images.append({'file_path': str(file_path), 'filename': filename, 'size': file_size, 'index': idx})
            result = {'success': True, 'operation': 'generate_and_store_image_no_input_images', 'note': 'New images are generated and saved to the specified path.', 'images': saved_images, 'model': model, 'prompt': prompt, 'total_images': len(saved_images)}
            return result
        except Exception as api_error:
            print(f'OpenAI API error: {str(api_error)}')
            return {'success': False, 'operation': 'generate_and_store_image_no_input_images', 'error': f'OpenAI API error: {str(api_error)}'}
    except Exception as e:
        return {'success': False, 'operation': 'generate_and_store_image_no_input_images', 'error': f'Failed to generate or save image: {str(e)}'}

@mcp.tool()
def generate_text_with_input_audio(audio_paths: List[str], model: str='gpt-4o-transcribe') -> Dict[str, Any]:
    """
        Transcribe audio file(s) to text using OpenAI's Transcription API.

        This tool processes one or more audio files through OpenAI's Transcription API
        to extract the text content from the audio. Each file is processed separately.

        Args:
            audio_paths: List of paths to input audio files (WAV, MP3, M4A, etc.)
                        - Relative path: Resolved relative to workspace
                        - Absolute path: Must be within allowed directories
            model: Model to use (default: "gpt-4o-transcribe")

        Returns:
            Dictionary containing:
            - success: Whether operation succeeded
            - operation: "generate_text_with_input_audio"
            - transcriptions: List of transcription results for each file
            - audio_files: List of paths to the input audio files
            - model: Model used

        Examples:
            generate_text_with_input_audio(["recording.wav"])
            → Returns transcription for recording.wav

            generate_text_with_input_audio(["interview1.mp3", "interview2.mp3"])
            → Returns separate transcriptions for each file

        Security:
            - Requires valid OpenAI API key
            - All input audio files must exist and be readable
        """
    try:
        script_dir = Path(__file__).parent.parent.parent
        env_path = script_dir / '.env'
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            return {'success': False, 'operation': 'generate_text_with_input_audio', 'error': 'OpenAI API key not found. Please set OPENAI_API_KEY in .env file or environment variable.'}
        client = OpenAI(api_key=openai_api_key)
        validated_audio_paths = []
        audio_extensions = ['.wav', '.mp3', '.m4a', '.mp4', '.ogg', '.flac', '.aac', '.wma', '.opus']
        for audio_path_str in audio_paths:
            if Path(audio_path_str).is_absolute():
                audio_path = Path(audio_path_str).resolve()
            else:
                audio_path = (Path.cwd() / audio_path_str).resolve()
            _validate_path_access(audio_path, mcp.allowed_paths)
            if not audio_path.exists():
                return {'success': False, 'operation': 'generate_text_with_input_audio', 'error': f'Audio file does not exist: {audio_path}'}
            if audio_path.suffix.lower() not in audio_extensions:
                return {'success': False, 'operation': 'generate_text_with_input_audio', 'error': f'File does not appear to be an audio file: {audio_path}'}
            validated_audio_paths.append(audio_path)
        transcriptions = []
        for audio_path in validated_audio_paths:
            try:
                with open(audio_path, 'rb') as audio_file:
                    transcription = client.audio.transcriptions.create(model=model, file=audio_file, response_format='text')
                transcriptions.append({'file': str(audio_path), 'transcription': transcription})
            except Exception as api_error:
                return {'success': False, 'operation': 'generate_text_with_input_audio', 'error': f'Transcription API error for file {audio_path}: {str(api_error)}'}
        return {'success': True, 'operation': 'generate_text_with_input_audio', 'transcriptions': transcriptions, 'audio_files': [str(p) for p in validated_audio_paths], 'model': model}
    except Exception as e:
        return {'success': False, 'operation': 'generate_text_with_input_audio', 'error': f'Failed to transcribe audio: {str(e)}'}

@mcp.tool()
def convert_text_to_speech(input_text: str, model: str='gpt-4o-mini-tts', voice: str='alloy', instructions: Optional[str]=None, storage_path: Optional[str]=None, audio_format: str='mp3') -> Dict[str, Any]:
    """
        Convert text (transcription) directly to speech using OpenAI's TTS API with streaming response.

        This tool converts text directly to speech audio using OpenAI's Text-to-Speech API,
        designed specifically for converting transcriptions or any text content to spoken audio.
        Uses streaming response for efficient file handling.

        Args:
            input_text: The text content to convert to speech (e.g., transcription text)
            model: TTS model to use (default: "gpt-4o-mini-tts")
                   Options: "gpt-4o-mini-tts", "tts-1", "tts-1-hd"
            voice: Voice to use for speech synthesis (default: "alloy")
                   Options: "alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral", "sage"
            instructions: Optional speaking instructions for tone and style (e.g., "Speak in a cheerful tone")
            storage_path: Directory path where to save the audio file (optional)
                         - Relative path: Resolved relative to workspace
                         - Absolute path: Must be within allowed directories
                         - None/empty: Saves to workspace root
            audio_format: Output audio format (default: "mp3")
                         Options: "mp3", "opus", "aac", "flac", "wav", "pcm"

        Returns:
            Dictionary containing:
            - success: Whether operation succeeded
            - operation: "convert_text_to_speech"
            - audio_file: Generated audio file with path and metadata
            - model: TTS model used
            - voice: Voice used
            - format: Audio format used
            - text_length: Length of input text
            - instructions: Speaking instructions if provided

        Examples:
            convert_text_to_speech("Hello world, this is a test.")
            → Converts text to speech and saves as MP3

            convert_text_to_speech(
                "Today is a wonderful day to build something people love!",
                voice="coral",
                instructions="Speak in a cheerful and positive tone."
            )
            → Converts with specific voice and speaking instructions

        Security:
            - Requires valid OpenAI API key
            - Files are saved to specified path within workspace
            - Path must be within allowed directories
        """
    from datetime import datetime
    try:
        script_dir = Path(__file__).parent.parent.parent
        env_path = script_dir / '.env'
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            return {'success': False, 'operation': 'convert_text_to_speech', 'error': 'OpenAI API key not found. Please set OPENAI_API_KEY in .env file or environment variable.'}
        client = OpenAI(api_key=openai_api_key)
        if storage_path:
            if Path(storage_path).is_absolute():
                storage_dir = Path(storage_path).resolve()
            else:
                storage_dir = (Path.cwd() / storage_path).resolve()
        else:
            storage_dir = Path.cwd()
        _validate_path_access(storage_dir, mcp.allowed_paths)
        storage_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        clean_text = ''.join((c for c in input_text[:30] if c.isalnum() or c in (' ', '-', '_'))).strip()
        clean_text = clean_text.replace(' ', '_')
        filename = f'speech_{timestamp}_{clean_text}.{audio_format}'
        file_path = storage_dir / filename
        try:
            request_params = {'model': model, 'voice': voice, 'input': input_text}
            if instructions and model in ['gpt-4o-mini-tts']:
                request_params['instructions'] = instructions
            with client.audio.speech.with_streaming_response.create(**request_params) as response:
                response.stream_to_file(file_path)
            file_size = file_path.stat().st_size
            return {'success': True, 'operation': 'convert_text_to_speech', 'audio_file': {'file_path': str(file_path), 'filename': filename, 'size': file_size, 'format': audio_format}, 'model': model, 'voice': voice, 'format': audio_format, 'text_length': len(input_text), 'instructions': instructions if instructions else None}
        except Exception as api_error:
            return {'success': False, 'operation': 'convert_text_to_speech', 'error': f'OpenAI TTS API error: {str(api_error)}'}
    except Exception as e:
        return {'success': False, 'operation': 'convert_text_to_speech', 'error': f'Failed to convert text to speech: {str(e)}'}

@mcp.tool()
def generate_and_store_video_no_input_images(prompt: str, model: str='sora-2', seconds: int=4, storage_path: Optional[str]=None) -> Dict[str, Any]:
    """
        Generate a video from a text prompt using OpenAI's Sora-2 API.

        This tool generates a video based on a text prompt using OpenAI's Sora-2 API
        and saves it to the workspace with automatic organization.

        Args:
            prompt: Text description for the video to generate
            model: Model to use (default: "sora-2")
            storage_path: Directory path where to save the video (optional)
                         - Relative path: Resolved relative to workspace
                         - Absolute path: Must be within allowed directories
                         - None/empty: Saves to workspace root

        Returns:
            Dictionary containing:
            - success: Whether operation succeeded
            - operation: "generate_and_store_video_no_input_images"
            - video_path: Path to the saved video file
            - model: Model used for generation
            - prompt: The prompt used
            - duration: Time taken for generation in seconds

        Examples:
            generate_and_store_video_no_input_images("A cool cat on a motorcycle in the night")
            → Generates a video and saves to workspace root

            generate_and_store_video_no_input_images("Dancing robot", storage_path="videos/")
            → Generates a video and saves to videos/ directory

        Security:
            - Requires valid OpenAI API key with Sora-2 access
            - Files are saved to specified path within workspace
        """
    import time
    from datetime import datetime
    try:
        script_dir = Path(__file__).parent.parent.parent
        env_path = script_dir / '.env'
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            return {'success': False, 'operation': 'generate_and_store_video_no_input_images', 'error': 'OpenAI API key not found. Please set OPENAI_API_KEY in .env file or environment variable.'}
        client = OpenAI(api_key=openai_api_key)
        if storage_path:
            if Path(storage_path).is_absolute():
                storage_dir = Path(storage_path).resolve()
            else:
                storage_dir = (Path.cwd() / storage_path).resolve()
        else:
            storage_dir = Path.cwd()
        _validate_path_access(storage_dir, mcp.allowed_paths)
        storage_dir.mkdir(parents=True, exist_ok=True)
        try:
            start_time = time.time()
            video = client.videos.create(model=model, prompt=prompt, seconds=str(seconds))
            getattr(video, 'progress', 0)
            while video.status in ('in_progress', 'queued'):
                video = client.videos.retrieve(video.id)
                getattr(video, 'progress', 0)
                time.sleep(2)
            if video.status == 'failed':
                message = getattr(getattr(video, 'error', None), 'message', 'Video generation failed')
                return {'success': False, 'operation': 'generate_and_store_video_no_input_images', 'error': message}
            content = client.videos.download_content(video.id, variant='video')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            clean_prompt = ''.join((c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_'))).strip()
            clean_prompt = clean_prompt.replace(' ', '_')
            filename = f'{timestamp}_{clean_prompt}.mp4'
            file_path = storage_dir / filename
            content.write_to_file(str(file_path))
            duration = time.time() - start_time
            file_size = file_path.stat().st_size
            return {'success': True, 'operation': 'generate_and_store_video_no_input_images', 'video_path': str(file_path), 'filename': filename, 'size': file_size, 'model': model, 'prompt': prompt, 'duration': duration}
        except Exception as api_error:
            return {'success': False, 'operation': 'generate_and_store_video_no_input_images', 'error': f'OpenAI API error: {str(api_error)}'}
    except Exception as e:
        return {'success': False, 'operation': 'generate_and_store_video_no_input_images', 'error': f'Failed to generate or save video: {str(e)}'}

class PathPermissionManager:
    """
    Manages all filesystem paths and implements PreToolUse hook functionality similar to Claude Code,
    allowing us to intercept and validate tool calls based on some predefined rules (here, permissions).

    This manager handles all types of paths with unified permission control:
    - Workspace paths (typically write)
    - Temporary workspace paths (typically read-only)
    - Context paths (user-specified permissions)
    - Tool call validation (PreToolUse hook)
    - Path access control
    """
    DEFAULT_EXCLUDED_PATTERNS = ['.massgen', '.env', '.git', 'node_modules', '__pycache__', '.venv', 'venv', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.DS_Store', 'massgen_logs']

    def __init__(self, context_write_access_enabled: bool=False, enforce_read_before_delete: bool=True):
        """
        Initialize path permission manager.

        Args:
            context_write_access_enabled: Whether write access is enabled for context paths (workspace paths always
                have write access). If False, we change all context paths to read-only. Can be later updated with
                set_context_write_access_enabled(), in which case all existing context paths will be updated
                accordingly so that those that were "write" in YAML become writable again.
            enforce_read_before_delete: Whether to enforce read-before-delete policy for workspace files
        """
        self.managed_paths: List[ManagedPath] = []
        self.context_write_access_enabled = context_write_access_enabled
        self._permission_cache: Dict[Path, Permission] = {}
        self.file_operation_tracker = FileOperationTracker(enforce_read_before_delete=enforce_read_before_delete)
        logger.info(f'[PathPermissionManager] Initialized with context_write_access_enabled={context_write_access_enabled}, enforce_read_before_delete={enforce_read_before_delete}')

    def add_path(self, path: Path, permission: Permission, path_type: str) -> None:
        """
        Add a managed path.

        Args:
            path: Path to manage
            permission: Permission level for this path
            path_type: Type of path ("workspace", "temp_workspace", "context", etc.)
        """
        if not path.exists():
            if path_type == 'context':
                logger.warning(f'[PathPermissionManager] Context path does not exist: {path}')
                return
            else:
                logger.debug(f'[PathPermissionManager] Path will be created later: {path} ({path_type})')
        managed_path = ManagedPath(path=path.resolve(), permission=permission, path_type=path_type)
        self.managed_paths.append(managed_path)
        self._permission_cache.clear()
        logger.info(f'[PathPermissionManager] Added {path_type} path: {path} ({permission.value})')

    def get_context_paths(self) -> List[Dict[str, str]]:
        """
        Get context paths in configuration format for system prompts.

        Returns:
            List of context path dictionaries with path, permission, and will_be_writable flag
        """
        context_paths = []
        for mp in self.managed_paths:
            if mp.path_type == 'context':
                context_paths.append({'path': str(mp.path), 'permission': mp.permission.value, 'will_be_writable': mp.will_be_writable})
        return context_paths

    def set_context_write_access_enabled(self, enabled: bool) -> None:
        """
        Update write access setting for context paths and recalculate their permissions.
        Note: Workspace paths always have write access regardless of this setting.

        Args:
            enabled: Whether to enable write access for context paths
        """
        if self.context_write_access_enabled == enabled:
            return
        logger.info(f'[PathPermissionManager] Setting context_write_access_enabled to {enabled}')
        logger.info(f'[PathPermissionManager] Before update: self.managed_paths={self.managed_paths!r}')
        self.context_write_access_enabled = enabled
        for mp in self.managed_paths:
            if mp.path_type == 'context' and mp.will_be_writable:
                if enabled:
                    mp.permission = Permission.WRITE
                    logger.debug(f'[PathPermissionManager] Enabled write access for {mp.path}')
                else:
                    mp.permission = Permission.READ
                    logger.debug(f'[PathPermissionManager] Keeping read-only for {mp.path}')
        logger.info(f'[PathPermissionManager] Updated context path permissions based on context_write_access_enabled={enabled}, now is self.managed_paths={self.managed_paths!r}')
        self._permission_cache.clear()

    def add_context_paths(self, context_paths: List[Dict[str, Any]]) -> None:
        """
        Add context paths from configuration.

        Now supports both files and directories as context paths, with optional protected paths.

        Args:
            context_paths: List of context path configurations
                Format: [
                    {
                        "path": "C:/project/src",
                        "permission": "write",
                        "protected_paths": ["tests/do-not-touch/", "config.yaml"]  # Optional
                    },
                    {"path": "C:/project/logo.png", "permission": "read"}
                ]

        Note: During coordination, all context paths are read-only regardless of YAML settings.
              Only the final agent with context_write_access_enabled=True can write to paths marked as "write".
              Protected paths are ALWAYS read-only and immune from deletion, even if parent has write permission.
        """
        for config in context_paths:
            path_str = config.get('path', '')
            permission_str = config.get('permission', 'read')
            protected_paths_config = config.get('protected_paths', [])
            if not path_str:
                continue
            path = Path(path_str)
            if not path.exists():
                logger.warning(f'[PathPermissionManager] Context path does not exist: {path}')
                continue
            is_file = path.is_file()
            protected_paths = []
            for protected_str in protected_paths_config:
                protected_path = Path(protected_str)
                if not protected_path.is_absolute():
                    if is_file:
                        protected_path = (path.parent / protected_str).resolve()
                    else:
                        protected_path = (path / protected_str).resolve()
                else:
                    protected_path = protected_path.resolve()
                try:
                    if is_file:
                        protected_path.relative_to(path.parent.resolve())
                    else:
                        protected_path.relative_to(path.resolve())
                    protected_paths.append(protected_path)
                    logger.info(f'[PathPermissionManager] Added protected path: {protected_path}')
                except ValueError:
                    logger.warning(f'[PathPermissionManager] Protected path {protected_path} is not within context path {path}, skipping')
            if is_file:
                logger.info(f'[PathPermissionManager] Detected file context path: {path}')
                parent_dir = path.parent
                if not any((mp.path == parent_dir.resolve() and mp.path_type == 'file_context_parent' for mp in self.managed_paths)):
                    parent_managed = ManagedPath(path=parent_dir.resolve(), permission=Permission.READ, path_type='file_context_parent', will_be_writable=False, is_file=False)
                    self.managed_paths.append(parent_managed)
                    logger.debug(f'[PathPermissionManager] Added parent directory for file context: {parent_dir}')
            try:
                yaml_permission = Permission(permission_str.lower())
            except ValueError:
                logger.warning(f"[PathPermissionManager] Invalid permission '{permission_str}', using 'read'")
                yaml_permission = Permission.READ
            will_be_writable = yaml_permission == Permission.WRITE
            if self.context_write_access_enabled and will_be_writable:
                actual_permission = Permission.WRITE
                logger.debug(f'[PathPermissionManager] Final agent: context path {path} gets write permission')
            else:
                actual_permission = Permission.READ if will_be_writable else yaml_permission
                if will_be_writable:
                    logger.debug(f'[PathPermissionManager] Coordination agent: context path {path} read-only (will be writable later)')
            managed_path = ManagedPath(path=path.resolve(), permission=actual_permission, path_type='context', will_be_writable=will_be_writable, is_file=is_file, protected_paths=protected_paths)
            self.managed_paths.append(managed_path)
            self._permission_cache.clear()
            path_type_str = 'file' if is_file else 'directory'
            protected_count = len(protected_paths)
            logger.info(f'[PathPermissionManager] Added context {path_type_str}: {path} ({actual_permission.value}, will_be_writable: {will_be_writable}, protected_paths: {protected_count})')

    def add_previous_turn_paths(self, turn_paths: List[Dict[str, Any]]) -> None:
        """
        Add previous turn workspace paths for read access.
        These are tracked separately from regular context paths.

        Args:
            turn_paths: List of turn path configurations
                Format: [{"path": "/path/to/turn_1/workspace", "permission": "read"}, ...]
        """
        for config in turn_paths:
            path_str = config.get('path', '')
            if not path_str:
                continue
            path = Path(path_str).resolve()
            managed_path = ManagedPath(path=path, permission=Permission.READ, path_type='previous_turn', will_be_writable=False)
            self.managed_paths.append(managed_path)
            self._permission_cache.clear()
            logger.info(f'[PathPermissionManager] Added previous turn path: {path} (read-only)')

    def _is_excluded_path(self, path: Path) -> bool:
        """
        Check if a path matches any default excluded patterns.

        System files like .massgen/, .env, .git/ are always excluded from write access,
        EXCEPT when they are within a managed workspace path (which has explicit permissions).

        Args:
            path: Path to check

        Returns:
            True if path should be excluded from write access
        """
        for managed_path in self.managed_paths:
            if managed_path.path_type == 'workspace' and managed_path.contains(path):
                return False
        parts = path.parts
        for part in parts:
            if part in self.DEFAULT_EXCLUDED_PATTERNS:
                return True
        return False

    def get_permission(self, path: Path) -> Optional[Permission]:
        """
        Get permission level for a path.

        Now handles file-specific context paths correctly.

        Args:
            path: Path to check

        Returns:
            Permission level or None if path is not in context
        """
        resolved_path = path.resolve()
        if resolved_path in self._permission_cache:
            logger.debug(f'[PathPermissionManager] Permission cache hit for {resolved_path}: {self._permission_cache[resolved_path].value}')
            return self._permission_cache[resolved_path]
        if self._is_excluded_path(resolved_path):
            logger.info(f'[PathPermissionManager] Path {resolved_path} matches excluded pattern, forcing read-only')
            self._permission_cache[resolved_path] = Permission.READ
            return Permission.READ
        for managed_path in self.managed_paths:
            if managed_path.contains(resolved_path) and managed_path.is_protected(resolved_path):
                logger.info(f'[PathPermissionManager] Path {resolved_path} is protected, forcing read-only')
                self._permission_cache[resolved_path] = Permission.READ
                return Permission.READ
        file_paths = [mp for mp in self.managed_paths if mp.is_file]
        dir_paths = [mp for mp in self.managed_paths if not mp.is_file and mp.path_type != 'file_context_parent']
        for managed_path in file_paths:
            if managed_path.contains(resolved_path):
                logger.info(f'[PathPermissionManager] Found file-specific permission for {resolved_path}: {managed_path.permission.value} (from {managed_path.path}, type: {managed_path.path_type}, will_be_writable: {managed_path.will_be_writable})')
                self._permission_cache[resolved_path] = managed_path.permission
                return managed_path.permission
        sorted_dir_paths = sorted(dir_paths, key=lambda mp: len(mp.path.parts), reverse=True)
        for managed_path in sorted_dir_paths:
            if managed_path.contains(resolved_path) or managed_path.path == resolved_path:
                logger.info(f'[PathPermissionManager] Found permission for {resolved_path}: {managed_path.permission.value} (from {managed_path.path}, type: {managed_path.path_type}, will_be_writable: {managed_path.will_be_writable})')
                self._permission_cache[resolved_path] = managed_path.permission
                return managed_path.permission
        logger.debug(f'[PathPermissionManager] No permission found for {resolved_path} in managed paths: {[(str(mp.path), mp.permission.value, mp.path_type) for mp in self.managed_paths]}')
        return None

    async def pre_tool_use_hook(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        PreToolUse hook to validate tool calls based on permissions.

        This can be used directly with Claude Code SDK hooks or as validation
        for other backends that need manual tool call filtering.

        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments passed to the tool

        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
            - allowed: Whether the tool call should proceed
            - reason: Explanation if blocked (None if allowed)
        """
        if self._is_read_tool(tool_name):
            self._track_read_operation(tool_name, tool_args)
        if self._is_write_tool(tool_name):
            result = self._validate_write_tool(tool_name, tool_args)
            if result[0] and self._is_create_tool(tool_name):
                self._track_create_operation(tool_name, tool_args)
            return result
        if self._is_delete_tool(tool_name):
            return self._validate_delete_tool(tool_name, tool_args)
        command_tools = {'Bash', 'bash', 'shell', 'exec', 'execute_command'}
        if tool_name in command_tools:
            return self._validate_command_tool(tool_name, tool_args)
        return self._validate_file_context_access(tool_name, tool_args)

    def _is_write_tool(self, tool_name: str) -> bool:
        """
        Check if a tool is a write operation using pattern matching.

        Main Claude Code tools: Bash, Glob, Grep, Read, Edit, MultiEdit, Write, WebFetch, WebSearch

        This catches various write tools including:
        - Claude Code: Write, Edit, MultiEdit, NotebookEdit, etc.
        - MCP filesystem: write_file, edit_file, create_directory, move_file
        - Any other tools with write/edit/create/move in the name

        Note: Delete operations are handled separately by _is_delete_tool
        """
        write_patterns = ['.*[Ww]rite.*', '.*[Ee]dit.*', '.*[Cc]reate.*', '.*[Mm]ove.*', '.*[Cc]opy.*']
        for pattern in write_patterns:
            if re.match(pattern, tool_name):
                return True
        return False

    def _is_read_tool(self, tool_name: str) -> bool:
        """
        Check if a tool is a read operation that should be tracked.

        Uses substring matching to handle MCP prefixes (e.g., mcp__workspace_tools__compare_files)

        Tools that read file contents:
        - read/Read: File content reading (matches: Read, read_text_file, read_multimodal_files, etc.)
        - compare_files: File comparison
        - compare_directories: Directory comparison
        """
        tool_lower = tool_name.lower()
        read_keywords = ['compare_files', 'compare_directories']
        return any((keyword in tool_lower for keyword in read_keywords))

    def _is_delete_tool(self, tool_name: str) -> bool:
        """
        Check if a tool is a delete operation.

        Tools that delete files:
        - delete_file: Single file deletion
        - delete_files_batch: Batch file deletion
        - Any tool with delete/remove in the name
        """
        delete_patterns = ['.*[Dd]elete.*', '.*[Rr]emove.*']
        for pattern in delete_patterns:
            if re.match(pattern, tool_name):
                return True
        return False

    def _is_create_tool(self, tool_name: str) -> bool:
        """
        Check if a tool creates new files (for tracking created files).

        Tools that create files:
        - Write: Creates new files
        - write_file: MCP filesystem write
        - create_directory: Creates directories
        """
        create_patterns = ['.*[Ww]rite.*', '.*[Cc]reate.*']
        for pattern in create_patterns:
            if re.match(pattern, tool_name):
                return True
        return False

    def _track_read_operation(self, tool_name: str, tool_args: Dict[str, Any]) -> None:
        """
        Track files that are read by the agent.

        Uses substring matching to handle MCP prefixes consistently.

        Args:
            tool_name: Name of the read tool
            tool_args: Arguments passed to the tool
        """
        tool_lower = tool_name.lower()
        if 'compare_files' in tool_lower:
            file1 = tool_args.get('file1') or tool_args.get('file_path1')
            file2 = tool_args.get('file2') or tool_args.get('file_path2')
            if file1:
                path1 = self._resolve_path_against_workspace(file1)
                self.file_operation_tracker.mark_as_read(Path(path1))
            if file2:
                path2 = self._resolve_path_against_workspace(file2)
                self.file_operation_tracker.mark_as_read(Path(path2))
        elif 'compare_directories' in tool_lower:
            if tool_args.get('show_content_diff'):
                pass
        elif 'read_multiple_files' in tool_lower:
            paths = tool_args.get('paths', [])
            for file_path in paths:
                resolved_path = self._resolve_path_against_workspace(file_path)
                self.file_operation_tracker.mark_as_read(Path(resolved_path))
        else:
            file_path = self._extract_file_path(tool_args)
            if file_path:
                resolved_path = self._resolve_path_against_workspace(file_path)
                self.file_operation_tracker.mark_as_read(Path(resolved_path))

    def _track_create_operation(self, tool_name: str, tool_args: Dict[str, Any]) -> None:
        """
        Track files that are created by the agent.

        Args:
            tool_name: Name of the create tool
            tool_args: Arguments passed to the tool
        """
        file_path = self._extract_file_path(tool_args)
        if file_path:
            resolved_path = self._resolve_path_against_workspace(file_path)
            self.file_operation_tracker.mark_as_created(Path(resolved_path))

    def _validate_delete_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate delete tool operations using read-before-delete policy.

        Args:
            tool_name: Name of the delete tool
            tool_args: Arguments passed to the tool

        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
        """
        permission_result = self._validate_write_tool(tool_name, tool_args)
        if not permission_result[0]:
            return permission_result
        if tool_name == 'delete_files_batch':
            return self._validate_delete_files_batch(tool_args)
        file_path = self._extract_file_path(tool_args)
        if not file_path:
            return (True, None)
        resolved_path = self._resolve_path_against_workspace(file_path)
        path = Path(resolved_path)
        if path.is_dir():
            can_delete, reason = self.file_operation_tracker.can_delete_directory(path)
            if not can_delete:
                return (False, reason)
        else:
            can_delete, reason = self.file_operation_tracker.can_delete(path)
            if not can_delete:
                return (False, reason)
        return (True, None)

    def _validate_delete_files_batch(self, tool_args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate batch delete operations by checking all files that would be deleted.

        Args:
            tool_args: Arguments for delete_files_batch

        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
        """
        try:
            base_path = tool_args.get('base_path')
            include_patterns = tool_args.get('include_patterns') or ['*']
            exclude_patterns = tool_args.get('exclude_patterns') or []
            if not base_path:
                return (False, 'delete_files_batch requires base_path')
            resolved_base = self._resolve_path_against_workspace(base_path)
            base = Path(resolved_base)
            if not base.exists():
                return (True, None)
            unread_files = []
            for item in base.rglob('*'):
                if not item.is_file():
                    continue
                rel_path = item.relative_to(base)
                rel_path_str = str(rel_path)
                included = any((fnmatch.fnmatch(rel_path_str, pattern) for pattern in include_patterns))
                if not included:
                    continue
                excluded = any((fnmatch.fnmatch(rel_path_str, pattern) for pattern in exclude_patterns))
                if excluded:
                    continue
                if not self.file_operation_tracker.was_read(item):
                    unread_files.append(rel_path_str)
            if unread_files:
                example_files = unread_files[:3]
                suffix = f' (and {len(unread_files) - 3} more)' if len(unread_files) > 3 else ''
                reason = f'Cannot delete {len(unread_files)} unread file(s). Examples: {', '.join(example_files)}{suffix}. Please read files before deletion using Read or read_multimodal_files.'
                logger.info(f'[PathPermissionManager] Blocking batch delete: {reason}')
                return (False, reason)
            return (True, None)
        except Exception as e:
            logger.error(f'[PathPermissionManager] Error validating batch delete: {e}')
            return (False, f'Batch delete validation failed: {e}')

    def _is_path_within_allowed_directories(self, path: Path) -> bool:
        """
        Check if a path is within any allowed directory (workspace or context paths).

        This enforces directory boundaries - paths outside managed directories are not allowed.

        Args:
            path: Path to check

        Returns:
            True if path is within allowed directories, False otherwise
        """
        resolved_path = path.resolve()
        for managed_path in self.managed_paths:
            if managed_path.path_type == 'file_context_parent':
                continue
            if managed_path.contains(resolved_path) or managed_path.path == resolved_path:
                return True
        return False

    def _validate_file_context_access(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate access for all file operations - enforces directory boundaries and permissions.

        This method ensures that:
        1. ALL file operations are restricted to workspace + context paths (directory boundary)
        2. Read/write permissions are enforced within allowed directories
        3. Sibling file access is prevented for file-specific context paths

        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments passed to the tool

        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
        """
        file_path = self._extract_file_path(tool_args)
        if not file_path:
            return (True, None)
        file_path = self._resolve_path_against_workspace(file_path)
        path = Path(file_path).resolve()
        if not self._is_path_within_allowed_directories(path):
            logger.warning(f"[PathPermissionManager] BLOCKED: '{tool_name}' attempted to access path outside allowed directories: {path}")
            return (False, f"Access denied: '{path}' is outside allowed directories. Only workspace and context paths are accessible.")
        permission = self.get_permission(path)
        logger.debug(f"[PathPermissionManager] Validating '{tool_name}' on path: {path} with permission: {permission}")
        if permission is None:
            parent_paths = [mp for mp in self.managed_paths if mp.path_type == 'file_context_parent']
            for parent_mp in parent_paths:
                if parent_mp.contains(path):
                    return (False, f"Access denied: '{path}' is not an explicitly allowed file in this directory")
            return (True, None)
        return (True, None)

    def _validate_write_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate write tool access."""
        if tool_name == 'copy_files_batch':
            return self._validate_copy_files_batch(tool_args)
        file_path = self._extract_file_path(tool_args)
        if not file_path:
            return (True, None)
        file_path = self._resolve_path_against_workspace(file_path)
        path = Path(file_path).resolve()
        permission = self.get_permission(path)
        logger.debug(f"[PathPermissionManager] Validating write tool '{tool_name}' for path: {path} with permission: {permission}")
        if permission is None:
            parent_paths = [mp for mp in self.managed_paths if mp.path_type == 'file_context_parent']
            for parent_mp in parent_paths:
                if parent_mp.contains(path):
                    return (False, f"Access denied: '{path}' is not an explicitly allowed file in this directory")
            return (True, None)
        if permission == Permission.WRITE:
            return (True, None)
        else:
            return (False, f"No write permission for '{path}' (read-only context path)")

    def _resolve_path_against_workspace(self, path_str: str) -> str:
        """
        Resolve a path string against the workspace directory if it's relative.

        When MCP servers run with cwd set to workspace, they resolve relative paths
        against the workspace. This function does the same for validation purposes.

        Args:
            path_str: Path string that may be relative or absolute

        Returns:
            Absolute path string (resolved against workspace if relative)
        """
        if not path_str:
            return path_str
        if path_str.startswith('~'):
            path = Path(path_str).expanduser()
            return str(path)
        path = Path(path_str)
        if path.is_absolute():
            return path_str
        mcp_paths = self.get_mcp_filesystem_paths()
        if mcp_paths:
            workspace_path = Path(mcp_paths[0])
            resolved = workspace_path / path_str
            logger.debug(f"[PathPermissionManager] Resolved relative path '{path_str}' to '{resolved}'")
            return str(resolved)
        return path_str

    def _validate_copy_files_batch(self, tool_args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate copy_files_batch by checking all destination paths after globbing."""
        try:
            logger.debug(f'[PathPermissionManager] copy_files_batch validation - context_write_access_enabled: {self.context_write_access_enabled}')
            source_base_path = tool_args.get('source_base_path')
            destination_base_path = tool_args.get('destination_base_path', '')
            include_patterns = tool_args.get('include_patterns')
            exclude_patterns = tool_args.get('exclude_patterns')
            if not source_base_path:
                return (False, 'copy_files_batch requires source_base_path')
            destination_base_path = self._resolve_path_against_workspace(destination_base_path)
            file_pairs = get_copy_file_pairs(self.get_mcp_filesystem_paths(), source_base_path, destination_base_path, include_patterns, exclude_patterns)
            blocked_paths = []
            for source_file, dest_file in file_pairs:
                permission = self.get_permission(dest_file)
                logger.debug(f'[PathPermissionManager] copy_files_batch checking dest: {dest_file}, permission: {permission}')
                if permission == Permission.READ:
                    blocked_paths.append(str(dest_file))
            if blocked_paths:
                example_paths = blocked_paths[:3]
                suffix = f' (and {len(blocked_paths) - 3} more)' if len(blocked_paths) > 3 else ''
                return (False, f'No write permission for destination paths: {', '.join(example_paths)}{suffix}')
            return (True, None)
        except Exception as e:
            return (False, f'copy_files_batch validation failed: {e}')

    def _validate_command_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate command tool access.

        As of v0.0.20, only Claude Code supports execution.

        For Claude Code: Validates directory boundaries for all paths in Bash commands.
        This prevents access to paths outside workspace + context paths.

        """
        command = tool_args.get('command', '') or tool_args.get('cmd', '')
        dangerous_patterns = ['rm ', 'rm -', 'rmdir', 'del ', 'sudo ', 'su ', 'chmod ', 'chown ', 'format ', 'fdisk', 'mkfs']
        write_patterns = ['>', '>>', 'mv ', 'move ', 'cp ', 'copy ', 'touch ', 'mkdir ', 'echo ', 'sed -i', 'perl -i']
        for pattern in write_patterns:
            if pattern in command:
                target_file = self._extract_file_from_command(command, pattern)
                if target_file:
                    path = Path(target_file).resolve()
                    permission = self.get_permission(path)
                    if permission and permission == Permission.READ:
                        return (False, f'Command would modify read-only context path: {path}')
        for pattern in dangerous_patterns:
            if pattern in command.lower():
                return (False, f"Dangerous command pattern '{pattern}' is not allowed")
        if '$' in command:
            safe_vars = ['$?', '$#', '$$']
            has_unsafe_var = False
            if '$(' in command or '${' in command:
                has_unsafe_var = True
            elif any((c in command for c in ['$HOME', '$USER', '$TMPDIR', '$PWD', '$OLDPWD', '$PATH'])):
                has_unsafe_var = True
            else:
                import re
                if re.search('\\$[A-Za-z_][A-Za-z0-9_]*', command):
                    for safe in safe_vars:
                        command = command.replace(safe, '')
                    if re.search('\\$[A-Za-z_][A-Za-z0-9_]*', command):
                        has_unsafe_var = True
            if has_unsafe_var:
                return (False, 'Environment variables in Bash commands are not allowed (security risk: can reference paths outside workspace)')
        if '`' in command:
            return (False, 'Backtick command substitution is not allowed (security risk)')
        if '<(' in command or '>(' in command:
            return (False, 'Process substitution is not allowed (security risk)')
        paths = self._extract_paths_from_command(command)
        for path_str in paths:
            try:
                resolved_path_str = self._resolve_path_against_workspace(path_str)
                path = Path(resolved_path_str).resolve()
                if not self._is_path_within_allowed_directories(path):
                    logger.warning(f'[PathPermissionManager] BLOCKED Bash command accessing path outside allowed directories: {path} (from: {path_str})')
                    return (False, f"Access denied: Bash command references '{path_str}' which resolves to '{path}' outside allowed directories")
            except Exception as e:
                logger.debug(f"[PathPermissionManager] Could not validate path '{path_str}' in Bash command: {e}")
                continue
        return (True, None)

    def _extract_file_path(self, tool_args: Dict[str, Any]) -> Optional[str]:
        """Extract file path from tool arguments."""
        path_keys = ['file_path', 'path', 'filename', 'file', 'notebook_path', 'target', 'destination', 'destination_path', 'destination_base_path']
        for key in path_keys:
            if key in tool_args:
                return tool_args[key]
        return None

    def _extract_file_from_command(self, command: str, pattern: str) -> Optional[str]:
        """Try to extract target file from a command string."""
        if pattern in ['>', '>>']:
            parts = command.split(pattern)
            if len(parts) > 1:
                target = parts[1].strip().split()[0] if parts[1].strip() else None
                if target:
                    return target.strip('"\'')
        if pattern in ['mv ', 'cp ', 'move ', 'copy ']:
            parts = command.split()
            try:
                idx = parts.index(pattern.strip())
                if idx + 2 < len(parts):
                    return parts[idx + 2]
            except (ValueError, IndexError):
                pass
        if pattern in ['touch ', 'mkdir ', 'echo ']:
            parts = command.split()
            try:
                idx = parts.index(pattern.strip())
                if idx + 1 < len(parts):
                    return parts[idx + 1].strip('"\'')
            except (ValueError, IndexError):
                pass
        return None

    def _extract_paths_from_command(self, command: str) -> List[str]:
        """
        Extract all potential file/directory paths from a Bash command for validation.

        This is Claude Code specific - extracts paths to validate directory boundaries.
        Looks for both absolute paths (starting with /) and relative paths (including ../).

        Args:
            command: Bash command string

        Returns:
            List of path strings found in the command
        """
        import shlex
        paths = []
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        for token in tokens:
            cleaned = token.strip('"\'').strip()
            if not cleaned:
                continue
            if cleaned.startswith('-'):
                continue
            if cleaned in ['&&', '||', '|', ';', '>']:
                continue
            if cleaned.startswith('/') or cleaned.startswith('~') or cleaned.startswith('../') or (cleaned == '..') or cleaned.startswith('./'):
                if '*' in cleaned or '?' in cleaned or '[' in cleaned:
                    base = cleaned.split('*')[0].split('?')[0].split('[')[0]
                    if base.endswith('/'):
                        base = base[:-1]
                    if base:
                        paths.append(base)
                else:
                    paths.append(cleaned)
        return paths

    def get_accessible_paths(self) -> List[Path]:
        """Get list of all accessible paths."""
        return [path.path for path in self.managed_paths]

    def get_mcp_filesystem_paths(self) -> List[str]:
        """
        Get all managed paths for MCP filesystem server configuration. Workspace path will be first.

        Only returns directories, as MCP filesystem server cannot accept file paths as arguments.
        For file context paths, the parent directory is already added with path_type="file_context_parent".

        Returns:
            List of directory path strings to include in MCP filesystem server args
        """
        workspace_paths = [str(mp.path) for mp in self.managed_paths if mp.path_type == 'workspace']
        other_paths = [str(mp.path) for mp in self.managed_paths if mp.path_type != 'workspace' and (not mp.is_file)]
        out = workspace_paths + other_paths
        return out

    def get_permission_summary(self) -> str:
        """Get a human-readable summary of permissions."""
        if not self.managed_paths:
            return 'No managed paths configured'
        lines = [f'Managed paths ({len(self.managed_paths)} total):']
        for managed_path in self.managed_paths:
            emoji = '📝' if managed_path.permission == Permission.WRITE else '👁️'
            lines.append(f'  {emoji} {managed_path.path} ({managed_path.permission.value}, {managed_path.path_type})')
        return '\n'.join(lines)

    async def validate_context_access(self, input_data: Dict[str, Any], tool_use_id: Optional[str], context: Any) -> Dict[str, Any]:
        """
        Claude Code SDK compatible hook function for PreToolUse.

        Args:
            input_data: Tool input data with 'tool_name' and 'tool_input'
            tool_use_id: Tool use identifier
            context: HookContext from claude_code_sdk

        Returns:
            Hook response dict with permission decision
        """
        logger.info(f'[PathPermissionManager] PreToolUse hook called for tool_use_id={tool_use_id}, input_data={input_data}')
        tool_name = input_data.get('tool_name', '')
        tool_input = input_data.get('tool_input', {})
        allowed, reason = await self.pre_tool_use_hook(tool_name, tool_input)
        if not allowed:
            logger.warning(f'[PathPermissionManager] Blocked {tool_name}: {reason}')
            return {'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'permissionDecision': 'deny', 'permissionDecisionReason': reason or 'Access denied based on context path permissions'}}
        return {}

    def get_claude_code_hooks_config(self) -> Dict[str, Any]:
        """
        Get Claude Agent SDK hooks configuration.

        Returns:
            Hooks configuration dict for ClaudeAgentOptions
        """
        if not self.managed_paths:
            return {}
        try:
            from claude_agent_sdk import HookMatcher
        except ImportError:
            logger.warning('[PathPermissionManager] claude_agent_sdk not available, hooks disabled')
            return {}
        return {'PreToolUse': [HookMatcher(matcher='Read', hooks=[self.validate_context_access]), HookMatcher(matcher='Write', hooks=[self.validate_context_access]), HookMatcher(matcher='Edit', hooks=[self.validate_context_access]), HookMatcher(matcher='MultiEdit', hooks=[self.validate_context_access]), HookMatcher(matcher='NotebookEdit', hooks=[self.validate_context_access]), HookMatcher(matcher='Grep', hooks=[self.validate_context_access]), HookMatcher(matcher='Glob', hooks=[self.validate_context_access]), HookMatcher(matcher='LS', hooks=[self.validate_context_access]), HookMatcher(matcher='Bash', hooks=[self.validate_context_access])]}

def test_workspace_tools_server_path_validation():
    print('\n🏗️  Testing workspace tools server path validation...')
    helper = TestHelper()
    helper.setup()
    try:
        allowed_paths = [helper.workspace_dir.resolve(), helper.context_dir.resolve(), helper.readonly_dir.resolve()]
        test_source_dir = helper.temp_dir / 'source'
        test_source_dir.mkdir()
        (test_source_dir / 'test_file.txt').write_text('test content')
        (test_source_dir / 'subdir' / 'nested_file.txt').parent.mkdir(parents=True)
        (test_source_dir / 'subdir' / 'nested_file.txt').write_text('nested content')
        allowed_paths.append(test_source_dir.resolve())
        print('  Testing valid absolute destination path...')
        try:
            dest_path = helper.workspace_dir / 'output'
            file_pairs = get_copy_file_pairs(allowed_paths, str(test_source_dir), str(dest_path))
            if len(file_pairs) < 2:
                print(f'❌ Failed: Expected at least 2 files, got {len(file_pairs)}')
                return False
            print(f'  ✓ Found {len(file_pairs)} files to copy')
        except Exception as e:
            print(f'❌ Failed: Valid absolute path should work. Error: {e}')
            return False
        print('  Testing destination outside allowed paths...')
        outside_dir = helper.temp_dir / 'outside'
        outside_dir.mkdir()
        try:
            file_pairs = get_copy_file_pairs(allowed_paths, str(test_source_dir), str(outside_dir / 'output'))
            print('❌ Failed: Should have raised ValueError for path outside allowed directories')
            return False
        except ValueError as e:
            if 'Path not in allowed directories' in str(e):
                print('  ✓ Correctly blocked path outside allowed directories')
            else:
                print(f'❌ Failed: Unexpected error: {e}')
                return False
        except Exception as e:
            print(f'❌ Failed: Unexpected exception: {e}')
            return False
        print('  Testing source outside allowed paths...')
        outside_source = helper.temp_dir / 'outside_source'
        outside_source.mkdir()
        (outside_source / 'bad_file.txt').write_text('bad content')
        try:
            file_pairs = get_copy_file_pairs(allowed_paths, str(outside_source), str(helper.workspace_dir / 'output'))
            print('❌ Failed: Should have raised ValueError for source outside allowed directories')
            return False
        except ValueError as e:
            if 'Path not in allowed directories' in str(e):
                print('  ✓ Correctly blocked source outside allowed directories')
            else:
                print(f'❌ Failed: Unexpected error: {e}')
                return False
        print('  Testing empty destination_base_path...')
        try:
            file_pairs = get_copy_file_pairs(allowed_paths, str(test_source_dir), '')
            print('❌ Failed: Should have raised ValueError for empty destination_base_path')
            return False
        except ValueError as e:
            if 'destination_base_path is required' in str(e):
                print('  ✓ Correctly required destination_base_path')
            else:
                print(f'❌ Failed: Unexpected error: {e}')
                return False
        print('  Testing _validate_path_access function...')
        try:
            test_path = (helper.workspace_dir / 'test.txt').resolve()
            resolved_allowed_paths = [p.resolve() for p in allowed_paths]
            _validate_path_access(test_path, resolved_allowed_paths)
            print('  ✓ Valid path accepted')
        except Exception as e:
            print(f'❌ Failed: Valid path should be accepted. Error: {e}')
            return False
        try:
            test_path = (outside_dir / 'test.txt').resolve()
            resolved_allowed_paths = [p.resolve() for p in allowed_paths]
            _validate_path_access(test_path, resolved_allowed_paths)
            print('❌ Failed: Invalid path should be rejected')
            return False
        except ValueError as e:
            if 'Path not in allowed directories' in str(e):
                print('  ✓ Invalid path correctly rejected')
            else:
                print(f'❌ Failed: Unexpected error: {e}')
                return False
        print('  Testing relative path resolution...')
        original_cwd = os.getcwd()
        try:
            os.chdir(str(helper.workspace_dir))
            source, dest = _validate_and_resolve_paths(allowed_paths, str(test_source_dir / 'test_file.txt'), 'subdir/relative_dest.txt')
            expected_dest = helper.workspace_dir / 'subdir' / 'relative_dest.txt'
            if dest != expected_dest.resolve():
                print(f'❌ Failed: Relative path should resolve to {expected_dest.resolve()}, got {dest}')
                return False
            print('  ✓ Relative path correctly resolved to workspace')
        except Exception as e:
            print(f'❌ Failed: Relative path resolution failed: {e}')
            return False
        finally:
            os.chdir(original_cwd)
        print('✅ Workspace copy server path validation works correctly')
        return True
    finally:
        helper.teardown()

def test_permission_path_root_protection():
    print('\n🛡️  Testing permission path root protection...')
    helper = TestHelper()
    helper.setup()
    try:
        from massgen.filesystem_manager._workspace_tools_server import _is_permission_path_root
        print('  Testing workspace root is protected...')
        if not _is_permission_path_root(helper.workspace_dir, [helper.workspace_dir]):
            print('❌ Failed: Workspace root should be protected from deletion')
            return False
        print('  Testing files within workspace are NOT protected...')
        test_file = helper.workspace_dir / 'file.txt'
        test_file.write_text('content')
        if _is_permission_path_root(test_file, [helper.workspace_dir]):
            print('❌ Failed: Files within workspace should not be protected by root check')
            return False
        test_subdir = helper.workspace_dir / 'subdir'
        test_subdir.mkdir()
        if _is_permission_path_root(test_subdir, [helper.workspace_dir]):
            print('❌ Failed: Subdirs within workspace should not be protected by root check')
            return False
        print('  Testing nested directories are NOT protected...')
        nested = helper.workspace_dir / 'a' / 'b' / 'c'
        nested.mkdir(parents=True)
        if _is_permission_path_root(nested, [helper.workspace_dir]):
            print('❌ Failed: Nested directories should not be protected by root check')
            return False
        print('  Testing system files still protected within workspace...')
        from massgen.filesystem_manager._workspace_tools_server import _is_critical_path
        system_dir = helper.workspace_dir / '.massgen'
        system_dir.mkdir()
        if not _is_critical_path(system_dir, [helper.workspace_dir]):
            print('❌ Failed: .massgen should still be protected by critical path check')
            return False
        if _is_critical_path(helper.workspace_dir, [helper.workspace_dir]):
            print('❌ Failed: Workspace root should not be a critical path when within allowed paths')
            return False
        user_dir = helper.workspace_dir / 'user_project'
        user_dir.mkdir()
        if _is_critical_path(user_dir, [helper.workspace_dir]):
            print('❌ Failed: Regular user directory should not be critical within workspace')
            return False
        print('  Testing real-world scenario: workspace under .massgen/workspaces/...')
        massgen_dir = helper.temp_dir / '.massgen'
        massgen_dir.mkdir()
        workspaces_dir = massgen_dir / 'workspaces'
        workspaces_dir.mkdir()
        real_workspace = workspaces_dir / 'workspace1'
        real_workspace.mkdir()
        user_project = real_workspace / 'bob_dylan_website'
        user_project.mkdir()
        (user_project / 'index.html').write_text('<html></html>')
        if _is_critical_path(user_project, [real_workspace]):
            print('❌ Failed: User project should not be critical within workspace even if parent has .massgen')
            print(f'   Path: {user_project}')
            print(f'   Workspace: {real_workspace}')
            return False
        git_dir = real_workspace / '.git'
        git_dir.mkdir()
        if not _is_critical_path(git_dir, [real_workspace]):
            print('❌ Failed: .git should still be critical within workspace')
            return False
        massgen_subdir = real_workspace / '.massgen'
        massgen_subdir.mkdir()
        if not _is_critical_path(massgen_subdir, [real_workspace]):
            print('❌ Failed: .massgen subdir should be critical within workspace')
            return False
        print('  Testing multiple permission paths...')
        allowed_paths = [helper.workspace_dir, helper.context_dir, helper.readonly_dir]
        for path in allowed_paths:
            if not _is_permission_path_root(path, allowed_paths):
                print(f'❌ Failed: {path} should be protected as root')
                return False
        for root_dir in allowed_paths:
            test_file = root_dir / 'test.txt'
            test_file.write_text('test')
            if _is_permission_path_root(test_file, allowed_paths):
                print(f'❌ Failed: File {test_file} should not be protected as root')
                return False
        print('✅ Permission path root protection works correctly')
        return True
    finally:
        helper.teardown()

