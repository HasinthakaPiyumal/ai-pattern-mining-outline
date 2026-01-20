# Cluster 34

def format_dir(dir_path, add_header, strip_header, format_code):
    for subdir, dir, files in os.walk(dir_path):
        for file in files:
            if file in IGNORE_FILES:
                continue
            file_path = subdir + os.path.sep + file
            if file_path.endswith('.py'):
                format_file(file_path, add_header, strip_header, format_code)

def format_file(file_path, add_header, strip_header, format_code):
    if file_path.endswith('version.py'):
        return
    abs_path = os.path.abspath(file_path)
    with open(abs_path, 'r+') as fd:
        file_data = fd.read()
        if add_header:
            LOG.info('Adding header: ' + file_path)
            new_file_data = header + file_data
            fd.seek(0, 0)
            fd.truncate()
            fd.write(new_file_data)
        elif strip_header:
            LOG.info('Stripping headers : ' + file_path)
            header_match = header_regex.match(file_data)
            if header_match is None:
                return
            header_comment = header_match.group(1)
            new_file_data = file_data.replace(header_comment, '')
            fd.seek(0, 0)
            fd.truncate()
            fd.write(new_file_data)
        elif format_code:
            isort_command = f'{ISORT_BINARY} --profile  black  {file_path}'
            os.system(isort_command)
            black_command = f'{BLACK_BINARY} -q {file_path}'
            os.system(black_command)
            autoflake_command = f"{FLAKE_BINARY} --config='{FLAKE8_CONFIG}' {file_path}"
            ret_val = os.system(autoflake_command)
            if ret_val:
                sys.exit(1)
            pylint_command = f'{PYLINT_BINARY} --spelling-private-dict-file {ignored_words_file} --rcfile={PYLINTRC}  {file_path}'
            with open(file_path, 'r') as file:
                for line_num, line in enumerate(file, start=1):
                    if file_path not in IGNORE_PRINT_FILES and ' print(' in line:
                        LOG.warning(f'print() found in {file_path}, line {line_num}: {line.strip()}')
                        sys.exit(1)
    fd.close()

@background
def check_file(file):
    valid = False
    file_path = str(Path(file).absolute())
    for source_dir in DEFAULT_DIRS:
        source_path = str(Path(source_dir).resolve())
        if file_path.startswith(source_path):
            valid = True
    if valid:
        if not check_header(file):
            format_file(file, False, True, False)
            format_file(file, True, False, False)
        format_file(file, False, False, True)

def check_header(file_path):
    abs_path = os.path.abspath(file_path)
    with open(abs_path, 'r+') as fd:
        file_data = fd.read()
        header_match = header_regex.match(file_data)
        if header_match is None:
            return False
        header_comment = header_match.group(1)
        if header_comment == header:
            return True
        else:
            return False

def check_notebook_format(notebook_file):
    notebook_file_name = os.path.basename(notebook_file)
    if notebook_file_name == 'ignore_tag.ipynb':
        return True
    with open(notebook_file) as f:
        nb = nbformat.read(f, as_version=4)
    if not nb.cells:
        LOG.error(f'ERROR: Notebook {notebook_file} has no cells')
        sys.exit(1)
    for cell in nb.cells:
        if cell.cell_type not in ['code', 'markdown', 'raw']:
            LOG.error(f'ERROR: Notebook {notebook_file} contains an invalid cell type: {cell.cell_type}')
            sys.exit(1)
    for cell in nb.cells:
        if cell.cell_type == 'code' and (not cell.source.strip()):
            LOG.error(f'ERROR: Notebook {notebook_file} contains an empty code cell')
            sys.exit(1)
    contains_colab_link = False
    for cell in nb.cells:
        if cell.cell_type == 'markdown' and 'colab' in cell.source:
            if notebook_file_name in cell.source:
                contains_colab_link = True
                break
    if contains_colab_link is False:
        LOG.error(f'ERROR: Notebook {notebook_file} does not contain correct Colab link -- update the link.')
        sys.exit(1)
    return True
    import enchant
    from enchant.checker import SpellChecker
    chkr = SpellChecker('en_US')
    for cell in nb.cells:
        if cell.cell_type == 'code':
            continue
        chkr.set_text(cell.source)
        for err in chkr:
            if err.word not in ignored_words:
                LOG.warning(f'WARNING: Notebook {notebook_file} contains the misspelled word: {err.word}')

