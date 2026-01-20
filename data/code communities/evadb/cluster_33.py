# Cluster 33

def validate_file(file):
    file = os.path.abspath(file)
    if not os.path.isfile(file):
        LOG.info('ERROR: ' + file + " isn't a file")
        sys.exit(EXIT_FAILURE)
    if not file.endswith('.py'):
        return True
    code_validation = True
    line_number = 1
    commented_code = False
    with open(file, 'r') as opened_file:
        for line in opened_file:
            if line.lstrip().startswith('#'):
                commented_code = contains_commented_out_code(line)
                if commented_code:
                    LOG.info('Commented code ' + 'in file ' + file + ' Line {}: {}'.format(line_number, line.strip()))
            for validator_pattern in VALIDATOR_PATTERNS:
                if validator_pattern.search(line):
                    code_validation = False
                    LOG.info('Unacceptable pattern:' + validator_pattern.pattern.strip() + ' in file ' + file + ' Line {}: {}'.format(line_number, line.strip()))
            line_number += 1
    return code_validation

def contains_commented_out_code(line):
    line = line.lstrip()
    if 'utf-8' in line:
        return False
    if not line.startswith('#'):
        return False
    line = line.lstrip(' \t\x0b\n#').strip()
    regex_list = ['def .+\\)[\\s]*[->]*[\\s]*[a-zA-Z_]*[a-zA-Z0-9_]*:$', 'with .+ as [a-zA-Z_][a-zA-Z0-9_]*:$', 'for [a-zA-Z_][a-zA-Z0-9_]* in .+:$', 'continue$', 'break$']
    for regex in regex_list:
        if re.search(regex, line):
            return True
    symbol_list = list('[]{}=%') + ['print', 'break', 'import ', 'elif ']
    for symbol in symbol_list:
        if symbol in line:
            return True
    if 'return' in line:
        if len(line.split(' ')) >= 2:
            return False
        else:
            return True
    return False

def validate_directory(directory_list):
    code_validation = True
    for dir in directory_list:
        for dir_path, _, files in os.walk(dir):
            for each_file in files:
                file_path = dir_path + os.path.sep + each_file
                if not validate_file(file_path):
                    code_validation = False
    return code_validation

