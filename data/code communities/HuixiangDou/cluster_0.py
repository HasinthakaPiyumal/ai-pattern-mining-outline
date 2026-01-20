# Cluster 0

def read_requirements():
    lines = []
    with open('requirements.txt', 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            if 'textract' in line:
                continue
            if len(line) > 0:
                lines.append(line)
    return lines

def get_version():
    with open(os.path.join(pwd, version_file), 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

def readme():
    with open(os.path.join(pwd, 'README.md'), encoding='utf-8') as f:
        content = f.read()
    return content

