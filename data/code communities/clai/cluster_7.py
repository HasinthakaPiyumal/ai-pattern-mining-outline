# Cluster 7

def recursively_tag_untagged_files_as_ebcdic(path):
    for file in os.listdir(path):
        filepath = f'{path}{os.path.sep}{file}'
        if os.path.isdir(filepath) and (not os.path.islink(filepath)):
            recursively_tag_untagged_files_as_ebcdic(filepath)
        else:
            try:
                result: CompletedProcess = subprocess.run(['ls', '-T', filepath], encoding='UTF8', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                result = result.stdout.strip()
                match = re.match('(?P<type>\\S+)\\s+(?P<encoding>\\S+)\\s+T\\=(?P<tagging>on|off)\\s+(?P<filepath>.*)', result)
                if match:
                    tagging: str = match.group('tagging')
                    if tagging != 'on':
                        os.system(f'chtag -tc 1047 {filepath}')
            except:
                pass

