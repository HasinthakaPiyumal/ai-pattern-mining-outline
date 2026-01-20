# Cluster 8

def renameCD(src_dir, mode):
    logfile = mode + '.links.log'
    cd_dir = os.path.join(src_dir, mode)
    count = 0
    for subdir in os.listdir(cd_dir):
        if subdir.startswith('CD') or subdir.startswith('cd'):
            newName = lookup(subdir, os.path.join(src_dir, logfile))
            cd_path = os.path.join(src_dir, mode, subdir)
            new_cd_path = os.path.join(src_dir, mode, newName)
            os.rename(cd_path, new_cd_path)
            count += 1
            print('new file ', count, ': ', new_cd_path)

def lookup(cd_id, logfile):
    with open(logfile, 'r') as f:
        content = f.readlines()
    for line in content:
        if int(line.split(' ')[-1][2:]) == int(cd_id[2:]):
            if '.' in line.split(' ')[-3]:
                newName = line.split(' ')[-3]
                return newName
        else:
            continue

