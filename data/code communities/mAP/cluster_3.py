# Cluster 3

def backup(src_folder, backup_files, backup_folder):
    if not backup_files:
        print('No backup required for', src_folder)
        return
    os.chdir(src_folder)
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    for file in backup_files:
        os.rename(file, backup_folder + '/' + file)

