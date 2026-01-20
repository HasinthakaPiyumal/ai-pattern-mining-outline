# Cluster 71

def copy_files(src_dir, dest_dir):
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_dir, file)
            shutil.copy(src_file, dest_file)
            print(f"Copied '{src_file}' to '{dest_file}'")

