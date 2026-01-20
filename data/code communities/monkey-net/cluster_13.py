# Cluster 13

def mv_all_images(images, in_folder, out_folder):
    for img in images:
        move(os.path.join(in_folder, img), os.path.join(out_folder, img))

