# Cluster 17

def main_process():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='/home/gofinge/Documents/datasets/Stanford3dDataset_v1.2_Aligned_Version')
    parser.add_argument('--output_root', type=str, default='/home/gofinge/Documents/datasets/processed/s3dis')
    parser.add_argument('--parse_normals', action='store_true')
    opt = parser.parse_args()
    room_list = []
    for i in range(1, 7):
        if 'Aligned_Version' in opt.dataset_root:
            area_dir = os.path.join(opt.dataset_root, 'Area_{}'.format(i))
            room_name_list = os.listdir(area_dir)
            room_name_list = [room_name for room_name in room_name_list if room_name != '.DS_Store' and '.txt' not in room_name]
            room_list += [os.path.join('Area_{}'.format(i), room_name) for room_name in room_name_list]
        else:
            area_dir = os.path.join(opt.dataset_root, 'Area_{}'.format(i))
            align_dir = os.path.join(area_dir, 'Area_{}_alignmentAngle.txt'.format(i))
            room_name_list = np.loadtxt(align_dir, dtype=str)
            room_list += [[os.path.join('Area_{}'.format(i), room_name[0]), int(room_name[1])] for room_name in room_name_list]
    pool = mp.Pool(processes=mp.cpu_count())
    pool.starmap(parse_room, [(room, opt.dataset_root, opt.save_root, opt.parse_normals) for room in room_list])
    pool.close()
    pool.join()

