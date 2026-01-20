# Cluster 18

def vertex_normal(vertex, face):
    nf, area = face_normal(vertex, face)
    nf = nf * area
    nv = np.zeros_like(vertex)
    for i in range(face.shape[0]):
        nv[face[i]] += nf[i]
    length = np.sqrt(np.sum(nv ** 2, axis=1, keepdims=True)) + 1e-08
    nv = nv / length
    return nv

def face_normal(vertex, face):
    v01 = vertex[face[:, 1]] - vertex[face[:, 0]]
    v02 = vertex[face[:, 2]] - vertex[face[:, 0]]
    vec = np.cross(v01, v02)
    length = np.sqrt(np.sum(vec ** 2, axis=1, keepdims=True)) + 1e-08
    nf = vec / length
    area = length * 0.5
    return (nf, area)

def parse_scene(scene_path, output_dir):
    print(f'Parsing scene {scene_path}')
    split = os.path.basename(os.path.dirname(os.path.dirname(scene_path)))
    scene_id = os.path.basename(os.path.dirname(scene_path))
    vertices, faces = read_plymesh(scene_path)
    coords = vertices[:, :3]
    colors = vertices[:, 3:6]
    data_dict = dict(coord=coords, color=colors, scene_id=scene_id)
    data_dict['normal'] = vertex_normal(coords, faces)
    torch.save(data_dict, os.path.join(output_dir, split, f'{scene_id}.pth'))

def read_plymesh(filepath):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(filepath, 'rb') as f:
        plydata = plyfile.PlyData.read(f)
    if plydata.elements:
        vertices = pd.DataFrame(plydata['vertex'].data).values
        faces = np.stack(plydata['face'].data['vertex_indices'], axis=0)
        return (vertices, faces)

def vertex_normal(vertex, face):
    nf, area = face_normal(vertex, face)
    nf = nf * area
    nv = np.zeros_like(vertex)
    for i in range(face.shape[0]):
        nv[face[i]] += nf[i]
    length = np.sqrt(np.sum(nv ** 2, axis=1, keepdims=True)) + 1e-08
    nv = nv / length
    return nv

def handle_process(scene_path, output_path, labels_pd, train_scenes, val_scenes, parse_normals=True):
    scene_id = os.path.basename(scene_path)
    mesh_path = os.path.join(scene_path, f'{scene_id}{CLOUD_FILE_PFIX}.ply')
    segments_file = os.path.join(scene_path, f'{scene_id}{CLOUD_FILE_PFIX}{SEGMENTS_FILE_PFIX}')
    aggregations_file = os.path.join(scene_path, f'{scene_id}{AGGREGATIONS_FILE_PFIX}')
    info_file = os.path.join(scene_path, f'{scene_id}.txt')
    if scene_id in train_scenes:
        output_file = os.path.join(output_path, 'train', f'{scene_id}.pth')
        split_name = 'train'
    elif scene_id in val_scenes:
        output_file = os.path.join(output_path, 'val', f'{scene_id}.pth')
        split_name = 'val'
    else:
        output_file = os.path.join(output_path, 'test', f'{scene_id}.pth')
        split_name = 'test'
    print(f'Processing: {scene_id} in {split_name}')
    vertices, faces = read_plymesh(mesh_path)
    coords = vertices[:, :3]
    colors = vertices[:, 3:6]
    save_dict = dict(coord=coords, color=colors, scene_id=scene_id)
    if parse_normals:
        save_dict['normal'] = vertex_normal(coords, faces)
    if split_name != 'test':
        with open(segments_file) as f:
            segments = json.load(f)
            seg_indices = np.array(segments['segIndices'])
        with open(aggregations_file) as f:
            aggregation = json.load(f)
            seg_groups = np.array(aggregation['segGroups'])
        semantic_gt20 = np.ones((vertices.shape[0], 1)) * IGNORE_INDEX
        semantic_gt200 = np.ones((vertices.shape[0], 1)) * IGNORE_INDEX
        instance_ids = np.ones((vertices.shape[0], 1)) * IGNORE_INDEX
        for group in seg_groups:
            point_idx, label_id20, label_id200 = point_indices_from_group(seg_indices, group, labels_pd)
            semantic_gt20[point_idx] = label_id20
            semantic_gt200[point_idx] = label_id200
            instance_ids[point_idx] = group['id']
        semantic_gt20 = semantic_gt20.astype(int)
        semantic_gt200 = semantic_gt200.astype(int)
        instance_ids = instance_ids.astype(int)
        save_dict['semantic_gt20'] = semantic_gt20
        save_dict['semantic_gt200'] = semantic_gt200
        save_dict['instance_gt'] = instance_ids
        processed_vertices = np.hstack((semantic_gt200, instance_ids))
        if np.any(np.isnan(processed_vertices)) or not np.all(np.isfinite(processed_vertices)):
            raise ValueError(f'Find NaN in Scene: {scene_id}')
    torch.save(save_dict, output_file)

def point_indices_from_group(seg_indices, group, labels_pd):
    group_segments = np.array(group['segments'])
    label = group['label']
    label_id20 = labels_pd[labels_pd['raw_category'] == label]['nyu40id']
    label_id20 = int(label_id20.iloc[0]) if len(label_id20) > 0 else 0
    label_id200 = labels_pd[labels_pd['raw_category'] == label]['id']
    label_id200 = int(label_id200.iloc[0]) if len(label_id200) > 0 else 0
    if label_id20 in CLASS_IDS20:
        label_id20 = CLASS_IDS20.index(label_id20)
    else:
        label_id20 = IGNORE_INDEX
    if label_id200 in CLASS_IDS200:
        label_id200 = CLASS_IDS200.index(label_id200)
    else:
        label_id200 = IGNORE_INDEX
    point_idx = np.where(np.isin(seg_indices, group_segments))[0]
    return (point_idx, label_id20, label_id200)

