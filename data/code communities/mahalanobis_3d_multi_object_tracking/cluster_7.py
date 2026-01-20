# Cluster 7

def track_nuscenes(data_split, covariance_id, match_distance, match_threshold, match_algorithm, save_root, use_angular_velocity):
    """
  submission {
    "meta": {
        "use_camera":   <bool>  -- Whether this submission uses camera data as an input.
        "use_lidar":    <bool>  -- Whether this submission uses lidar data as an input.
        "use_radar":    <bool>  -- Whether this submission uses radar data as an input.
        "use_map":      <bool>  -- Whether this submission uses map data as an input.
        "use_external": <bool>  -- Whether this submission uses external data as an input.
    },
    "results": {
        sample_token <str>: List[sample_result] -- Maps each sample_token to a list of sample_results.
    }
  }
  
  """
    save_dir = os.path.join(save_root, data_split)
    mkdir_if_missing(save_dir)
    if 'train' in data_split:
        detection_file = '/juno/u/hkchiu/dataset/nuscenes_new/megvii_train.json'
        data_root = '/juno/u/hkchiu/dataset/nuscenes/trainval'
        version = 'v1.0-trainval'
        output_path = os.path.join(save_dir, 'results_train_probabilistic_tracking.json')
    elif 'val' in data_split:
        detection_file = '/juno/u/hkchiu/dataset/nuscenes_new/megvii_val.json'
        data_root = '/juno/u/hkchiu/dataset/nuscenes/trainval'
        version = 'v1.0-trainval'
        output_path = os.path.join(save_dir, 'results_val_probabilistic_tracking.json')
    elif 'test' in data_split:
        detection_file = '/juno/u/hkchiu/dataset/nuscenes_new/megvii_test.json'
        data_root = '/juno/u/hkchiu/dataset/nuscenes/test'
        version = 'v1.0-test'
        output_path = os.path.join(save_dir, 'results_test_probabilistic_tracking.json')
    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
    results = {}
    total_time = 0.0
    total_frames = 0
    with open(detection_file) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.See https://www.nuscenes.org/object-detection for more information.'
    all_results = EvalBoxes.deserialize(data['results'], DetectionBox)
    meta = data['meta']
    print('meta: ', meta)
    print('Loaded results from {}. Found detections for {} samples.'.format(detection_file, len(all_results.sample_tokens)))
    processed_scene_tokens = set()
    for sample_token_idx in tqdm(range(len(all_results.sample_tokens))):
        sample_token = all_results.sample_tokens[sample_token_idx]
        scene_token = nusc.get('sample', sample_token)['scene_token']
        if scene_token in processed_scene_tokens:
            continue
        first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
        current_sample_token = first_sample_token
        mot_trackers = {tracking_name: AB3DMOT(covariance_id, tracking_name=tracking_name, use_angular_velocity=use_angular_velocity, tracking_nuscenes=True) for tracking_name in NUSCENES_TRACKING_NAMES}
        while current_sample_token != '':
            results[current_sample_token] = []
            dets = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            info = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            for box in all_results.boxes[current_sample_token]:
                if box.detection_name not in NUSCENES_TRACKING_NAMES:
                    continue
                q = Quaternion(box.rotation)
                angle = q.angle if q.axis[2] > 0 else -q.angle
                detection = np.array([box.size[2], box.size[0], box.size[1], box.translation[0], box.translation[1], box.translation[2], angle])
                information = np.array([box.detection_score])
                dets[box.detection_name].append(detection)
                info[box.detection_name].append(information)
            dets_all = {tracking_name: {'dets': np.array(dets[tracking_name]), 'info': np.array(info[tracking_name])} for tracking_name in NUSCENES_TRACKING_NAMES}
            total_frames += 1
            start_time = time.time()
            for tracking_name in NUSCENES_TRACKING_NAMES:
                if dets_all[tracking_name]['dets'].shape[0] > 0:
                    trackers = mot_trackers[tracking_name].update(dets_all[tracking_name], match_distance, match_threshold, match_algorithm, scene_token)
                    for i in range(trackers.shape[0]):
                        sample_result = format_sample_result(current_sample_token, tracking_name, trackers[i])
                        results[current_sample_token].append(sample_result)
            cycle_time = time.time() - start_time
            total_time += cycle_time
            current_sample_token = nusc.get('sample', current_sample_token)['next']
        processed_scene_tokens.add(scene_token)
    output_data = {'meta': meta, 'results': results}
    with open(output_path, 'w') as outfile:
        json.dump(output_data, outfile)
    print('Total Tracking took: %.3f for %d frames or %.1f FPS' % (total_time, total_frames, total_frames / total_time))

def format_sample_result(sample_token, tracking_name, tracker):
    """
  Input:
    tracker: (9): [h, w, l, x, y, z, rot_y], tracking_id, tracking_score
  Output:
  sample_result {
    "sample_token":   <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
    "translation":    <float> [3]   -- Estimated bounding box location in meters in the global frame: center_x, center_y, center_z.
    "size":           <float> [3]   -- Estimated bounding box size in meters: width, length, height.
    "rotation":       <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
    "velocity":       <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
    "tracking_id":    <str>         -- Unique object id that is used to identify an object track across samples.
    "tracking_name":  <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
                                       Note that the tracking_name cannot change throughout a track.
    "tracking_score": <float>       -- Object prediction score between 0 and 1 for the class identified by tracking_name.
                                       We average over frame level scores to compute the track level score.
                                       The score is used to determine positive and negative tracks via thresholding.
  }
  """
    rotation = Quaternion(axis=[0, 0, 1], angle=tracker[6]).elements
    sample_result = {'sample_token': sample_token, 'translation': [tracker[3], tracker[4], tracker[5]], 'size': [tracker[1], tracker[2], tracker[0]], 'rotation': [rotation[0], rotation[1], rotation[2], rotation[3]], 'velocity': [0, 0], 'tracking_id': str(int(tracker[7])), 'tracking_name': tracking_name, 'tracking_score': tracker[8]}
    return sample_result

