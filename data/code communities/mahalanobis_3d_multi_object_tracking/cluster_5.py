# Cluster 5

def diff_orientation_correction(det, trk):
    """
  return the angle diff = det - trk
  if angle diff > 90 or < -90, rotate trk and update the angle diff
  """
    diff = det - trk
    diff = angle_in_range(diff)
    if diff > np.pi / 2:
        diff -= np.pi
    if diff < -np.pi / 2:
        diff += np.pi
    diff = angle_in_range(diff)
    return diff

def angle_in_range(angle):
    """
  Input angle: -2pi ~ 2pi
  Output angle: -pi ~ pi
  """
    if angle > np.pi:
        angle -= 2 * np.pi
    if angle < -np.pi:
        angle += 2 * np.pi
    return angle

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.1, use_mahalanobis=False, dets=None, trks=None, trks_S=None, mahalanobis_threshold=0.1, print_debug=False, match_algorithm='greedy'):
    """
  Assigns detections to tracked object (both represented as bounding boxes)

  detections:  N x 8 x 3
  trackers:    M x 8 x 3

  dets: N x 7
  trks: M x 7
  trks_S: N x 7 x 7

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 8, 3), dtype=int))
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    distance_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    if use_mahalanobis:
        assert dets is not None
        assert trks is not None
        assert trks_S is not None
    if use_mahalanobis and print_debug:
        print('dets.shape: ', dets.shape)
        print('dets: ', dets)
        print('trks.shape: ', trks.shape)
        print('trks: ', trks)
        print('trks_S.shape: ', trks_S.shape)
        print('trks_S: ', trks_S)
        S_inv = [np.linalg.inv(S_tmp) for S_tmp in trks_S]
        S_inv_diag = [S_inv_tmp.diagonal() for S_inv_tmp in S_inv]
        print('S_inv_diag: ', S_inv_diag)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            if use_mahalanobis:
                S_inv = np.linalg.inv(trks_S[t])
                diff = np.expand_dims(dets[d] - trks[t], axis=1)
                corrected_angle_diff = diff_orientation_correction(dets[d][3], trks[t][3])
                diff[3] = corrected_angle_diff
                distance_matrix[d, t] = np.sqrt(np.matmul(np.matmul(diff.T, S_inv), diff)[0][0])
            else:
                iou_matrix[d, t] = iou3d(det, trk)[0]
                distance_matrix = -iou_matrix
    if match_algorithm == 'greedy':
        matched_indices = greedy_match(distance_matrix)
    elif match_algorithm == 'pre_threshold':
        if use_mahalanobis:
            to_max_mask = distance_matrix > mahalanobis_threshold
            distance_matrix[to_max_mask] = mahalanobis_threshold + 1
        else:
            to_max_mask = iou_matrix < iou_threshold
            distance_matrix[to_max_mask] = 0
            iou_matrix[to_max_mask] = 0
        matched_indices = linear_assignment(distance_matrix)
    else:
        matched_indices = linear_assignment(distance_matrix)
    if print_debug:
        print('distance_matrix.shape: ', distance_matrix.shape)
        print('distance_matrix: ', distance_matrix)
        print('matched_indices: ', matched_indices)
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if len(matched_indices) == 0 or t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    matches = []
    for m in matched_indices:
        match = True
        if use_mahalanobis:
            if distance_matrix[m[0], m[1]] > mahalanobis_threshold:
                match = False
        elif iou_matrix[m[0], m[1]] < iou_threshold:
            match = False
        if not match:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    if print_debug:
        print('matches: ', matches)
        print('unmatched_detections: ', unmatched_detections)
        print('unmatched_trackers: ', unmatched_trackers)
    return (matches, np.array(unmatched_detections), np.array(unmatched_trackers))

def greedy_match(distance_matrix):
    """
  Find the one-to-one matching using greedy allgorithm choosing small distance
  distance_matrix: (num_detections, num_tracks)
  """
    matched_indices = []
    num_detections, num_tracks = distance_matrix.shape
    distance_1d = distance_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)
    detection_id_matches_to_tracking_id = [-1] * num_detections
    tracking_id_matches_to_detection_id = [-1] * num_tracks
    for sort_i in range(index_2d.shape[0]):
        detection_id = int(index_2d[sort_i][0])
        tracking_id = int(index_2d[sort_i][1])
        if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[detection_id] == -1:
            tracking_id_matches_to_detection_id[tracking_id] = detection_id
            detection_id_matches_to_tracking_id[detection_id] = tracking_id
            matched_indices.append([detection_id, tracking_id])
    matched_indices = np.array(matched_indices)
    return matched_indices

