# Cluster 8

def convert_3dbox_to_8corner(bbox3d_input, nuscenes_to_kitti=False):
    """ Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
        Note: the output of this function will be passed to the funciton iou3d
            for calculating the 3D-IOU. But the function iou3d was written for 
            kitti, so the caller needs to set nuscenes_to_kitti to True if 
            the input bbox3d_input is in nuscenes format.
    """
    bbox3d = copy.copy(bbox3d_input)
    if nuscenes_to_kitti:
        bbox3d_nuscenes = copy.copy(bbox3d)
        bbox3d[0] = bbox3d_nuscenes[1]
        bbox3d[1] = -bbox3d_nuscenes[2]
        bbox3d[2] = -bbox3d_nuscenes[0]
        bbox3d[3] = -bbox3d_nuscenes[3]
        bbox3d[4] = bbox3d_nuscenes[5]
        bbox3d[5] = bbox3d_nuscenes[4]
    R = roty(bbox3d[3])
    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + bbox3d[0]
    corners_3d[1, :] = corners_3d[1, :] + bbox3d[1]
    corners_3d[2, :] = corners_3d[2, :] + bbox3d[2]
    return np.transpose(corners_3d)

@jit
def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

class AB3DMOT(object):

    def __init__(self, covariance_id=0, max_age=2, min_hits=3, tracking_name='car', use_angular_velocity=False, tracking_nuscenes=False):
        """              
    observation: 
      before reorder: [h, w, l, x, y, z, rot_y]
      after reorder:  [x, y, z, rot_y, l, w, h]
    state:
      [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot]
    """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.reorder = [3, 4, 5, 6, 2, 1, 0]
        self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
        self.covariance_id = covariance_id
        self.tracking_name = tracking_name
        self.use_angular_velocity = use_angular_velocity
        self.tracking_nuscenes = tracking_nuscenes

    def update(self, dets_all, match_distance, match_threshold, match_algorithm, seq_name):
        """
    Params:
      dets_all: dict
        dets - a numpy array of detections in the format [[x,y,z,theta,l,w,h],[x,y,z,theta,l,w,h],...]
        info: a array of other info for each det
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
        dets, info = (dets_all['dets'], dets_all['info'])
        dets = dets[:, self.reorder]
        self.frame_count += 1
        print_debug = False
        if False and seq_name == '2f56eb47c64f43df8902d9f88aa8a019' and (self.frame_count >= 25) and (self.frame_count <= 30):
            print_debug = True
            print('self.frame_count: ', self.frame_count)
        if print_debug:
            for trk_tmp in self.trackers:
                print('trk_tmp.id: ', trk_tmp.id)
        trks = np.zeros((len(self.trackers), 7))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict().reshape((-1, 1))
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        if print_debug:
            for trk_tmp in self.trackers:
                print('trk_tmp.id: ', trk_tmp.id)
        dets_8corner = [convert_3dbox_to_8corner(det_tmp, match_distance == 'iou' and self.tracking_nuscenes) for det_tmp in dets]
        if len(dets_8corner) > 0:
            dets_8corner = np.stack(dets_8corner, axis=0)
        else:
            dets_8corner = []
        trks_8corner = [convert_3dbox_to_8corner(trk_tmp, match_distance == 'iou' and self.tracking_nuscenes) for trk_tmp in trks]
        trks_S = [np.matmul(np.matmul(tracker.kf.H, tracker.kf.P), tracker.kf.H.T) + tracker.kf.R for tracker in self.trackers]
        if len(trks_8corner) > 0:
            trks_8corner = np.stack(trks_8corner, axis=0)
            trks_S = np.stack(trks_S, axis=0)
        if match_distance == 'iou':
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner, iou_threshold=match_threshold, print_debug=print_debug, match_algorithm=match_algorithm)
        else:
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner, use_mahalanobis=True, dets=dets, trks=trks, trks_S=trks_S, mahalanobis_threshold=match_threshold, print_debug=print_debug, match_algorithm=match_algorithm)
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0], info[d, :][0])
                detection_score = info[d, :][0][-1]
                trk.track_score = detection_score
        for i in unmatched_dets:
            detection_score = info[i][-1]
            track_score = detection_score
            trk = KalmanBoxTracker(dets[i, :], info[i, :], self.covariance_id, track_score, self.tracking_name, use_angular_velocity)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            d = d[self.reorder_back]
            if trk.time_since_update < self.max_age and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1], trk.info[:-1], [trk.track_score])).reshape(1, -1))
            i -= 1
            if trk.time_since_update >= self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 15 + 7))

