# Cluster 121

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int))
    iou_matrix = iou_batch(detections, trackers)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return (matches, np.array(unmatched_detections), np.array(unmatched_trackers))

def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    return o

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def associate(detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight, emb_cost, w_assoc_emb, aw_off, aw_param):
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int))
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = (velocities[:, 0], velocities[:, 1])
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi
    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0
    iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)
    angle_diff_cost = valid_mask * diff_angle * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            if emb_cost is None:
                emb_cost = 0
            else:
                emb_cost = emb_cost.cpu().numpy()
                emb_cost[iou_matrix <= 0] = 0
                if not aw_off:
                    emb_cost = compute_aw_max_metric(emb_cost, w_assoc_emb, bottom=aw_param)
                else:
                    emb_cost *= w_assoc_emb
            final_cost = -(iou_matrix + angle_diff_cost + emb_cost)
            matched_indices = linear_assignment(final_cost)
    else:
        matched_indices = np.empty(shape=(0, 2))
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return (matches, np.array(unmatched_detections), np.array(unmatched_trackers))

def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = ((dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0)
    CX2, CY2 = ((tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0)
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-06
    dx = dx / norm
    dy = dy / norm
    return (dy, dx)

def compute_aw_max_metric(emb_cost, w_association_emb, bottom=0.5):
    w_emb = np.full_like(emb_cost, w_association_emb)
    for idx in range(emb_cost.shape[0]):
        inds = np.argsort(-emb_cost[idx])
        if len(inds) < 2:
            continue
        if emb_cost[idx, inds[0]] == 0:
            row_weight = 0
        else:
            row_weight = 1 - max(emb_cost[idx, inds[1]] / emb_cost[idx, inds[0]] - bottom, 0) / (1 - bottom)
        w_emb[idx] *= row_weight
    for idj in range(emb_cost.shape[1]):
        inds = np.argsort(-emb_cost[:, idj])
        if len(inds) < 2:
            continue
        if emb_cost[inds[0], idj] == 0:
            col_weight = 0
        else:
            col_weight = 1 - max(emb_cost[inds[1], idj] / emb_cost[inds[0], idj] - bottom, 0) / (1 - bottom)
        w_emb[:, idj] *= col_weight
    return w_emb * emb_cost

def associate_kitti(detections, trackers, det_cates, iou_threshold, velocities, previous_obs, vdc_weight):
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int))
    '\n        Cost from the velocity direction consistency\n    '
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = (velocities[:, 0], velocities[:, 1])
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi
    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    angle_diff_cost = valid_mask * diff_angle * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores
    '\n        Cost from IoU\n    '
    iou_matrix = iou_batch(detections, trackers)
    '\n        With multiple categories, generate the cost for catgory mismatch\n    '
    num_dets = detections.shape[0]
    num_trk = trackers.shape[0]
    cate_matrix = np.zeros((num_dets, num_trk))
    for i in range(num_dets):
        for j in range(num_trk):
            if det_cates[i] != trackers[j, 4]:
                cate_matrix[i][j] = -1000000.0
    cost_matrix = -iou_matrix - angle_diff_cost - cate_matrix
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(cost_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return (matches, np.array(unmatched_detections), np.array(unmatched_trackers))

class OCSort(object):

    def __init__(self, model_weights, device, fp16, det_thresh, max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3, asso_func='iou', inertia=0.2, w_association_emb=0.75, alpha_fixed_emb=0.95, aw_param=0.5, embedding_off=False, cmc_off=False, aw_off=False, new_kf_off=False, **kwargs):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.w_association_emb = w_association_emb
        self.alpha_fixed_emb = alpha_fixed_emb
        self.aw_param = aw_param
        KalmanBoxTracker.count = 0
        self.embedder = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)
        self.cmc = CMCComputer()
        self.embedding_off = embedding_off
        self.cmc_off = cmc_off
        self.aw_off = aw_off
        self.new_kf_off = new_kf_off

    def update(self, dets, img_numpy, tag='blub'):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        xyxys = dets[:, 0:4]
        scores = dets[:, 4]
        clss = dets[:, 5]
        classes = clss.numpy()
        xyxys = xyxys.numpy()
        scores = scores.numpy()
        dets = dets[:, 0:6].numpy()
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]
        self.height, self.width = img_numpy.shape[:2]
        if self.embedding_off or dets.shape[0] == 0:
            dets_embs = np.ones((dets.shape[0], 1))
        else:
            dets_embs = self._get_features(dets[:, :4], img_numpy)
        if not self.cmc_off:
            transform = self.cmc.compute_affine(img_numpy, dets[:, :4], tag)
            for trk in self.trackers:
                trk.apply_affine_correction(transform)
        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = self.alpha_fixed_emb
        dets_alpha = af + (1 - af) * (1 - trust)
        trks = np.zeros((len(self.trackers), 5))
        trk_embs = []
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trk_embs.append(self.trackers[t].get_emb())
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        if len(trk_embs) > 0:
            trk_embs = np.vstack(trk_embs)
        else:
            trk_embs = np.array(trk_embs)
        for t in reversed(to_del):
            self.trackers.pop(t)
        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])
        '\n            First round of association\n        '
        if self.embedding_off or dets.shape[0] == 0 or trk_embs.shape[0] == 0:
            stage1_emb_cost = None
        else:
            stage1_emb_cost = dets_embs @ trk_embs.T
        matched, unmatched_dets, unmatched_trks = associate(dets, trks, self.iou_threshold, velocities, k_observations, self.inertia, stage1_emb_cost, self.w_association_emb, self.aw_off, self.aw_param)
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :5], dets[m[0], 5])
            self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])
        '\n            Second round of associaton by OCR\n        '
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_dets_embs = dets_embs[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_trks_embs = trk_embs[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            emb_cost_left = left_dets_embs @ left_trks_embs.T
            if self.embedding_off:
                emb_cost_left = np.zeros_like(emb_cost_left)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                '\n                NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may\n                get a higher performance especially on MOT17/MOT20 datasets. But we keep it\n                uniform here for simplicity\n                '
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = (unmatched_dets[m[0]], unmatched_trks[m[1]])
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :5], dets[det_ind, 5])
                    self.trackers[trk_ind].update_emb(dets_embs[det_ind], alpha=dets_alpha[det_ind])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))
        for m in unmatched_trks:
            self.trackers[m].update(None, None)
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :5], dets[i, 5], delta_t=self.delta_t, emb=dets_embs[i], alpha=dets_alpha[i], new_kf=not self.new_kf_off)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                "\n                this is optional to use the recent observation or the kalman filter prediction,\n                we didn't notice significant difference here\n                "
                d = trk.last_observation[:4]
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1], [trk.cls], [trk.conf])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return (x1, y1, x2, y2)

    def _get_features(self, bbox_xyxy, ori_img):
        im_crops = []
        for box in bbox_xyxy:
            x1, y1, x2, y2 = box.astype(int)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.embedder(im_crops).cpu()
        else:
            features = np.array([])
        return features

    def update_public(self, dets, cates, scores):
        self.frame_count += 1
        det_scores = np.ones((dets.shape[0], 1))
        dets = np.concatenate((dets, det_scores), axis=1)
        remain_inds = scores > self.det_thresh
        cates = cates[remain_inds]
        dets = dets[remain_inds]
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            cat = self.trackers[t].cate
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cat]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])
        matched, unmatched_dets, unmatched_trks = associate_kitti(dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia)
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            '\n            The re-association stage by OCR.\n            NOTE: at this stage, adding other strategy might be able to continue improve\n            the performance, such as BYTE association by ByteTrack.\n            '
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()
            iou_left = self.asso_func(left_dets_c, left_trks_c)
            iou_left = np.array(iou_left)
            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:, 4]
            num_dets = unmatched_dets.shape[0]
            num_trks = unmatched_trks.shape[0]
            cate_matrix = np.zeros((num_dets, num_trks))
            for i in range(num_dets):
                for j in range(num_trks):
                    if det_cates_left[i] != trk_cates_left[j]:
                        '\n                        For some datasets, such as KITTI, there are different categories,\n                        we have to avoid associate them together.\n                        '
                        cate_matrix[i][j] = -1000000.0
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold - 0.1:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = (unmatched_dets[m[0]], unmatched_trks[m[1]])
                    if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            trk.cate = cates[i]
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0]
            if trk.time_since_update < 1:
                if self.frame_count <= self.min_hits or trk.hit_streak >= self.min_hits:
                    ret.append(np.concatenate((d, [trk.id + 1], [trk.cls], [trk.conf])).reshape(1, -1))
                if trk.hit_streak == self.min_hits:
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i + 2)]
                        ret.append(np.concatenate((prev_observation[:4], [trk.id + 1], [trk.cls], [trk.conf])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))

    def dump_cache(self):
        self.cmc.dump_cache()
        self.embedder.dump_cache()

def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int))
    iou_matrix = iou_batch(detections, trackers)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return (matches, np.array(unmatched_detections), np.array(unmatched_trackers))

def associate(detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight):
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int))
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = (velocities[:, 0], velocities[:, 1])
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi
    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0
    iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)
    angle_diff_cost = valid_mask * diff_angle * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-(iou_matrix + angle_diff_cost))
    else:
        matched_indices = np.empty(shape=(0, 2))
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return (matches, np.array(unmatched_detections), np.array(unmatched_trackers))

def associate_kitti(detections, trackers, det_cates, iou_threshold, velocities, previous_obs, vdc_weight):
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int))
    '\n        Cost from the velocity direction consistency\n    '
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = (velocities[:, 0], velocities[:, 1])
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi
    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    angle_diff_cost = valid_mask * diff_angle * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores
    '\n        Cost from IoU\n    '
    iou_matrix = iou_batch(detections, trackers)
    '\n        With multiple categories, generate the cost for catgory mismatch\n    '
    num_dets = detections.shape[0]
    num_trk = trackers.shape[0]
    cate_matrix = np.zeros((num_dets, num_trk))
    for i in range(num_dets):
        for j in range(num_trk):
            if det_cates[i] != trackers[j, 4]:
                cate_matrix[i][j] = -1000000.0
    cost_matrix = -iou_matrix - angle_diff_cost - cate_matrix
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(cost_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return (matches, np.array(unmatched_detections), np.array(unmatched_trackers))

class OCSort(object):

    def __init__(self, det_thresh, max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3, asso_func='iou', inertia=0.2, use_byte=False):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte
        KalmanBoxTracker.count = 0

    def update(self, dets, _):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        xyxys = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]
        classes = clss.numpy()
        xyxys = xyxys.numpy()
        confs = confs.numpy()
        output_results = np.column_stack((xyxys, confs, classes))
        inds_low = confs > 0.1
        inds_high = confs < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = output_results[inds_second]
        remain_inds = confs > self.det_thresh
        dets = output_results[remain_inds]
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])
        '\n            First round of association\n        '
        matched, unmatched_dets, unmatched_trks = associate(dets, trks, self.iou_threshold, velocities, k_observations, self.inertia)
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :5], dets[m[0], 5])
        '\n            Second round of associaton by OCR\n        '
        if self.use_byte and len(dets_second) > 0 and (unmatched_trks.shape[0] > 0):
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(dets_second, u_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                '\n                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may\n                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it\n                    uniform here for simplicity\n                '
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = (m[0], unmatched_trks[m[1]])
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets_second[det_ind, :5], dets_second[det_ind, 5])
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                '\n                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may\n                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it\n                    uniform here for simplicity\n                '
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = (unmatched_dets[m[0]], unmatched_trks[m[1]])
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :5], dets[det_ind, 5])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))
        for m in unmatched_trks:
            self.trackers[m].update(None, None)
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :5], dets[i, 5], delta_t=self.delta_t)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                "\n                    this is optional to use the recent observation or the kalman filter prediction,\n                    we didn't notice significant difference here\n                "
                d = trk.last_observation[:4]
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1], [trk.cls], [trk.conf])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

