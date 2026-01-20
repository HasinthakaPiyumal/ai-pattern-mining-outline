# Cluster 1

class BoTSORT(object):

    def __init__(self, model_weights, device, fp16, track_high_thresh: float=0.45, new_track_thresh: float=0.6, track_buffer: int=30, match_thresh: float=0.8, proximity_thresh: float=0.5, appearance_thresh: float=0.25, cmc_method: str='sparseOptFlow', frame_rate=30, lambda_=0.985):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        BaseTrack.clear_count()
        self.frame_id = 0
        self.lambda_ = lambda_
        self.track_high_thresh = track_high_thresh
        self.new_track_thresh = new_track_thresh
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.match_thresh = match_thresh
        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)
        self.gmc = GMC(method=cmc_method, verbose=[None, False])

    def update(self, output_results, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        xyxys = output_results[:, 0:4]
        xywh = xyxy2xywh(xyxys.numpy())
        confs = output_results[:, 4]
        clss = output_results[:, 5]
        classes = clss.numpy()
        xyxys = xyxys.numpy()
        confs = confs.numpy()
        remain_inds = confs > self.track_high_thresh
        inds_low = confs > 0.1
        inds_high = confs < self.track_high_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = xywh[inds_second]
        dets = xywh[remain_inds]
        scores_keep = confs[remain_inds]
        scores_second = confs[inds_second]
        classes_keep = classes[remain_inds]
        clss_second = classes[inds_second]
        self.height, self.width = img.shape[:2]
        'Extract embeddings '
        features_keep = self._get_features(dets, img)
        if len(dets) > 0:
            'Detections'
            detections = [STrack(xyxy, s, c, f.cpu().numpy()) for xyxy, s, c, f in zip(dets, scores_keep, classes_keep, features_keep)]
        else:
            detections = []
        ' Add newly detected tracklets to tracked_stracks'
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        ' Step 2: First association, with high score detection boxes'
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)
        raw_emb_dists = matching.embedding_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections, only_position=False, lambda_=self.lambda_)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        ' Step 3: Second association, with low score detection boxes'
        if len(dets_second) > 0:
            'Detections'
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for tlbr, s, c in zip(dets_second, scores_second, clss_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        'Deal with unconfirmed tracks, usually tracks with only one beginning frame'
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        ious_dists = matching.fuse_score(ious_dists, detections)
        emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
        raw_emb_dists = emb_dists.copy()
        emb_dists[emb_dists > self.appearance_thresh] = 1.0
        emb_dists[ious_dists_mask] = 1.0
        dists = np.minimum(ious_dists, emb_dists)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        ' Step 4: Init new stracks'
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        ' Step 5: Update state'
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        ' Merge '
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        outputs = []
        for t in output_stracks:
            output = []
            tlwh = t.tlwh
            tid = t.track_id
            tlwh = np.expand_dims(tlwh, axis=0)
            xyxy = xywh2xyxy(tlwh)
            xyxy = np.squeeze(xyxy, axis=0)
            output.extend(xyxy)
            output.append(tid)
            output.append(t.cls)
            output.append(t.score)
            outputs.append(output)
        return outputs

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return (x1, y1, x2, y2)

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = (list(), list())
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return (resa, resb)

