# Cluster 58

def paste_object(source, source_mask, target, target_coords, resize_scale=1):
    assert target_coords[0] < target.shape[1] and target_coords[1] < target.shape[0]
    x, y, w, h = cv2.boundingRect(source_mask)
    assert h < source.shape[0] and w < source.shape[1]
    obj = source[y:y + h, x:x + w]
    obj_msk = source_mask[y:y + h, x:x + w]
    if resize_scale != 1:
        obj = cv2.resize(obj, (0, 0), fx=resize_scale, fy=resize_scale)
        obj_msk = cv2.resize(obj_msk, (0, 0), fx=resize_scale, fy=resize_scale)
        _, _, w, h = cv2.boundingRect(obj_msk)
    xt = max(0, target_coords[0] - w // 2)
    yt = max(0, target_coords[1] - h // 2)
    if target_coords[0] - w // 2 < 0:
        obj = obj[:, w // 2 - target_coords[0]:]
        obj_msk = obj_msk[:, w // 2 - target_coords[0]:]
    if target_coords[0] + w // 2 > target.shape[1]:
        obj = obj[:, :target.shape[1] - target_coords[0] + w // 2]
        obj_msk = obj_msk[:, :target.shape[1] - target_coords[0] + w // 2]
    if target_coords[1] - h // 2 < 0:
        obj = obj[h // 2 - target_coords[1]:, :]
        obj_msk = obj_msk[h // 2 - target_coords[1]:, :]
    if target_coords[1] + h // 2 > target.shape[0]:
        obj = obj[:target.shape[0] - target_coords[1] + h // 2, :]
        obj_msk = obj_msk[:target.shape[0] - target_coords[1] + h // 2, :]
    _, _, w, h = cv2.boundingRect(obj_msk)
    target[yt:yt + h, xt:xt + w][obj_msk == 255] = obj[obj_msk == 255]
    target_mask = np.zeros_like(target)
    target_mask = cv2.cvtColor(target_mask, cv2.COLOR_BGR2GRAY)
    target_mask[yt:yt + h, xt:xt + w][obj_msk == 255] = 255
    return (target, target_mask)

