# Cluster 3

def update_id_in_listObjframes(listObj, frames, shape, old_id, new_id=None):
    """
    Summary:
        Update the id of a shape in a list of frames in listObj.
        
    Args:
        listObj: a list of objects (each object is a dictionary of a frame with keys (frame_idx, frame_data))
        frames: a list of frames to update
        shape: the shape to update
        old_id: the old id
        new_id: the new id, if None then the old id is used (no id change)
        
    Returns:
        listObj: a list of objects (each object is a dictionary of a frame with keys (frame_idx, frame_data))
    """
    for frame in frames:
        listObj = update_id_in_listObjframe(listObj, frame, shape, old_id, new_id)
    return listObj

def update_id_in_listObjframe(listObj, frame, shape, old_id, new_id=None):
    """
        Summary:
            Update the id of a shape in a frame in listObj.
            
        Args:
            listObj: a list of objects (each object is a dictionary of a frame with keys (frame_idx, frame_data))
            frame: the frame to update
            shape: the shape to update
            old_id: the old id
            new_id: the new id, if None then the old id is used (no id change)
            
        Returns:
            listObj: a list of objects (each object is a dictionary of a frame with keys (frame_idx, frame_data))
        """
    new_id = old_id if new_id is None else new_id
    for object_ in listObj[frame - 1]['frame_data']:
        if object_['tracker_id'] == old_id:
            object_['tracker_id'] = new_id
            object_['class_name'] = shape.label
            object_['confidence'] = str(1.0)
            object_['class_id'] = coco_classes.index(shape.label) if shape.label in coco_classes else -1
            break
    return listObj

