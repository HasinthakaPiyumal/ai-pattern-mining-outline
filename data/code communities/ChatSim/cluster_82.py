# Cluster 82

def set_camera_params(intrinsic, cam2world, camera_obj_name='Camera'):
    """
    Args:
        int: Dict
            {H: xx, W:xx, focal:xx}
        cam2world: np.ndarray, OpenCV coordinate system for camera.
            shape [4,4]
        camera_obj_name: str
            name of Camera
    """
    if camera_obj_name not in bpy.data.objects:
        cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
        for camera in cameras:
            bpy.data.objects.remove(camera, do_unlink=True)
        bpy.ops.object.camera_add(enter_editmode=False)
        if hasattr(bpy.context, 'object'):
            camera = bpy.context.object
            camera.name = camera_obj_name
        else:
            camera = bpy.data.objects['Camera']
        bpy.context.scene.camera = camera
    rot = get_rotation_quaternion(cam2world)
    loc = get_location(cam2world)
    focal_in_mm = get_focal_in_mm(intrinsic['H'], intrinsic['focal'])
    camera = bpy.data.objects[camera_obj_name]
    camera.location = loc
    camera.rotation_mode = 'QUATERNION'
    camera.rotation_quaternion = rot
    camera.data.sensor_fit = 'VERTICAL'
    camera.data.sensor_height = default_sensor_height
    camera.data.lens = focal_in_mm
    camera.data.lens_unit = 'MILLIMETERS'

def get_rotation_quaternion(ext):
    flip_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    ext = ext @ flip_matrix
    q = pyquaternion.Quaternion(matrix=ext)
    return [q.w, q.x, q.y, q.z]

def get_location(ext):
    return ext[:3, 3]

def get_focal_in_mm(H_pixel, focal_pixel):
    focal_mm = default_sensor_height / H_pixel * focal_pixel
    return focal_mm

