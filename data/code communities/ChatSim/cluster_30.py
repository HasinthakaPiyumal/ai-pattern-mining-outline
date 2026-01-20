# Cluster 30

def receive():
    message = read()
    width = message['resolution_x']
    height = message['resolution_y']
    if width != 0 and height != 0:
        try:
            do_training = bool(message['train'])
            fovy = message['fov_y']
            fovx = message['fov_x']
            znear = message['z_near']
            zfar = message['z_far']
            do_shs_python = bool(message['shs_python'])
            do_rot_scale_python = bool(message['rot_scale_python'])
            keep_alive = bool(message['keep_alive'])
            scaling_modifier = message['scaling_modifier']
            world_view_transform = torch.reshape(torch.tensor(message['view_matrix']), (4, 4)).cuda()
            world_view_transform[:, 1] = -world_view_transform[:, 1]
            world_view_transform[:, 2] = -world_view_transform[:, 2]
            full_proj_transform = torch.reshape(torch.tensor(message['view_projection_matrix']), (4, 4)).cuda()
            full_proj_transform[:, 1] = -full_proj_transform[:, 1]
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        except Exception as e:
            print('')
            traceback.print_exc()
            raise e
        return (custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier)
    else:
        return (None, None, None, None, None, None)

def read():
    global conn
    messageLength = conn.recv(4)
    messageLength = int.from_bytes(messageLength, 'little')
    message = conn.recv(messageLength)
    return json.loads(message.decode('utf-8'))

