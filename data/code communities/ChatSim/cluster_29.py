# Cluster 29

def render(viewpoint_camera, pc, args, bg_color, scaling_modifier=1.0, override_color=None, exposure_scale=None):
    return gsplat_render(viewpoint_camera, pc, args, bg_color, scaling_modifier, override_color, exposure_scale)

def gsplat_render(viewpoint_camera, pc: GaussianModel, args: omegaconf.dictconfig.DictConfig, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None, exposure_scale=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if viewpoint_camera.K is not None:
        focal_length_x, focal_length_y, cx, cy = viewpoint_camera.K
        K = torch.tensor([[focal_length_x, 0, cx], [0, focal_length_y, cy], [0, 0, 1.0]]).to(pc.get_xyz)
    else:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
        focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
        K = torch.tensor([[focal_length_x, 0, viewpoint_camera.image_width / 2.0], [0, focal_length_y, viewpoint_camera.image_height / 2.0], [0, 0, 1]]).to(pc.get_xyz)
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation
    if override_color is not None:
        colors = override_color
        sh_degree = None
    else:
        colors = pc.get_features
        sh_degree = pc.active_sh_degree
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)
    render_colors, render_alphas, info = rasterization(means=means3D, quats=rotations, scales=scales, opacities=opacity.squeeze(-1), colors=colors, viewmats=viewmat[None], Ks=K[None], backgrounds=None, width=int(viewpoint_camera.image_width), height=int(viewpoint_camera.image_height), packed=False, sh_degree=sh_degree, render_mode='RGB+ED')
    rendered_image = render_colors[0].permute(2, 0, 1)[:3]
    rendered_depth = render_colors[0].permute(2, 0, 1)[3:]
    rendered_alphas = render_alphas[0].permute(2, 0, 1)
    if exposure_scale is not None:
        rendered_image *= exposure_scale
        rendered_image = OETF(rendered_image)
    radii = info['radii'].squeeze(0)
    try:
        info['means2d'].retain_grad()
    except:
        pass
    return_pkg = {'render_3dgs': rendered_image, 'viewspace_points': info['means2d'], 'visibility_filter': radii > 0, 'radii': radii}
    if args.render_depth:
        return_pkg['depth'] = rendered_depth
    if args.render_opacity:
        return_pkg['opacity'] = rendered_alphas
    if args.render_sky:
        sky_bg = pc.get_sky_bg(viewpoint_camera)
        return_pkg['sky_bg'] = sky_bg
        if args.blend_sky:
            assert args.render_opacity
            rendered_image = rendered_image + (1 - rendered_alphas) * sky_bg
    return_pkg['render'] = rendered_image
    return return_pkg

def main():

    def parse_argv(argv):
        result = []
        for i in range(1, len(argv)):
            if argv[i] == '--':
                if i + 1 < len(argv):
                    result.append(argv[i + 1])
        return result
    argv = sys.argv
    argv = parse_argv(argv)
    render_yaml = argv[0]
    start_frame = int(argv[1])
    end_frame = int(argv[2])
    for frame in range(start_frame, end_frame):
        with open(os.path.join(render_yaml, f'{frame}.yaml'), 'r') as file:
            render_opt = yaml.safe_load(file)
        bpy.ops.wm.read_homefile(app_template='')
        rm_all_in_blender()
        scene_data = render_opt['scene_file']
        data_dict = np.load(scene_data)
        H = data_dict['H'].tolist()
        W = data_dict['W'].tolist()
        focal = data_dict['focal'].tolist()
        render_opt['intrinsic'] = {'H': H, 'W': W, 'focal': focal}
        render_opt['cam2world'] = data_dict['extrinsic']
        render_opt['background_RGB'] = data_dict['rgb']
        render_opt['background_depth'] = data_dict['depth']
        render(render_opt)

def parse_argv(argv):
    result = []
    for i in range(1, len(argv)):
        if argv[i] == '--':
            if i + 1 < len(argv):
                result.append(argv[i + 1])
    return result

