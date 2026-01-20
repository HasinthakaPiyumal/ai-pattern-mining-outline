# Cluster 5

@click.command()
@click.option('--image_data_input', '-i', default='/home/jiahuih/workspace/yiflu-workspace/video_for_xcube++/000', help='The directory of the waymo data')
@click.option('--semantic_folder_name', default='semantic_masks', help='The name of folder to save the semantic masks (same directory as images)')
@click.option('--sky_mask_folder_name', default='sky_masks', help='The name of folder to save the sky masks (same directory as images)')
@click.option('--overwrite', '-o', is_flag=True, help='Whether to overwrite the existing masks')
def main(image_data_input, semantic_folder_name, sky_mask_folder_name, overwrite):
    semantic_seg(image_data_input, semantic_folder_name, sky_mask_folder_name)

def semantic_seg(image_data_input, semantic_folder_name, sky_folder_name):
    sky_masks_dir = image_data_input.rstrip('/') + '_' + sky_folder_name
    segformer_path = Path(__file__).parent / 'SegFormer'
    segformer_path = segformer_path.as_posix()
    config = os.path.join(segformer_path, 'local_configs', 'segformer', 'B5', 'segformer.b5.1024x1024.city.160k.py')
    checkpoint = os.path.join(segformer_path, 'segformer.b5.1024x1024.city.160k.pth')
    model = init_segmentor(config, checkpoint, device='cuda')
    for filename in os.listdir(image_data_input):
        image_path = os.path.join(image_data_input, filename)
        result = inference_segmentor(model, image_path)
        semantic_mask = result[0].astype(np.uint8)
        sky_mask = (semantic_mask == 10).astype(np.uint8)
        sky_mask = (1 - sky_mask) * 255
        sky_mask_path = os.path.join(sky_masks_dir, filename) + '.png'
        os.makedirs(os.path.dirname(sky_mask_path), exist_ok=True)
        cv2.imwrite(sky_mask_path, sky_mask)

