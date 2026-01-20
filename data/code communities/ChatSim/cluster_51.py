# Cluster 51

def create_random_shape_with_random_motion(video_length, imageHeight=240, imageWidth=432):
    height = random.randint(imageHeight // 3, imageHeight - 1)
    width = random.randint(imageWidth // 3, imageWidth - 1)
    edge_num = random.randint(6, 8)
    ratio = random.randint(6, 8) / 10
    region = get_random_shape(edge_num=edge_num, ratio=ratio, height=height, width=width)
    region_width, region_height = region.size
    x, y = (random.randint(0, imageHeight - region_height), random.randint(0, imageWidth - region_width))
    velocity = get_random_velocity(max_speed=3)
    m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
    m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
    masks = [m.convert('L')]
    if random.uniform(0, 1) > 0.5:
        return masks * video_length
    for _ in range(video_length - 1):
        x, y, velocity = random_move_control_points(x, y, imageHeight, imageWidth, velocity, region.size, maxLineAcceleration=(3, 0.5), maxInitSpeed=3)
        m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
        m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
        masks.append(m.convert('L'))
    return masks

