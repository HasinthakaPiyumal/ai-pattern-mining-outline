# Cluster 6

def read_as_input_condition_img(png_path):
    img = boxx.imread(png_path)
    return crop_and_resize(img, (H, W))

def flatten_results(results):
    return sum([(results[key] + [None] * n_samples)[:n_samples] for key in sorted(result_blocks)], [])

def read_as_example_input(png_path, guided_path=None):
    condition_img = read_as_input_condition_img(png_path)
    background = np.uint8(condition_img.mean(-1)) // 2
    if guided_path is None:
        layers = []
    else:
        guided_rgba_526 = boxx.imread(guided_path)
        import cv2
        guided_rgba_526 = boxx.resize(guided_rgba_526, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        background = boxx.resize(boxx.resize(background, (128, 128)), (1024, 1024), interpolation=cv2.INTER_NEAREST)
        layers = [guided_rgba_526]
    return [condition_img, dict(background=background, layers=layers, composite=None), '']

