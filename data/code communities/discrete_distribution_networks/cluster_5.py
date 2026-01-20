# Cluster 5

def generate(example_description, input_condition_img, editor_value, prompt_value=None):
    print(f'{boxx.localTimeStr(True)} {boxx.increase('requests')}th requests')
    boxx.tree([example_description, input_condition_img, editor_value, prompt_value])
    ddn = get_model()
    n = n_samples
    if prompt_value:
        n = n_samples_with_clip
    guided_rgba = editor_value['layers'][0] if len(editor_value['layers']) else None
    d = ddn.coloring_demo_inference(input_condition_img, n_samples=n, guided_rgba=guided_rgba, clip_prompt=prompt_value)
    stage_last_predicts = d['stage_last_predicts_np']
    if 0:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        if guided_rgba is not None:
            boxx.imsave(f'/tmp/condition256_gen_{timestamp}_guided_rgba.png', guided_rgba)
            boxx.imsave(f'/tmp/condition256_gen_{timestamp}_background.png', rgba_edit['background'])
        if input_condition_img is not None:
            boxx.imsave(f'/tmp/condition256_gen_{timestamp}_input_condition_img.png', input_condition_img)
        if prompt_value:
            with open(f'/tmp/condition256_gen_{timestamp}_prompt.txt', 'w') as f:
                f.write(prompt_value)
        largest_resolution_key = max(stage_last_predicts.keys(), key=lambda x: int(x.split('x')[0]))
        stage_last_predict = stage_last_predicts[largest_resolution_key]
        for i, img in enumerate(stage_last_predict):
            if img is not None:
                boxx.imsave(f'/tmp/condition256_gen_{timestamp}_stage_last_predict_{i}.png', img)
    return flatten_results(stage_last_predicts)

def get_model():
    with threading.Lock():
        global ddn
        if ddn is None:
            ddn = DDNInference(ddn_asset_paths['v32-00003-ffhq-256x256-ffhq256_cond.color_chain.dropout0.05_batch64_k64-shot-200000.pkl'])
        return ddn

