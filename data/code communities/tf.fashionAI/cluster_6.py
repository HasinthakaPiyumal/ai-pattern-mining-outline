# Cluster 6

def main(_):
    import subprocess
    import copy
    all_category = parse_str_comma_list(FLAGS.model_to_train)
    for cat in all_category:
        tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, cat))
    for cat in all_category:
        temp_params = copy.deepcopy(total_params)
        for k, v in total_params.items():
            if k[2:] in detail_params[cat]:
                temp_params[k] = detail_params[cat][k[2:]]
        params_str = []
        for k, v in temp_params.items():
            if v is not None:
                params_str.append(k)
                params_str.append(str(v))
        print('params send: ', params_str)
        train_process = subprocess.Popen(['python', './train_subnet.py'] + params_str, stdout=subprocess.PIPE, cwd=os.getcwd())
        output, _ = train_process.communicate()
        print(output)

def parse_str_comma_list(args):
    return [s.strip() for s in args.split(',')]

