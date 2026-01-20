# Cluster 70

def find_best_checkpoint(model_list, models_dir):
    with open(model_list) as f:
        models = [m.strip() for m in f.readlines()]
    with open(f'{model_list}_best', 'w') as f:
        for model in models:
            print(model)
            best_f1 = 0
            best_epoch = 0
            best_step = 0
            with open(os.path.join(models_dir, model, 'train.log')) as fm:
                lines = fm.readlines()
                for line_index in range(len(lines)):
                    line = lines[line_index]
                    if 'Validation metrics after epoch' in line:
                        sharp_index = line.index('#')
                        cur_ep = line[sharp_index + 1:]
                        comma_index = cur_ep.index(',')
                        cur_ep = int(cur_ep[:comma_index])
                        total_index = line.index('total ')
                        step = int(line[total_index:].split()[1].strip())
                        total_line = lines[line_index + 5]
                        if not total_line.startswith('total'):
                            continue
                        words = total_line.strip().split()
                        f1 = float(words[-1])
                        print(f'\tEpoch: {cur_ep}, f1={f1}')
                        if f1 > best_f1:
                            best_f1 = f1
                            best_epoch = cur_ep
                            best_step = step
            f.write(f'{model}\t{best_epoch}\t{best_step}\t{best_f1}\n')

