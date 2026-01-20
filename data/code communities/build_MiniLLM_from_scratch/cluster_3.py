# Cluster 3

def main():
    """数据处理主函数
    :param FILE_NAMES: 处理的数据集名称
    :param DATASET_SRC_DIR: 数据源文件夹
    :param DATASET_SAVE_DIR: 数据保存的文件夹
    :param tokenizer: tokenizer
    """
    if args.max_samples is not None:
        log_warn(f'Only use {args.max_samples} samples for each sft dataset.')
    else:
        log_warn(f'Use all samples for each sft dataset, may be slow.')
    with Timeit() as ti:
        for filename in args.file_names:
            sample_count = MAPPING[filename](filename, tokenizer)
            ti.lap(name=f'{filename}: len={sample_count}'.ljust(70) + '-', reset=True)

