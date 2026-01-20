# Cluster 69

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    ignored_events = set()
    for orig_fname in glob.glob(args.inglob):
        cur_dirpath = os.path.dirname(orig_fname)
        subdirname = os.path.basename(cur_dirpath)
        exp_root_path = os.path.dirname(cur_dirpath)
        exp_name = os.path.basename(exp_root_path)
        writers_by_group = {}
        for e in tf.compat.v1.train.summary_iterator(orig_fname):
            for v in e.summary.value:
                if need_drop(v.tag):
                    continue
                cur_group, cur_title = get_group_and_title(v.tag)
                if cur_group is None:
                    if v.tag not in ignored_events:
                        print(f'WARNING: Could not detect group for {v.tag}, ignoring it')
                        ignored_events.add(v.tag)
                    continue
                cur_writer = writers_by_group.get(cur_group, None)
                if cur_writer is None:
                    if args.include_version:
                        cur_outdir = os.path.join(args.outdir, exp_name, f'{subdirname}_{cur_group}')
                    else:
                        cur_outdir = os.path.join(args.outdir, exp_name, cur_group)
                    cur_writer = SummaryWriter(cur_outdir)
                    writers_by_group[cur_group] = cur_writer
                cur_writer.add_scalar(cur_title, v.simple_value, global_step=e.step, walltime=e.wall_time)

def need_drop(tag):
    for rule in DROP_RULES:
        if rule.search(tag):
            return True
    return False

def get_group_and_title(tag):
    for rule in GROUPING_RULES:
        match = rule.search(tag)
        if match is None:
            continue
        return (match.group('group'), match.group('title'))
    return (None, None)

