# Cluster 1

def get_trainloader(args):
    if not args.one_dataset_every_time:
        dataset = SFTDataset(datadir=filenames)
        train_dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, pin_memory=False, drop_last=False, shuffle=False, num_workers=0 if os.name == 'nt' else 2, sampler=DistributedSampler(dataset) if args.ddp_config is not None else None, collate_fn=collate_train_fn)
    else:
        if len(args.filenames) == 0:
            args.filenames = deque(filenames)
        filename = args.filenames.popleft()
        dataset = SFTDataset([filename], verbose=0)
        train_dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=False, drop_last=False, shuffle=False, num_workers=0 if os.name == 'nt' else 2, sampler=DistributedSampler(dataset) if args.ddp_config is not None else None, collate_fn=collate_train_fn)
    return train_dataloader

class GenTrainLoader(Callback):
    """当前dataloader消耗完，自动用下一个文件生成dataloder
    """

    def on_dataloader_end(self, logs=None):
        model.train_dataloader = get_trainloader(args)

