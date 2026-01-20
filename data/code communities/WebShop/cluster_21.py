# Cluster 21

def get_data(split, mem=False, filter_search=True):
    path = MEM_PATH if mem else PATH
    print('Loading data from {}'.format(path))
    with open(path, 'r') as json_file:
        json_list = list(json_file)
    human_goals = json.load(open(HUMAN_GOAL_PATH, 'r'))
    random.seed(233)
    random.shuffle(json_list)
    goal_range = range(len(human_goals))
    if split == 'train':
        goal_range = range(1500, len(human_goals))
    elif split == 'eval':
        goal_range = range(500, 1500)
    elif split == 'test':
        goal_range = range(0, 500)
    bad = cnt = 0
    state_list, action_list, idx_list, size_list = ([], [], [], [])
    image_list = []
    num_trajs = 0
    for json_str in json_list:
        result = json.loads(json_str)
        s = process_goal(result['states'][0])
        assert s in human_goals, s
        goal_idx = human_goals.index(s)
        if goal_idx not in goal_range:
            continue
        num_trajs += 1
        if 'images' not in result:
            result['images'] = [0] * len(result['states'])
        for state, valid_acts, idx, image in zip(result['states'], result['available_actions'], result['action_idxs'], result['images']):
            cnt += 1
            if filter_search and idx == -1:
                continue
            state_list.append(state)
            image_list.append([0.0] * 512 if image == 0 else image)
            if len(valid_acts) > 20:
                bad += 1
                new_idxs = list(range(6)) + random.sample(range(6, len(valid_acts)), 10)
                if idx not in new_idxs:
                    new_idxs += [idx]
                new_idxs = sorted(new_idxs)
                valid_acts = [valid_acts[i] for i in new_idxs]
                idx = new_idxs.index(idx)
            action_list.extend(valid_acts)
            idx_list.append(idx)
            size_list.append(len(valid_acts))
    print('num of {} trajs: {}'.format(split, num_trajs))
    print('total transitions and bad transitions: {} {}'.format(cnt, bad))
    state_list, action_list = (list(map(process, state_list)), list(map(process, action_list)))
    return (state_list, action_list, idx_list, size_list, image_list)

def get_dataset(split, mem=False):
    states, actions, idxs, sizes, images = get_data(split, mem)
    state_encodings = tokenizer(states, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
    action_encodings = tokenizer(actions, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
    dataset = {'state_input_ids': state_encodings['input_ids'], 'state_attention_mask': state_encodings['attention_mask'], 'action_input_ids': action_encodings['input_ids'].split(sizes), 'action_attention_mask': action_encodings['attention_mask'].split(sizes), 'sizes': sizes, 'images': torch.tensor(images), 'labels': idxs}
    return Dataset.from_dict(dataset)

def main():
    args = parse_args()
    accelerator = Accelerator()
    wandb.init(project='bert_il', config=args, name=args.output_dir)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)
    config = BertConfigForWebshop(image=args.image, pretrain_bert=args.pretrain)
    model = BertModelForWebshop(config)
    train_dataset = get_dataset('train', mem=args.mem)
    eval_dataset = get_dataset('eval', mem=args.mem)
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f'Sample {index} of the training set: {train_dataset[index]}.')
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if not any((nd in n for nd in no_decay))], 'weight_decay': args.weight_decay}, {'params': [p for n, p in model.named_parameters() if any((nd in n for nd in no_decay))], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.max_train_steps)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    if hasattr(args.checkpointing_steps, 'isdigit'):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None
    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config['lr_scheduler_type'] = experiment_config['lr_scheduler_type'].value
        accelerator.init_trackers('glue_no_trainer', experiment_config)
    metric = load_metric('accuracy')
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f'  Num Epochs = {args.num_train_epochs}')
    logger.info(f'  Instantaneous batch size per device = {args.per_device_train_batch_size}')
    logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}')
    logger.info(f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}')
    logger.info(f'  Total optimization steps = {args.max_train_steps}')
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != '':
            accelerator.print(f'Resumed from checkpoint: {args.resume_from_checkpoint}')
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]
        training_difference = os.path.splitext(path)[0]
        if 'epoch' in training_difference:
            starting_epoch = int(training_difference.replace('epoch_', '')) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace('step_', ''))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = total_step = 0
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            outputs = model(**batch)
            loss = outputs.loss
            if args.with_tracking:
                total_loss += loss.detach().float()
                total_step += 1
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            metric.add_batch(predictions=torch.stack([logit.argmax(dim=0) for logit in outputs.logits]), references=batch['labels'])
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                if args.with_tracking and args.logging_steps > 0 and (completed_steps % args.logging_steps == 0):
                    train_metric = metric.compute()
                    wandb.log({'train_accuracy': train_metric, 'train_loss': total_loss / total_step, 'train_step': completed_steps})
                    total_loss = total_step = 0
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f'step_{completed_steps}'
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= args.max_train_steps:
                break
        model.eval()
        samples_seen = 0
        total_loss = total_step = 0
        if len(metric) > 0:
            metric.compute()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = torch.stack([logit.argmax(dim=0) for logit in outputs.logits])
            predictions, references = accelerator.gather((predictions, batch['labels']))
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader):
                    predictions = predictions[:len(eval_dataloader.dataset) - samples_seen]
                    references = references[:len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(predictions=predictions, references=references)
            total_loss += outputs.loss.detach().float()
            total_step += 1
        eval_metric = metric.compute()
        logger.info(f'epoch {epoch}: {eval_metric}')
        if args.with_tracking:
            wandb.log({'eval_accuracy': eval_metric, 'eval_loss': total_loss / total_step, 'epoch': epoch, 'epoch_step': completed_steps})
        if args.checkpointing_steps == 'epoch':
            output_dir = f'epoch_{epoch}'
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            os.makedirs(output_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(output_dir, 'model.pth'))
    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
            json.dump({'eval_accuracy': eval_metric['accuracy']}, f)

def get_dataset(name, flip=False, variant=None, size=None):
    fname = name + '-flip' if flip else name
    fpath = os.path.join(os.path.dirname(__file__), fname)
    d = {}
    splits = ['train', 'validation', 'test']
    if name == 'web_search':
        splits = ['train', 'validation', 'test', 'all']
    for split in splits:
        input, output = get_data(split) if name != 'nl2bash' else get_data(split, variant=variant)
        l = len(input) if size is None else int(len(input) * size)
        print('{} size: {}'.format(split, l))
        if flip:
            input, output = (output, input)
        input, output = (input[:l], output[:l])
        d[split] = process_dataset(input, output)
    d = DatasetDict(d)
    return d

