# Cluster 7

def parse_args() -> TrainLoraArguments:
    parser = HfArgumentParser(TrainLoraArguments)
    return parser.parse_args()

def train(args: TrainLoraArguments):
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16 if args.half else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    config = peft.LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=args.lora_target_modules, lora_dropout=args.lora_dropout, bias='none', task_type=peft.TaskType.CAUSAL_LM)
    model = peft.get_peft_model(model, config)
    data_files = glob.glob(os.path.join(args.data_path, '*.jsonl'))
    print('Collected data files...', data_files)
    dataset = load_dataset('json', data_files=data_files)['train']
    data = Dataset.from_generator(ConstantLengthDataset(tokenizer, dataset))
    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(resume_from_checkpoint, 'pytorch_model.bin')
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, 'adapter_model.bin')
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            print(f'Restarting from {checkpoint_name}')
            adapters_weights = torch.load(checkpoint_name)
            model = peft.set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f'Checkpoint {checkpoint_name} not found')
    model.print_trainable_parameters()
    train_val = data.train_test_split(test_size=args.val_set_size, shuffle=True, seed=42)
    train_data = train_val['train'].shuffle()
    val_data = train_val['test'].shuffle()
    trainer = Trainer(model=model, train_dataset=train_data, eval_dataset=val_data, args=TrainingArguments(per_device_train_batch_size=args.micro_batch_size, gradient_accumulation_steps=gradient_accumulation_steps, warmup_steps=100, num_train_epochs=args.num_epochs, learning_rate=args.learning_rate, fp16=args.half, logging_steps=10, evaluation_strategy='steps', save_strategy='steps', eval_steps=args.eval_steps, save_steps=args.eval_steps, output_dir=args.output_dir, save_total_limit=3, load_best_model_at_end=True))
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: peft.get_peft_model_state_dict(self, old_state_dict())).__get__(model, type(model))
    model = torch.compile(model)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(args.output_dir)
    print("\n If there's a warning about missing keys above, please disregard :)")

