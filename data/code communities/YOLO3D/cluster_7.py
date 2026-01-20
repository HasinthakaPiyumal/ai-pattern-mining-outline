# Cluster 7

def train(train_path=ROOT / 'dataset/KITTI/training', checkpoint_path=ROOT / 'weights/checkpoints', model_select='resnet18', epochs=10, batch_size=32, num_workers=2, gpu=1, val_split=0.1, model_path=ROOT / 'weights/', api_key=''):
    comet_logger = CometLogger(api_key=api_key, project_name='YOLO3D')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=checkpoint_path, filename='model_{epoch:02d}_{val_loss:.2f}', save_top_k=3, mode='min')
    trainer = Trainer(logger=comet_logger, callbacks=[checkpoint_callback], gpus=gpu, min_epochs=1, max_epochs=epochs)
    model = Model(model_select=model_select)
    try:
        latest_model = [x for x in sorted(os.listdir(model_path)) if x.endswith('.pkl')][-1]
    except:
        latest_model = None
    if latest_model is not None:
        model.load_from_checkpoint(latest_model)
        print(f'[INFO] Use previous model {latest_model}')
    dataset = KITTIDataModule(dataset_path=train_path, batch_size=batch_size, num_workers=num_workers, val_split=val_split)
    trainer.fit(model=model, datamodule=dataset)

