# Cluster 3

def main():
    num_epochs = 20
    patch_size = 128
    queue_length = 100
    patches_per_volume = 5
    batch_size = 2
    one_subject = Subject(T1=ScalarImage('../BRATS2018_crop_renamed/LGG75_T1.nii.gz'), T2=ScalarImage('../BRATS2018_crop_renamed/LGG75_T2.nii.gz'), label=LabelMap('../BRATS2018_crop_renamed/LGG75_Label.nii.gz'))
    another_subject = Subject(T1=ScalarImage('../BRATS2018_crop_renamed/LGG74_T1.nii.gz'), label=LabelMap('../BRATS2018_crop_renamed/LGG74_Label.nii.gz'))
    subjects = [one_subject, another_subject]
    subjects_dataset = SubjectsDataset(subjects)
    queue_dataset = Queue(subjects_dataset, queue_length, patches_per_volume, UniformSampler(patch_size))
    batch_loader = tio.SubjectsLoader(queue_dataset, batch_size=batch_size, collate_fn=lambda x: x)
    model = nn.Identity()
    for epoch_index in range(num_epochs):
        logging.info('Epoch %s', epoch_index)
        for batch in batch_loader:
            logits = model(batch)
            logging.info([batch[idx].keys() for idx in range(batch_size)])
            logging.info(logits.shape)
    logging.info('')

