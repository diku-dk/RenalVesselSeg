from CycleGAN_w_UNet import *


if __name__ == '__main__':


    batch_size = 1

    patch_size = 208
    num_workers = 0

    path = '../../data_folder/dataset3D/Train'

    n_class = 2
    label_subdir = 'labels'

    datamodule = PatchDatasetDouble(
        dataset_dir=path,
        batch_size=batch_size,
        train_val_ratio=1,
        patch_size=patch_size,
        num_workers=num_workers,
        label_subdir=label_subdir,
        sampling_strategy='Uniform',
        # sampling_strategy='Label',
        include_label=True,
        n_class=n_class,
        samples_per_volume=1000,
        queue_length=1000,
    )

    in_channels = 1

    available_gpus = torch.cuda.device_count()
    accelerator = 'gpu' if available_gpus > 0 else 'cpu'

    torch.set_float32_matmul_precision('medium')

    print(f'available_gpus = {available_gpus}')

    TEST = True
    TRAIN = True
    RESTORE = False
    resume_from_checkpoint = None if TRAIN else "path/to/checkpoints/"  # "./logs/CycleGAN/version_0/checkpoints/epoch=1.ckpt"

    d_lr: float = 2e-4
    g_lr: float = 2e-4
    epochs = 200
    epoch_decay = epochs // 5

    if TRAIN or RESTORE:
        model = CycleGAN3DUNet(epoch_decay=epoch_decay, d_lr=d_lr, g_lr=g_lr, img_sz=patch_size,
                               in_channels=in_channels, nb_resblks=3, nb_downsampling=3,
                               unet_weight=3, out_channels=32, n_class=n_class,
                               save_root_path=f'3DGAN{patch_size}',
                               norm='instance'
                               )

        tb_logger = pl_loggers.TensorBoardLogger('logs/', name="CycleGAN", log_graph=True)

        lr_logger = LearningRateMonitor(logging_interval='epoch')
        checkpoint_callback = ModelCheckpoint(monitor="g_tot_val_loss", save_top_k=1,
                                              # period=2,
                                              save_last=True)
        callbacks = [lr_logger, checkpoint_callback]

        # you can change the gpus argument to how many you have (I had only 1 :( )
        # Set the deterministic flag to True for full reproducibility

        trainer = pl.Trainer(accelerator=accelerator,
                             strategy='ddp',
                             max_epochs=epochs,
                             devices=available_gpus if available_gpus > 0 else None,
                             # progress_bar_refresh_rate=20,
                             precision=16,
                             callbacks=callbacks, num_sanity_val_steps=1,
                             logger=tb_logger,
                             resume_from_checkpoint=resume_from_checkpoint,
                             # log_every_n_steps=25,
                             profiler="simple",
                             # deterministic=True
                             )

        trainer.fit(model, datamodule)

