from CycleGAN_wUNet2D import *


if __name__ == '__main__':


    img_sz = 256

    in_channels = 1

    available_gpus = torch.cuda.device_count()

    accelerator = 'gpu' if available_gpus > 0 else 'cpu'

    num_workers = 0

    test_root = '../../data_folder/dataset2D'

    out_channels = 64
    save_root_path = test_root + str(out_channels)

    trn_batch_sz = 32

    datamodule = DataModule2D(test_root, trn_batch_sz=trn_batch_sz, tst_batch_sz=trn_batch_sz,
                              num_workers=0, img_sz=img_sz,
                            in_channels=in_channels
                            )

    datamodule.prepare_data()
    datamodule.setup("fit")


    TEST = True
    TRAIN = True
    RESTORE = False
    resume_from_checkpoint = None if TRAIN else "path/to/checkpoints/"  # "./logs/CycleGAN/version_0/checkpoints/epoch=1.ckpt"

    d_lr: float = 2e-4
    g_lr: float = 2e-4
    epochs = 1000
    # epoch_decay = epochs // 2

    if TRAIN or RESTORE:
        model = CycleGAN2DUNet(d_lr=d_lr, g_lr=g_lr, img_sz=img_sz,
                               in_channels=in_channels, nb_resblks=2, nb_downsampling=3,
                               out_channels=out_channels,
                               unet_weight=3,
                               save_root_path=save_root_path
                               )

        tb_logger = pl_loggers.TensorBoardLogger('logs/', name="CycleGAN", log_graph=True)

        early_stopping = pl.callbacks.early_stopping.EarlyStopping(
            monitor='g_tot_val_loss', patience=10, mode='min'
        )

        lr_logger = LearningRateMonitor(logging_interval='epoch')
        checkpoint_callback = ModelCheckpoint(monitor="g_tot_val_loss", save_top_k=1,
                                              # period=2,
                                              save_last=True)
        callbacks = [lr_logger, checkpoint_callback,
                     # early_stopping
                     ]


        trainer = pl.Trainer(accelerator=accelerator,
                             strategy='ddp',
                             max_epochs=epochs,
                             # progress_bar_refresh_rate=20,
                             devices=available_gpus if available_gpus > 0 else None,
                             precision=16,
                             callbacks=callbacks, num_sanity_val_steps=1, logger=tb_logger,
                             resume_from_checkpoint=resume_from_checkpoint,
                             # log_every_n_steps=25,
                             profiler="simple",
                             # deterministic=True
                             )

        trainer.fit(model, datamodule)
