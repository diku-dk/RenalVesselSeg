from CycleGAN_wUNet2D import *


class UNet2D(CycleGAN):

    def __init__(self, img_sz: int = 128, in_channels: int = 3, unet_weight: int = 1,
                 **kwargs):
        super().__init__(img_sz=img_sz, in_channels=in_channels, **kwargs)
        init = Initializer(init_type='normal', init_gain=0.02)

        self.A_Unet = init(UNet(num_class=2, n_channels=1, retain_dim=True,
                                  out_sz=img_sz, depth=4, first_channel=32,
                                  flatten_output=False, norm='batch'))

        self.unet_params = self.A_Unet.parameters()
        self.unet_weight = unet_weight

    def training_step(self, batch, batch_idx, optimizer_idx=0):

        real_B = batch['B']
        B_seg = batch['BLabel']


        fake_A_seg = self.A_Unet(real_B)

        # fake_A = self.A_Unet(real_B.detach())
        #     fake_A = self.A_Unet(real_B)

        # B_seg = B_seg.argmax(dim=1)

        dl = 0


        dice_loss = DiceLoss(include_background=False, reduction='mean',
                             softmax=True, to_onehot_y=True)
        dl = dice_loss(fake_A_seg, B_seg)

        ce = nn.CrossEntropyLoss()(fake_A_seg,  torch.squeeze(B_seg, dim=1))
        # ce = 0
        unet_loss = self.unet_weight * (ce + dl)

        self.log("A Unet loss", unet_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return unet_loss


    def save_images(self, grid, batch_idx='0'):

        root = os.path.join(self.save_root_path, f'{self.current_epoch}EPOCH')

        if not os.path.exists(root):
            # shutil.rmtree(root)
            os.mkdir(root)

        for i, subject in enumerate(grid):
            for j in range(len(subject)):
                img = subject[j].cpu().numpy()
                img = np.moveaxis(img, 0, -1)
                plt.subplot(1, 4, j + 1)
                plt.imshow(img, cmap='gray')

            fig_name = os.path.join(root, f'{i}_batch_{batch_idx}_idx.png')

            plt.savefig(fig_name)

        plt.close('all')

    def shared_step(self, batch, stage: str = 'val', batch_idx=0):


        real_B = batch['B']

        seg_real = batch['BLabel']

        seg_pred = self.A_Unet(real_B)


        dice_loss = DiceLoss(include_background=False, reduction='mean',
                             softmax=True, to_onehot_y=True)
        dl = dice_loss(seg_pred, seg_real)

        ce = nn.CrossEntropyLoss()(seg_pred,  torch.squeeze(seg_real, dim=1))
        # ce = 0
        unet_loss = self.unet_weight * (ce + dl)


        # loss = nn.CrossEntropyLoss()(seg_pred, torch.argmax(seg_real, dim=1))

        dict_ = {f'{stage}_unetloss': unet_loss}
        self.log_dict(dict_, on_step=False, on_epoch=True, prog_bar=True, logger=True)


        seg_pred = seg_pred.argmax(dim=1, keepdim=True)

        # seg_real = seg_real.argmax(dim=1, keepdim=True)

        if real_B.min() < 0:
            real_B = (real_B + 1)/2

        grid_B = []

        for i in np.random.choice(np.arange(len(real_B)), min(3, len(real_B))):

            grid_B.append(
                torch.stack(
                    [real_B[i], seg_pred[i], seg_real[i]]
                )
            )

        num_to_show = 3
        # log the results on tensorboard

        self.save_images(grid_B, batch_idx='B'+str(batch_idx))
        grid_B = torchvision.utils.make_grid(torch.cat(grid_B, 0), nrow=num_to_show)
        self.logger.experiment.add_image('Grid_B', grid_B, self.current_epoch, dataformats="CHW")



        # self.save_images(real_B_batch, 0)


    def configure_optimizers(self):

        unet_opt = torch.optim.Adam(self.unet_params, lr=g_lr,
                                    betas=(self.beta_1, self.beta_2)
                                    )

        unet_sch = optim.lr_scheduler.LambdaLR(unet_opt, lr_lambda=self.lr_lambda)

        unet_lambda  = lambda epoch: max(0.99 ** (epoch), min_lr / self.g_lr)

        min_lr = 3e-5

        unet_sch = optim.lr_scheduler.LambdaLR(unet_opt, lr_lambda=unet_lambda)


        # first return value is a list of optimizers and second is a list of lr_schedulers
        # (you can return empty list also)
        return [unet_opt], [unet_sch]


if __name__ == '__main__':


    img_sz = 256

    in_channels = 1

    available_gpus = torch.cuda.device_count()

    accelerator = 'gpu' if available_gpus > 0 else 'cpu'

    num_workers = 0

    root_dir = 'DRIVEUNet'

    trn_batch_sz = 32
    out_channels = 64

    datamodule = DataModule2DSingle(root_dir, trn_batch_sz=trn_batch_sz, tst_batch_sz=trn_batch_sz,
                              num_workers=0, img_sz=img_sz,
                            in_channels=in_channels
                            )

    datamodule.prepare_data()
    datamodule.setup("fit")


    TEST = True
    TRAIN = True
    RESTORE = False
    resume_from_checkpoint = None if TRAIN else "path/to/checkpoints/"  # "./logs/CycleGAN/version_0/checkpoints/epoch=1.ckpt"

    d_lr: float = 5e-3
    g_lr: float = 5e-3

    d_lr: float = 2e-4
    g_lr: float = 2e-4
    epochs = 1e5
    epochs = int(epochs)
    epoch_decay = epochs // 2

    if TRAIN or RESTORE:
        model = UNet2D(epoch_decay=epoch_decay, d_lr=d_lr, g_lr=g_lr, img_sz=img_sz,
                               in_channels=in_channels, nb_resblks=2, nb_downsampling=3,
                               out_channels=out_channels,
                               unet_weight=3,
                               save_root_path='unet_only_plot'
                               )

        tb_logger = pl_loggers.TensorBoardLogger('logs/', name="CycleGAN", log_graph=True)

        early_stopping = pl.callbacks.early_stopping.EarlyStopping(
            monitor='val_unetloss', patience=10, mode='min'
        )

        lr_logger = LearningRateMonitor(logging_interval='epoch')
        checkpoint_callback = ModelCheckpoint(monitor="val_unetloss", save_top_k=1,
                                              # period=2,
                                              save_last=True)
        callbacks = [lr_logger, checkpoint_callback,
                     # early_stopping
                     ]

        # you can change the gpus argument to how many you have (I had only 1 :( )
        # Set the deterministic flag to True for full reproducibility
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
