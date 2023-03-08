import matplotlib.pyplot as plt

from CycleGAN3D import *
from monai.losses.dice import DiceLoss, MaskedDiceLoss


class CycleGAN3DUNet(CycleGAN3D):

    def __init__(self, img_sz: int = 128, in_channels: int = 3, unet_weight: int = 1, n_class: int = 2,
                 out_channels: int = 32,
                 **kwargs):
        super().__init__(img_sz=img_sz, in_channels=in_channels, out_channels=out_channels, **kwargs)
        init = Initializer(init_type='normal', init_gain=0.02)

        self.A_Unet = init(UNet3D(num_class=n_class, n_channels=1, retain_dim=True,
                                  out_sz=img_sz, depth=4, first_channel=out_channels,
                                  flatten_output=False))

        self.unet_params = self.A_Unet.parameters()
        self.unet_weight = unet_weight



        self.example_input_array = [torch.rand(1, in_channels, img_sz, img_sz, img_sz, device=self.device),
                                    torch.rand(1, in_channels, img_sz, img_sz, img_sz, device=self.device)]

        real_A, _ = self.example_input_array

        real_A_seg = self.A_Unet(real_A)

        dis = self.d_A(real_A)
        gen = self.g_A2B(real_A)

        print(f'A_Unet shape: {real_A_seg.shape}')
        print(f'd_A shape: {dis.shape}')
        print(f'gen_A shape: {gen.shape}')


    def training_step(self, batch, batch_idx, optimizer_idx):

        real_A, real_B = batch['A'], batch['B']
        B_seg = real_B['label'][tio.DATA]

        real_A = real_A['image'][tio.DATA]
        real_B = real_B['image'][tio.DATA]

        fake_B, fake_A = self(real_A, real_B)

        if optimizer_idx == 0:
            cyc_A, idt_A, cyc_B, idt_B = self.forward_gen(real_A, real_B, fake_A, fake_B)

            # No need to calculate the gradients for Discriminators' parameters
            self.set_requires_grad([self.d_A, self.d_B], requires_grad=False)
            d_A_pred_fake_data = self.d_A(fake_A)
            d_B_pred_fake_data = self.d_B(fake_B)

            g_A2B_loss, g_B2A_loss, g_tot_loss, g_cyc_loss = self.loss.get_gen_loss(real_A, real_B, cyc_A, cyc_B,
                                                                                    idt_A, idt_B,
                                                                                    d_A_pred_fake_data,
                                                                                    d_B_pred_fake_data)

            dict_ = {'g_tot_train_loss': g_tot_loss, 'g_A2B_train_loss': g_A2B_loss,
                     'g_B2A_train_loss': g_B2A_loss, 'g_cyc_loss': g_cyc_loss}
            self.log_dict(dict_, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            return g_tot_loss

        if optimizer_idx == 1:
            self.set_requires_grad([self.d_A], requires_grad=True)
            fake_A = self.fake_pool_A.push_and_pop(fake_A)
            d_A_pred_real_data, d_A_pred_fake_data = self.forward_dis(self.d_A, real_A, fake_A.detach())

            # GAN loss
            d_A_loss = self.loss.get_dis_loss(d_A_pred_real_data, d_A_pred_fake_data)
            self.log("d_A_train_loss", d_A_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            return d_A_loss

        if optimizer_idx == 2:
            self.set_requires_grad([self.d_B], requires_grad=True)
            fake_B = self.fake_pool_B.push_and_pop(fake_B)
            d_B_pred_real_data, d_B_pred_fake_data = self.forward_dis(self.d_B, real_B, fake_B.detach())

            # GAN loss
            d_B_loss = self.loss.get_dis_loss(d_B_pred_real_data, d_B_pred_fake_data)
            self.log("d_B_train_loss", d_B_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            return d_B_loss

        if optimizer_idx == 3:
            self.set_requires_grad([self.d_A, self.d_B], requires_grad=False)

            # fake_A = self.fake_pool_A.push_and_pop(fake_A)

            fake_A_seg = self.A_Unet(fake_A.detach())

            dice_loss = DiceLoss(include_background=False, reduction='mean',
                                 softmax=True, to_onehot_y=False)
            dl = dice_loss(fake_A_seg, B_seg)
            ce = nn.CrossEntropyLoss()(fake_A_seg, B_seg.argmax(dim=1))
            unet_loss = self.unet_weight * (ce + dl)
            # CrossEntropyLoss
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
                plt.axis('off')

            fig_name = os.path.join(root, f'{i}_batch_{batch_idx}_idx.png')


            plt.savefig(fig_name)

        plt.close('all')

    # def save_images(self, batch, batch_idx=0):
    #
    #     if 'location' in batch.keys():
    #         del batch['location']
    #
    #     label = batch['label'][tio.DATA]
    #     img = batch['image'][tio.DATA]
    #
    #     label = label.argmax(dim=1, keepdim=True).cpu()
    #     batch['label']['data'] = label
    #
    #     batch['label']['affine'] = batch['label']['affine'].cpu()
    #
    #     batch['image']['affine'] = batch['image']['affine'].cpu()
    #     batch['image']['data'] = batch['image']['data'].cpu()
    #
    #     pred = self.A_Unet(img)
    #     pred = pred.argmax(dim=1, keepdim=True).cpu()
    #
    #     batch_subjects = tio.utils_crop.get_subjects_from_batch(batch)
    #     tio.utils_crop.add_images_from_batch(batch_subjects, pred, tio.LabelMap)
    #
    #     root = os.path.join(self.save_root_path, f'{self.current_epoch}EPOCH')
    #
    #     if not os.path.exists(root):
    #         # shutil.rmtree(root)
    #         os.mkdir(root)
    #
    #     for i, subject in enumerate(batch_subjects):
    #         fig_name = os.path.join(root, f'{i}_batch_{batch_idx}_idx.png')
    #         subject.plot(output_path=fig_name, show=False)
    #
    #         plt.close('all')


    def shared_step(self, batch, stage: str = 'val', batch_idx=0):

        if batch_idx > 100:
            return

        grid_A = []
        grid_B = []

        real_A_batch, real_B_batch = batch['A'], batch['B']
        #
        # grid_C = []
        # for dim in [1, 2, 3]:
        #     grid_C.append(
        #         torch.stack(
        #             self.max_intensity_projection([real_B_batch['image'][tio.DATA][0],
        #                                            real_B_batch['label'][tio.DATA].argmax(dim=1, keepdim=True)[0]],
        #                                           dim=dim)
        #         )
        #     )
        # # log the results on tensorboard
        # grid_C = torchvision.utils_crop.make_grid(torch.cat(grid_C, 0), nrow=2)
        #
        # self.logger.experiment.add_image('Grid_C', grid_C, self.current_epoch, dataformats="CHW")

        seg_real = real_B_batch['label'][tio.DATA]

        real_A = real_A_batch['image'][tio.DATA]
        real_B = real_B_batch['image'][tio.DATA]



        fake_B, fake_A = self(real_A, real_B)
        cyc_A, idt_A, cyc_B, idt_B = self.forward_gen(real_A, real_B, fake_A, fake_B)

        d_A_pred_real_data, d_A_pred_fake_data = self.forward_dis(self.d_A, real_A, fake_A)
        d_B_pred_real_data, d_B_pred_fake_data = self.forward_dis(self.d_B, real_B, fake_B)

        # G_A2B loss, G_B2A loss, G loss


        #
        g_A2B_loss, g_B2A_loss, g_tot_loss, g_cyc_loss = self.loss.get_gen_loss(real_A, real_B, cyc_A, cyc_B, idt_A,
                                                                                idt_B,
                                                                                d_A_pred_fake_data, d_B_pred_fake_data)
        #
        # # D_A loss, D_B loss
        # d_A_loss = self.loss.get_dis_loss(d_A_pred_real_data, d_A_pred_fake_data)
        # d_B_loss = self.loss.get_dis_loss(d_B_pred_real_data, d_B_pred_fake_data)
        #
        # dict_ = {f'g_tot_{stage}_loss': g_tot_loss, f'g_A2B_{stage}_loss': g_A2B_loss,
        #          f'g_B2A_{stage}_loss': g_B2A_loss,
        #          f'd_A_{stage}_loss': d_A_loss, f'd_B_{stage}_loss': d_B_loss, f'c_{stage}_cycloss': g_cyc_loss}
        #

        seg_pred = self.A_Unet(fake_A)
                
        dice_loss = DiceLoss(include_background=False, reduction='mean',
                             softmax=True, to_onehot_y=False)
        dl = dice_loss(seg_pred, seg_real)
        ce = nn.CrossEntropyLoss()(seg_pred, seg_real.argmax(dim=1))
        unet_loss = self.unet_weight * (ce + dl)
        g_tot_loss += unet_loss
        
        dict_ = {f'g_tot_{stage}_loss': g_tot_loss, f'unet_{stage}_loss': unet_loss}

        self.log_dict(dict_, on_step=False, on_epoch=True, prog_bar=True, logger=True)


        seg_pred = seg_pred.argmax(dim=1, keepdim=True)

        seg_real = seg_real.argmax(dim=1, keepdim=True)

        i = np.random.choice(np.arange(len(real_A)), 1)[0]

        for dim in [1, 2, 3]:
            grid_A.append(torch.stack(
                self.max_intensity_projection([real_A[i], fake_B[i], cyc_A[i], seg_pred[i]], dim=dim)
            )
            )

            grid_B.append(
                torch.stack(
                    self.max_intensity_projection([real_B[i], fake_A[i], cyc_B[i], seg_real[i]], dim=dim)
                )
            )

        self.save_images(grid_A, batch_idx=str(batch_idx) + 'A')
        self.save_images(grid_B, batch_idx=str(batch_idx) + 'B')

        num_to_show = 4
        # log the results on tensorboard
        grid_A = torchvision.utils.make_grid(torch.cat(grid_A, 0), nrow=num_to_show)
        grid_B = torchvision.utils.make_grid(torch.cat(grid_B, 0), nrow=num_to_show)

        self.logger.experiment.add_image('Grid_A', grid_A, self.current_epoch, dataformats="CHW")
        self.logger.experiment.add_image('Grid_B', grid_B, self.current_epoch, dataformats="CHW")

        #
        # grid = torchvision.utils_crop.make_grid(fake_A)
        # self.logger.experiment.add_image("generated_images_A", grid, self.current_epoch)
        #
        # grid = torchvision.utils_crop.make_grid(fake_B)
        # self.logger.experiment.add_image("generated_images_B", grid, self.current_epoch)

        # self.save_images(real_B_batch, batch_idx)


    def configure_optimizers(self):

        # define the optimizers here
        g_opt = torch.optim.Adam(self.g_params, lr=self.g_lr, betas=(self.beta_1, self.beta_2))
        d_A_opt = torch.optim.Adam(self.d_A_params, lr=self.d_lr, betas=(self.beta_1, self.beta_2))
        d_B_opt = torch.optim.Adam(self.d_B_params, lr=self.d_lr, betas=(self.beta_1, self.beta_2))

        unet_opt = torch.optim.Adam(self.unet_params, lr=self.g_lr, betas=(self.beta_1, self.beta_2))

        # define the lr_schedulers here
        g_sch = optim.lr_scheduler.LambdaLR(g_opt, lr_lambda=self.lr_lambda)
        d_A_sch = optim.lr_scheduler.LambdaLR(d_A_opt, lr_lambda=self.lr_lambda)
        d_B_sch = optim.lr_scheduler.LambdaLR(d_B_opt, lr_lambda=self.lr_lambda)

        unet_sch = optim.lr_scheduler.LambdaLR(unet_opt, lr_lambda=self.lr_lambda)

        min_lr = 3e-5

        g_lambda = lambda epoch: max(0.99 ** (epoch), min_lr / self.g_lr)
        d_A_lambda  = lambda epoch: max(0.99 ** (epoch), min_lr / self.d_lr)
        d_B_lambda  = lambda epoch: max(0.99 ** (epoch), min_lr / self.d_lr)
        unet_lambda  = lambda epoch: max(0.99 ** (epoch), min_lr / self.g_lr)
        g_sch = optim.lr_scheduler.LambdaLR(g_opt, lr_lambda=g_lambda)
        d_A_sch = optim.lr_scheduler.LambdaLR(d_A_opt, lr_lambda=d_A_lambda)
        d_B_sch = optim.lr_scheduler.LambdaLR(d_B_opt, lr_lambda=d_B_lambda)
        unet_sch = optim.lr_scheduler.LambdaLR(unet_opt, lr_lambda=unet_lambda)


        # first return value is a list of optimizers and second is a list of lr_schedulers
        # (you can return empty list also)
        return [g_opt, d_A_opt, d_B_opt, unet_opt], [g_sch, d_A_sch, d_B_sch, unet_sch]


