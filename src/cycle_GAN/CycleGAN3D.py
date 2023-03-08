import os

import numpy as np
import torch

from Imports import *
from CycleGAN_PL import *
from data_loader3D import *
from src.model3D import UNet3D
from src.utils import *

class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a Generator -
    maps each the output to the desired number of output channels
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=7, padding=3, padding_mode='reflect')

    def forward(self, x):
        '''
        Function for completing a forward pass of FeatureMapBlock:
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x

class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs a convolution followed by a max pool operation and an optional instance norm.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_bn=True, kernel_size=3, activation='relu'):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, padding=1, stride=2, padding_mode='reflect')
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels * 2)
        self.use_bn = use_bn

    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock:
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x


class ExpandingBlock(nn.Module):
    '''
    ExpandingBlock Class:
    Performs a convolutional transpose operation in order to upsample,
        with an optional instance norm
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()

    def forward(self, x):
        '''
        Function for completing a forward pass of ExpandingBlock:
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
            skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x



class ResBlock3D(nn.Module):

    def __init__(self, in_channels: int, apply_dp: bool = True):
        """
                            Defines a ResBlock
        X ------------------------identity------------------------
        |-- Convolution -- Norm -- ReLU -- Convolution -- Norm --|
        """

        """
        Parameters:
            in_channels:  Number of input channels
            apply_dp:     If apply_dp is set to True, then activations are 0'ed out with prob 0.5
        """

        super().__init__()

        conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding='same')
        layers = [conv, nn.InstanceNorm3d(in_channels), nn.ReLU(True)]

        if apply_dp:
            layers += [nn.Dropout(0.5)]

        conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding='same')
        layers += [conv, nn.InstanceNorm3d(in_channels)]

        self.net = nn.Sequential(*layers)

    def forward(self, x): return x + self.net(x)



class Generator3D(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 64, apply_dp: bool = True, img_sz: int = 128,
                 nb_resblks: int = 6, nb_downsampling: int = 2):

        """
                                Generator Architecture (Image Size: 256)
        c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3,

        where c7s1-k denote a 7 × 7 Conv-InstanceNorm-ReLU layer with k filters and stride 1, dk denotes a 3 × 3
        Conv-InstanceNorm-ReLU layer with k filters and stride 2, Rk denotes a residual block that contains two
        3 × 3 Conv layers with the same number of filters on both layer. uk denotes a 3 × 3 DeConv-InstanceNorm-
        ReLU layer with k filters and stride 1.
        """

        """
        Parameters:
            in_channels:  Number of input channels
            out_channels: Number of output channels
            apply_dp:     If apply_dp is set to True, then activations are 0'ed out with prob 0.5
        """

        super().__init__()

        f = 1

        # conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same')
        # self.layers = [conv, nn.InstanceNorm3d(out_channels), nn.ReLU(True)]

        conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1)
        self.layers = [nn.ReflectionPad3d(3), conv, nn.InstanceNorm3d(out_channels), nn.ReLU(True)]


        for i in range(nb_downsampling):

            # conv = nn.Conv3d(out_channels * f, out_channels * 2 * f, kernel_size=3, padding='same')
            # self.layers += [conv, nn.InstanceNorm3d(out_channels * 2 * f), nn.ReLU(True), nn.MaxPool3d(2)]

            conv = nn.Conv3d(out_channels * f, out_channels * 2 * f, kernel_size=3, stride=2, padding=1)
            self.layers += [conv, nn.InstanceNorm3d(out_channels * 2 * f), nn.ReLU(True)]


            f *= 2

        for i in range(nb_resblks):
            res_blk = ResBlock3D(in_channels=out_channels * f, apply_dp=apply_dp)
            self.layers += [res_blk]

        for i in range(nb_downsampling):
            # conv = nn.ConvTranspose3d(out_channels * f, out_channels * (f // 2), kernel_size=2, stride=2,
            #                           # padding=1, output_padding=1
            #                           )
            # conv2 = ResBlock3D(in_channels=out_channels * (f // 2),
            #                    apply_dp=apply_dp)
            # self.layers += [conv, nn.InstanceNorm3d(out_channels * (f // 2)), nn.ReLU(True), conv2]

            conv = nn.ConvTranspose3d(out_channels * f, out_channels * (f // 2), 3, 2, padding=1, output_padding=1)
            self.layers += [conv, nn.InstanceNorm3d(out_channels * (f // 2)), nn.ReLU(True)]

            f = f // 2
        #
        # conv = nn.Conv3d(in_channels=out_channels, out_channels=in_channels, kernel_size=3, stride=1, padding='same')
        # self.layers += [conv, nn.Tanh()]

        conv = nn.Conv3d(in_channels=out_channels, out_channels=in_channels, kernel_size=7, stride=1)
        self.layers += [nn.ReflectionPad3d(3), conv, nn.Tanh()]


        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)



# class Generator3D(nn.Module):

#     def __init__(self, in_channels: int = 3, out_channels: int = 64, apply_dp: bool = True, img_sz: int = 128,
#                  nb_resblks: int = 6, nb_downsampling: int = 2):


#         super().__init__()

#         f = 1

#         conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same')
#         self.layers = [conv, nn.InstanceNorm3d(out_channels), nn.ReLU(True)]

#         # conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1)
#         # self.layers = [nn.ReflectionPad3d(3), conv, nn.InstanceNorm3d(out_channels), nn.ReLU(True)]


#         for i in range(nb_downsampling):

#             conv = nn.Conv3d(out_channels * f, out_channels * 2 * f, kernel_size=3, padding='same')
#             self.layers += [conv, nn.InstanceNorm3d(out_channels * 2 * f), nn.ReLU(True), nn.MaxPool3d(2)]

#             # conv = nn.Conv3d(out_channels * f, out_channels * 2 * f, kernel_size=3, stride=2, padding=1)
#             # self.layers += [conv, nn.InstanceNorm3d(out_channels * 2 * f), nn.ReLU(True)]


#             f *= 2

#         for i in range(nb_resblks):
#             res_blk = ResBlock3D(in_channels=out_channels * f, apply_dp=apply_dp)
#             self.layers += [res_blk]

#         for i in range(nb_downsampling):
#             conv = nn.ConvTranspose3d(out_channels * f, out_channels * (f // 2), kernel_size=2, stride=2,
#                                       # padding=1, output_padding=1
#                                       )
#             conv2 = ResBlock3D(in_channels=out_channels * (f // 2),
#                                apply_dp=apply_dp)
#             self.layers += [conv, nn.InstanceNorm3d(out_channels * (f // 2)), nn.ReLU(True), conv2]

#             # conv = nn.ConvTranspose3d(out_channels * f, out_channels * (f // 2), 3, 2, padding=1, output_padding=1)
#             # self.layers += [conv, nn.InstanceNorm3d(out_channels * (f // 2)), nn.ReLU(True)]

#             f = f // 2
        
#         conv = nn.Conv3d(in_channels=out_channels, out_channels=in_channels, kernel_size=3, stride=1, padding='same')
#         self.layers += [conv, nn.Tanh()]

#         # conv = nn.Conv3d(in_channels=out_channels, out_channels=in_channels, kernel_size=7, stride=1)
#         # self.layers += [nn.ReflectionPad3d(3), conv, nn.Tanh()]


#         self.net = nn.Sequential(*self.layers)

#     def forward(self, x):
#         return self.net(x)



class Generator3DUNet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 64, apply_dp: bool = True, img_sz: int = 128,
                 nb_resblks: int = 6, nb_downsampling: int = 2):

        super().__init__()

        self.net = UNet3D(num_class=1, n_channels=in_channels, retain_dim=True, first_channel=out_channels,
                           out_sz=img_sz, depth=3, flatten_output=False)

    def forward(self, x):
        return self.net(x)



# class Discriminator3D(nn.Module):

#     def __init__(self, in_channels: int = 3, out_channels: int = 64, nb_layers: int = 3):
#         """
#                                     Discriminator Architecture!
#         C64 - C128 - C256 - C512, where Ck denote a Convolution-InstanceNorm-LeakyReLU layer with k filters
#         """

#         """
#         Parameters:
#             in_channels:    Number of input channels
#             out_channels:   Number of output channels
#             nb_layers:      Number of layers in the 70*70 Patch Discriminator
#         """

#         super().__init__()
#         in_f = 1
#         out_f = 2

#         conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
#         self.layers = [conv, nn.LeakyReLU(0.2, True)]

#         for idx in range(1, nb_layers):
#             conv = nn.Conv3d(out_channels * in_f, out_channels * out_f, kernel_size=3, stride=1, padding='same')
#             self.layers += [conv, nn.InstanceNorm3d(out_channels * out_f), nn.LeakyReLU(0.2, True), nn.MaxPool3d(2)]
#             in_f = out_f
#             out_f *= 2

#         out_f = min(2 ** nb_layers, 8)
#         conv = nn.Conv3d(out_channels * in_f, out_channels * out_f, kernel_size=3, stride=1, padding='same')
#         self.layers += [conv, nn.InstanceNorm3d(out_channels * out_f), nn.LeakyReLU(0.2, True), nn.MaxPool3d(2)]

#         conv = nn.Conv3d(out_channels * out_f, out_channels=1, kernel_size=3, stride=1, padding='same')
#         self.layers += [conv]

#         self.net = nn.Sequential(*self.layers)

#     def forward(self, x): return self.net(x)

class Discriminator3D(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 64, nb_layers: int = 3):
        super().__init__()
        in_f = 1
        out_f = 2

        conv = nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.layers = [conv, nn.LeakyReLU(0.2, True)]

        for idx in range(1, nb_layers):
            conv = nn.Conv3d(out_channels * in_f, out_channels * out_f, kernel_size=4, stride=2, padding=1)
            self.layers += [conv, nn.InstanceNorm3d(out_channels * out_f), nn.LeakyReLU(0.2, True)]
            in_f = out_f
            out_f *= 2

        out_f = min(2 ** nb_layers, 8)
        conv = nn.Conv3d(out_channels * in_f, out_channels * out_f, kernel_size=4, stride=1, padding=1)
        self.layers += [conv, nn.InstanceNorm3d(out_channels * out_f), nn.LeakyReLU(0.2, True)]

        conv = nn.Conv3d(out_channels * out_f, out_channels=1, kernel_size=4, stride=1, padding=1)
        self.layers += [conv]

        self.net = nn.Sequential(*self.layers)

    def forward(self, x): return self.net(x)

class Discriminator3DContract(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake.
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels, hidden_channels=64):
        super().__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False, kernel_size=4, activation='lrelu')
        self.contract2 = ContractingBlock(hidden_channels * 2, kernel_size=4, activation='lrelu')
        self.contract3 = ContractingBlock(hidden_channels * 4, kernel_size=4, activation='lrelu')
        self.final = nn.Conv2d(hidden_channels * 8, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        xn = self.final(x3)
        return xn

class CycleGAN3D(CycleGAN):

    def __init__(self, img_sz: int = 128, in_channels: int = 3, nb_resblks: int = 2, nb_downsampling: int = 3,
                 save_root_path='plots', out_channels: int = 64,
                 **kwargs):

        super().__init__(img_sz=img_sz, in_channels=in_channels,  **kwargs)
        self.save_root_path = save_root_path

        if not os.path.exists(save_root_path):
            os.mkdir(save_root_path)

        init = Initializer(init_type='normal', init_gain=0.02)

        self.d_A = init(Discriminator3D(in_channels=in_channels, out_channels=out_channels, nb_layers=3))
        self.d_B = init(Discriminator3D(in_channels=in_channels, out_channels=out_channels, nb_layers=3))

        self.g_A2B = init(Generator3DUNet(in_channels=in_channels, out_channels=out_channels, apply_dp=False, img_sz=img_sz,
                                      nb_resblks=nb_resblks, nb_downsampling=nb_downsampling))
        self.g_B2A = init(Generator3DUNet(in_channels=in_channels, out_channels=out_channels, apply_dp=False, img_sz=img_sz,
                                      nb_resblks=nb_resblks, nb_downsampling=nb_downsampling))


        self.g_A2B = init(Generator3D(in_channels=in_channels, out_channels=out_channels, apply_dp=False, img_sz=img_sz,
                                      nb_resblks=nb_resblks, nb_downsampling=nb_downsampling))
        self.g_B2A = init(Generator3D(in_channels=in_channels, out_channels=out_channels, apply_dp=False, img_sz=img_sz,
                                      nb_resblks=nb_resblks, nb_downsampling=nb_downsampling))


        self.d_A_params = self.d_A.parameters()
        self.d_B_params = self.d_B.parameters()
        self.g_params = itertools.chain([*self.g_A2B.parameters(), *self.g_B2A.parameters()])



        self.example_input_array = [torch.rand(1, in_channels, img_sz, img_sz, img_sz, device=self.device),
                                    torch.rand(1, in_channels, img_sz, img_sz, img_sz, device=self.device)]

        real_A, real_B = self.example_input_array

        fake_B, fake_A = self(real_A, real_B)
        cyc_A, idt_A, cyc_B, idt_B = self.forward_gen(real_A, real_B, fake_A, fake_B)

        d_A_pred_real_data, d_A_pred_fake_data = self.forward_dis(self.d_A, real_A, fake_A)
        d_B_pred_real_data, d_B_pred_fake_data = self.forward_dis(self.d_B, real_B, fake_B)

        print(f'example_input_array: {real_A.shape}')
        print(f'd_A_pred_real_data shape: {d_A_pred_real_data.shape}')
        print(f'fake_B shape: {fake_B.shape}')

        # G_A2B loss, G_B2A loss, G loss
        g_A2B_loss, g_B2A_loss, g_tot_loss, g_cycloss = self.loss.get_gen_loss(real_A, real_B, cyc_A, cyc_B, idt_A, idt_B,
                                                                    d_A_pred_fake_data, d_B_pred_fake_data)

        # D_A loss, D_B loss
        d_A_loss = self.loss.get_dis_loss(d_A_pred_real_data, d_A_pred_fake_data)
        d_B_loss = self.loss.get_dis_loss(d_B_pred_real_data, d_B_pred_fake_data)


        print('model ok')


    def training_step(self, batch, batch_idx, optimizer_idx):

        real_A, real_B = batch['A'], batch['B']

        real_A = real_A['image'][tio.DATA]
        real_B = real_B['image'][tio.DATA]


        fake_B, fake_A = self(real_A, real_B)

        if optimizer_idx == 0:
            cyc_A, idt_A, cyc_B, idt_B = self.forward_gen(real_A, real_B, fake_A, fake_B)

            # No need to calculate the gradients for Discriminators' parameters
            self.set_requires_grad([self.d_A, self.d_B], requires_grad=False)
            d_A_pred_fake_data = self.d_A(fake_A)
            d_B_pred_fake_data = self.d_B(fake_B)

            g_A2B_loss, g_B2A_loss, g_tot_loss, g_cyc_loss = self.loss.get_gen_loss(real_A, real_B, cyc_A, cyc_B, idt_A, idt_B,
                                                                        d_A_pred_fake_data, d_B_pred_fake_data)

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


    def max_intensity_projection(self, i, dim=1):
        # note that first dim (0) is channel dim
        return [torch.amax(a, dim=dim) if torch.min(a) == 0 else torch.amax((a+1)/2, dim=dim)  for a in i]


    def shared_step(self, batch, stage: str = 'val', batch_idx=0):

        grid_A = []
        grid_B = []

        real_A, real_B = batch['A'], batch['B']

        real_A = real_A['image'][tio.DATA]
        real_B = real_B['image'][tio.DATA]

        fake_B, fake_A = self(real_A, real_B)
        cyc_A, idt_A, cyc_B, idt_B = self.forward_gen(real_A, real_B, fake_A, fake_B)

        d_A_pred_real_data, d_A_pred_fake_data = self.forward_dis(self.d_A, real_A, fake_A)
        d_B_pred_real_data, d_B_pred_fake_data = self.forward_dis(self.d_B, real_B, fake_B)

        # G_A2B loss, G_B2A loss, G loss
        g_A2B_loss, g_B2A_loss, g_tot_loss, g_cyc_loss = self.loss.get_gen_loss(real_A, real_B, cyc_A, cyc_B, idt_A, idt_B,
                                                                    d_A_pred_fake_data, d_B_pred_fake_data)

        # D_A loss, D_B loss
        d_A_loss = self.loss.get_dis_loss(d_A_pred_real_data, d_A_pred_fake_data)
        d_B_loss = self.loss.get_dis_loss(d_B_pred_real_data, d_B_pred_fake_data)

        dict_ = {f'g_tot_{stage}_loss': g_tot_loss, f'g_A2B_{stage}_loss': g_A2B_loss,
                 f'g_B2A_{stage}_loss': g_B2A_loss,
                 f'd_A_{stage}_loss': d_A_loss, f'd_B_{stage}_loss': d_B_loss, f'c_{stage}_cycloss': g_cyc_loss}
        self.log_dict(dict_, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # num_to_show = 3
        #
        # for i in range(min(num_to_show, len(real_A))):
        #     tensor = torch.stack([real_A[i], fake_B[i], cyc_A[i],
        #                           real_B[i], fake_A[i], cyc_B[i]])
        #     tensor = (tensor + 1) / 2
        #     grid_A.append(tensor[:3])
        #     grid_B.append(tensor[3:])

        i = np.random.choice(np.arange(len(real_A)), 1)[0]

        for dim in [1, 2, 3]:

            grid_A.append(torch.stack(
                self.max_intensity_projection([real_A[i], fake_B[i], cyc_A[i]], dim=dim)
                                      )
                          )

            grid_B.append(
                torch.stack(
                    self.max_intensity_projection([real_B[i], fake_A[i], cyc_B[i]], dim=dim)
                            )
            )


        num_to_show = 3
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


    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, 'val', batch_idx)
        # real_A, real_B = batch['A'], batch['B']
        #
        # self.save_images(real_A, batch_idx='A'+str(batch_idx))
        # self.save_images(real_B, batch_idx='B'+str(batch_idx))

        return

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, 'test', batch_idx)


    def test_deterministic(self):
        pass

    def lr_lambda(self, epoch):

        fraction = (epoch - self.epoch_decay) / self.epoch_decay
        return 1 if epoch < self.epoch_decay else 1 - fraction

    def configure_optimizers(self):

        # define the optimizers here
        g_opt = torch.optim.Adam(self.g_params, lr=self.g_lr, betas=(self.beta_1, self.beta_2))
        d_A_opt = torch.optim.Adam(self.d_A_params, lr=self.d_lr, betas=(self.beta_1, self.beta_2))
        d_B_opt = torch.optim.Adam(self.d_B_params, lr=self.d_lr, betas=(self.beta_1, self.beta_2))

        # define the lr_schedulers here
        g_sch = optim.lr_scheduler.LambdaLR(g_opt, lr_lambda=self.lr_lambda)
        d_A_sch = optim.lr_scheduler.LambdaLR(d_A_opt, lr_lambda=self.lr_lambda)
        d_B_sch = optim.lr_scheduler.LambdaLR(d_B_opt, lr_lambda=self.lr_lambda)

        # first return value is a list of optimizers and second is a list of lr_schedulers
        # (you can return empty list also)
        return [g_opt, d_A_opt, d_B_opt], [g_sch, d_A_sch, d_B_sch]


    def save_images(self, batch, batch_idx=0):

        if 'location' in batch.keys():
            del batch['location']

        if 'label' in batch.keys():
            del batch['label']

        batch['image']['affine'] = batch['image']['affine'].cpu()
        batch['image']['data'] = batch['image']['data'].cpu()

        batch_subjects = tio.utils.get_subjects_from_batch(batch)
        # tio.utils_crop.add_images_from_batch(batch_subjects, pred, tio.LabelMap)

        root = os.path.join(self.save_root_path, f'{self.current_epoch}EPOCH')

        if not os.path.exists(root):
            # shutil.rmtree(root)
            os.mkdir(root)

        for i, subject in enumerate(batch_subjects):
            fig_name = os.path.join(root, f'{i}_batch_{batch_idx}_idx.png')
            subject.plot(output_path=fig_name, show=False)

            plt.close('all')


