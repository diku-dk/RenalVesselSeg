from src.model import *


class Block3D(Block):
    def __init__(self, in_ch, out_ch, norm=None):
        super().__init__(in_ch, out_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding='same')
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding='same')

        if norm is None:
            self.norm = norm
        elif 'batch' in norm.lower():
            self.norm = nn.BatchNorm3d(out_ch)
        elif 'instance' in norm.lower():
            self.norm = nn.InstanceNorm3d(out_ch)
        else:
            print('normalization has to be one of batch or instance')

class Encoder3D(Encoder):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024), norm=None):
        super().__init__(chs=chs)
        self.pool = nn.MaxPool3d(2)
        self.enc_blocks = nn.ModuleList([Block3D(chs[i], chs[i + 1], norm=norm) for i in range(len(chs) - 1)])


class Decoder3D(Decoder):
    def __init__(self, chs=(1024, 512, 256, 128, 64), norm=None):
        super().__init__(chs=chs)

        self.upconvs = nn.ModuleList([nn.ConvTranspose3d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block3D(chs[i], chs[i + 1], norm=norm) for i in range(len(chs) - 1)])

    def crop(self, enc_ftrs, x):
        x_shape = x.shape
        f_shape = enc_ftrs.shape

        if f_shape == x_shape:
            return enc_ftrs

        _, _, H, W, D = x_shape
        _, _, H_f, W_f, D_f = f_shape

        c_H, c_W, c_D = H_f // 2, W_f // 2, D_f // 2

        enc_ftrs = enc_ftrs[:, :, c_H - H // 2: c_H + H // 2, c_W - W // 2: c_W + W // 2, c_D - D // 2: c_D + D // 2]
        return enc_ftrs


class UNet3D(UNet):
    def __init__(self, num_class=1, n_channels=3, retain_dim=False, out_sz=(572, 572),
                 depth=4, first_channel=64,
                 **kwargs):
        super().__init__(num_class=num_class, retain_dim=retain_dim, n_channels=n_channels,
                         out_sz=out_sz, depth=depth,
                         first_channel=first_channel, **kwargs)

        enc_chs = [first_channel * 2 ** i for i in range(depth)]
        dec_chs = list(reversed(enc_chs))
        enc_chs = [n_channels] + enc_chs

        self.encoder = Encoder3D(enc_chs, self.norm)
        self.decoder = Decoder3D(dec_chs, self.norm)
        self.head = nn.Conv3d(dec_chs[-1], num_class, 1)


if __name__ == '__main__':
    patch_size = 65
    model = UNet3D(num_class=2, n_channels=1, retain_dim=True,
                   out_sz=64, depth=3, first_channel=16,
                   flatten_output=False, norm='batch')

    x = torch.rand(2, 1, patch_size, patch_size, patch_size)

    # torch.sum(torch.abs(x - y))

    y = model(x)
