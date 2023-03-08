import os

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import torchvision
from monai.losses.dice import DiceLoss, MaskedDiceLoss
import monai

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, norm=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding='same')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding='same')

        if norm is None:
            self.norm = norm
        elif 'batch' in norm.lower():
            self.norm = nn.BatchNorm2d(out_ch)
        elif 'instance' in norm.lower():
            self.norm = nn.InstanceNorm2d(out_ch)
        else:
            print('normalization has to be one of batch or instance')

    def forward(self, x):
        if self.norm is None:
            return self.relu(self.conv2(self.relu(self.conv1(x))))
        else:
            return self.norm(self.relu(self.conv2(self.relu(self.conv1(x)))))

class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024), norm=None):
        super().__init__()
        self.enc_blocks = nn.ModuleList([
            Block(chs[i], chs[i + 1], norm=norm) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64), norm=None):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1], norm=norm) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    # def crop(self, enc_ftrs, x):
    #     _, _, H, W = x.shape
    #     enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
    #     return enc_ftrs

    def crop(self, enc_ftrs, x):
        x_shape = x.shape
        f_shape = enc_ftrs.shape

        if f_shape == x_shape:
            return enc_ftrs

        _, _, H, W = x_shape
        _, _, H_f, W_f = f_shape

        c_H, c_W, = H_f // 2, W_f // 2

        enc_ftrs = enc_ftrs[:, :,
                   c_H - H // 2: c_H + H // 2, c_W - W // 2: c_W + W // 2]
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, num_class=1, n_channels=3, first_channel=64, depth=4,
                 retain_dim=False, out_sz=(572, 572), flatten_output=True, norm=None, bottom=None):
        super().__init__()
        self.n_channels = n_channels
        enc_chs = [first_channel * 2 ** i for i in range(depth)]
        dec_chs = list(reversed(enc_chs))
        enc_chs = [n_channels] + enc_chs
        self.encoder = Encoder(enc_chs, norm)
        self.decoder = Decoder(dec_chs, norm)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim
        self.out_sz = out_sz
        self.num_class = num_class
        self.flatten_output = flatten_output
        self.norm = norm

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)

        if self.flatten_output:
            out = out.contiguous().view(x.size(0), -1, self.num_class)

        return out


class Model(pl.LightningModule):
    def __init__(self, net, criterion=nn.CrossEntropyLoss(),
                 learning_rate=5e-4, optimizer_class=torch.optim.Adam,
                 save_root_path='./', argmax=True,
                 epoch_decay=50, per_n_epochs=1,
                 ):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.argmax = argmax
        self.save_root_path = save_root_path
        self.epoch_decay = epoch_decay
        self.per_n_epochs = per_n_epochs

        if not os.path.exists(save_root_path):
            os.mkdir(save_root_path)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)

        # g_sch = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_lambda)

        INITIAL_LEARNING_RATE = self.lr
        min_lr = 1e-4
        g_sch = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch:
        max(0.99 ** (epoch/self.per_n_epochs), min_lr / INITIAL_LEARNING_RATE))

        return {"optimizer": optimizer, "lr_scheduler": g_sch}

    def prepare_batch(self, batch):
        return batch['image'][tio.DATA], batch['label'][tio.DATA]

    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)

        if self.argmax:
            y = torch.argmax(y, dim=1)

        y_hat = self.net(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)

        # y_hat = y_hat.contiguous().view(y_hat.size(0), self.num_class, -1)
        # y = y_hat.contiguous().view(y_hat.size(0), self.num_class, -1)

        # loss = self.criterion(y_hat, y)
        loss = nn.CrossEntropyLoss()(y_hat, torch.argmax(y, dim=1))

        self.log('train_loss', loss, prog_bar=True, batch_size=1)

        dice_loss = DiceLoss(include_background=True, reduction='mean', softmax=True, to_onehot_y=False)
        dl = dice_loss(y_hat, y)
        self.log('dice_loss', dl, prog_bar=True, batch_size=1)

        return loss + dl

    def combined_binary_entropy(self, y_pred, y_true):
        y_true_binary = y_true > 0  # un-one_hot
        y_pred_binary = torch.sum(y_pred[..., 1:], dim=-1, keepdim=False)

        be = nn.BCEWithLogitsLoss()(y_pred_binary, y_true_binary)
        ce = self.criterion(y_pred, y_true)

        return be + ce

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        # loss = self.criterion(y_hat, y)
        loss = nn.CrossEntropyLoss()(y_hat, torch.argmax(y, dim=1))

        self.log('val_loss', loss, batch_size=1)

        # if self.current_epoch % 10 == 0:

        if batch_idx == 0:
            self.save_images(batch, batch_idx)

        dice = self.masked_dice(y_hat, y)

        self.log('val_dice', dice, batch_size=1)

        # acc = self.compute_acc(y, y_hat)
        # self.log('val_acc', acc, batch_size=1)

        return loss

    def test_step(self, *args, **kwargs):
        pass

    def compute_acc(self, y, y_hat):
        y = torch.argmax(y, dim=1)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(torch.eq(y, y_hat)) / len(y.view(-1))
        return acc

    def test_deterministic(self):
        pass

    def lr_lambda(self, epoch):

        fraction = (epoch - self.epoch_decay) / self.epoch_decay
        return 1 if epoch < self.epoch_decay else 1 - fraction


    def save_images(self, batch, batch_idx=0):

        if 'location' in batch.keys():
            del batch['location']

        names = [os.path.basename(i) for i in batch['image']['path']]

        label = batch['label'][tio.DATA]
        img = batch['image'][tio.DATA]

        label = label.argmax(dim=1, keepdim=True).cpu()
        batch['label']['data'] = label

        batch['label']['affine'] = batch['label']['affine'].cpu()

        batch['image']['affine'] = batch['image']['affine'].cpu()
        batch['image']['data'] = batch['image']['data'].cpu()

        if 'weight' in batch.keys():
            weight = batch['weight'][tio.DATA]
            weight = weight.argmax(dim=1, keepdim=True).cpu()

            batch['weight']['affine'] = batch['weight']['affine'].cpu()
            batch['weight']['data'] = weight

            del batch['weight']

        pred = self.net(img)
        pred = pred.argmax(dim=1, keepdim=True).cpu()

        batch_subjects = tio.utils.get_subjects_from_batch(batch)
        tio.utils.add_images_from_batch(batch_subjects, pred, tio.LabelMap)
        # tio.utils_crop.add_images_from_batch(batch_subjects, weight, tio.LabelMap)

        root = os.path.join(self.save_root_path, f'{self.current_epoch}EPOCH')

        if not os.path.exists(root):
            # shutil.rmtree(root)
            os.makedirs(root, exist_ok=True)

        for i, subject in enumerate(batch_subjects):
            fig_name = os.path.join(root, f'{i}_batch_{batch_idx}_idx_{names[i]}.png')
            subject.plot(output_path=fig_name, show=False)
            plt.title(names[i])
            plt.close('all')

    def masked_dice(self, y_pred, y):

        y_pred = y_pred.argmax(dim=1)
        y_pred = F.one_hot(y_pred, num_classes=y.shape[1])
        y_pred = torch.moveaxis(y_pred, -1, 1)

        include_background = False
        from monai.metrics.utils import do_metric_reduction, ignore_background
        if not include_background:
            y_pred, y = ignore_background(y_pred=y_pred, y=y)

        y = y.float()
        y_pred = y_pred.float()

        if y.shape != y_pred.shape:
            raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

        n_len = len(y_pred.shape)
        reduce_axis = list(range(2, n_len))
        intersection = torch.sum(y * y_pred, dim=reduce_axis)

        y_o = torch.sum(y, dim=reduce_axis)
        y_pred_o = torch.sum(y_pred, dim=reduce_axis)
        denominator = y_o + y_pred_o

        res = (2.0 * intersection) / denominator

        return torch.nanmean(res)


class MaskedModel(Model):

    def __init__(self, mask_full_off=False, **kwargs):
        super().__init__(**kwargs)
        self.mask_full_off = mask_full_off

    def prepare_batch(self, batch):

        return batch['image'][tio.DATA], batch['label'][tio.DATA], batch['weight'][tio.DATA]


    def infer_batch(self, batch, no_one_hot=True):
        x, y, w = self.prepare_batch(batch)

        if (not self.argmax) and no_one_hot:
            y = torch.argmax(y, dim=1)

        y_hat = self.net(x)
        return y_hat, y, w

    def training_step(self, batch, batch_idx):
        y_hat, y, w = self.infer_batch(batch)

        # y_hat = y_hat.contiguous().view(y_hat.size(0), self.num_class, -1)
        # y = y_hat.contiguous().view(y_hat.size(0), self.num_class, -1)

        loss = self.combined_binary_entropy(y_hat, y, w)
        self.log('train_loss', loss, prog_bar=True, batch_size=1)
        return loss

    def masked_dice(self, y_pred, y, w):

        y_pred = y_pred.argmax(dim=1)
        y_pred = F.one_hot(y_pred, num_classes=y.shape[1])
        y_pred = torch.moveaxis(y_pred, -1, 1)

        include_background = False
        from monai.metrics.utils import do_metric_reduction, ignore_background
        if not include_background:
            y_pred, y = ignore_background(y_pred=y_pred, y=y)

        y = y.float()
        y_pred = y_pred.float()

        w = w.argmax(dim=1, keepdims=True)
        repeat = [1] * len(w.shape)
        repeat[1] = y.shape[1]
        w = w.repeat(*repeat)

        if y.shape != y_pred.shape:
            raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

        # reducing only spatial dimensions (not batch nor channels)
        n_len = len(y_pred.shape)
        reduce_axis = list(range(2, n_len))
        intersection = torch.sum(y * y_pred * w, dim=reduce_axis)

        y_o = torch.sum(y * w, dim=reduce_axis)
        y_pred_o = torch.sum(y_pred * w, dim=reduce_axis)
        denominator = y_o + y_pred_o

        # res = torch.where(denominator > 0, (2.0 * intersection) / denominator, torch.tensor(1.0, device=y_o.device))
        res = (2.0 * intersection) / denominator
        return torch.nanmean(res)

    def combined_binary_entropy(self, y_pred, y_true, w=None):

        if w is None:
            w = torch.ones_like(y_true)

        repeat = [1] * len(w.shape)
        repeat[1] = y_true.shape[1]
        w = w.repeat(*repeat)

        y_true_binary = y_true > 0
        y_true_binary = y_true_binary.to(torch.float32)

        y_pred_binary = torch.sum(y_pred[:, 1:, ...], dim=1, keepdim=False)

        be = nn.BCEWithLogitsLoss()(y_pred_binary, y_true_binary)
        ce = nn.CrossEntropyLoss(reduction='none')(y_pred, y_true)

        if w.shape != y_true.shape:
            w = w.argmax(dim=1)

        ce = ce * w

        if self.mask_full_off:
            be = be * w

        # ce = torch.mean(ce)

        ce = 2 * torch.sum(ce)/(1 + torch.sum(w))

        dice_loss = MaskedDiceLoss(include_background=False, reduction='mean', softmax=True, to_onehot_y=True)
        dl = 2 * dice_loss(y_pred, torch.unsqueeze(y_true, dim=1), mask=torch.unsqueeze(w, dim=1))

        # dice_loss = DiceLoss(include_background=False, reduction='mean', softmax=True, to_onehot_y=True)
        # dl2 = 2 * dice_loss(y_pred, torch.unsqueeze(y_true, dim=1))


        return be + ce + dl

    def validation_step(self, batch, batch_idx):
        y_hat, y, w = self.infer_batch(batch, no_one_hot=False)
        loss = self.combined_binary_entropy(y_hat, torch.argmax(y, dim=1), w)

        self.log('val_loss', loss, batch_size=1)

        dice = self.masked_dice(y_hat, y, w)

        self.log('val_dice', dice, batch_size=1)


        # acc = self.compute_acc(y, y_hat)
        # self.log('val_acc', acc, batch_size=1)


        # if self.current_epoch % 10 == 0:
        self.save_images(batch, batch_idx)


        return loss
