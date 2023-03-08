import os

import torch
import torchio as tio
import monai
from src.data_loader_lightning import *
from CycleGAN3D import *
from src.model3D import UNet3D
from src.utils import *
from CycleGAN_w_UNet import CycleGAN3DUNet

class Tester:

    def __init__(self, model, dataset, patch_size, batch_size=2,
                 flatten_y=False, enforce_softmax=True, save_root_path=''
                 ):
        self.model = model
        self.dataset = dataset
        self.test_set = self.dataset.test_set
        self.n_class = self.dataset.n_class
        self.enforce_softmax = enforce_softmax
        self.batch_size = batch_size
        self.flatten_y = flatten_y
        patch_size = patch_size  # if type(patch_size) is list else [patch_size] * 3
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        model.to(device)

        if not os.path.exists(save_root_path):
            os.mkdir(save_root_path)
        self.save_root_path = save_root_path

        self.patch_size = patch_size if patch_size is not None else self.dataset.patch_size

    # def inference(self, test_set=None):
    #
    #
    #     # subjects = self.test_dataloader() if subjects is None else subjects
    #
    #     subjects = self.test_set if test_set is None else test_set
    #
    #     for i, cur_subj in enumerate(subjects):
    #
    #         cur_name = cur_subj['image']['stem']
    #
    #         model = self.model.eval()
    #
    #         with torch.no_grad():
    #
    #             input_tensor = cur_subj['image'][tio.DATA]
    #             input_tensor = torch.unsqueeze(input_tensor, dim=0)
    #
    #             input_tensor.to(self.device)
    #
    #             logits = model.net(input_tensor)
    #             labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
    #
    #             labels = torch.squeeze(labels, dim=0)
    #
    #             affine = cur_subj['image']['affine']
    #             # affine = torch.squeeze(affine)
    #
    #             image = tio.LabelMap(tensor=labels, affine=affine)
    #             image.save(f'{cur_name}.nii.gz')

    def inference_patch(self, patch_overlap=0, test_set=None, patch_size=None, batch_size=4):

        patch_overlap = patch_overlap

        patch_size = self.patch_size if patch_size is None else patch_size

        subjects = self.test_set if test_set is None else test_set

        model = self.model.eval()

        for i, cur_subj in enumerate(subjects):

            cur_name = cur_subj['image']['path']
            cur_name = os.path.basename(cur_name)
            print(f'predicting on {cur_name}')

            grid_sampler = tio.inference.GridSampler(
                cur_subj,
                patch_size,
                patch_overlap,
            )

            patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
            aggregator = tio.inference.GridAggregator(grid_sampler)

            with torch.no_grad():
                for patches_batch in patch_loader:
                    input_tensor = patches_batch['image'][tio.DATA]
                    locations = patches_batch[tio.LOCATION]

                    # input_tensor = torch.unsqueeze(input_tensor, dim=0)
                    # input_tensor = input_tensor.float()

                    input_tensor = input_tensor.to(self.device)

                    logits = model.A_Unet(input_tensor)
                    labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)

                    # labels = torch.squeeze(labels, dim=0)

                    aggregator.add_batch(labels, locations)

                    # labels = labels.cpu()
                    #
                    # if 'location' in patches_batch.keys():
                    #     del patches_batch['location']
                    #
                    # batch_subjects = tio.utils_crop.get_subjects_from_batch(patches_batch)
                    # tio.utils_crop.add_images_from_batch(batch_subjects, labels, tio.LabelMap)
                    # for subject in batch_subjects:
                    #     subject.plot()

            output_tensor = aggregator.get_output_tensor()

            image = tio.LabelMap(tensor=output_tensor, affine=cur_subj['image'].affine)
            image.save(os.path.join(self.save_root_path, cur_name))



if __name__ == '__main__':

    n_class = 2
    patch_size = 64
    in_channels = 1

    gpus = 1

    d_lr: float = 2e-4
    g_lr: float = 2e-4
    epochs = 200
    epoch_decay = epochs // 2
    img_sz = patch_size

    path = 'dataset3D/Test/A'


    batch_size = 1
    patch_size = 208

    num_workers = 0

    label_subdir = 'labels'


    v_num = None


    dataset = PatchDatset(
        dataset_dir=path,
        batch_size=batch_size,
        train_val_ratio=1,
        patch_size=patch_size,
        num_workers=num_workers,
        label_subdir=None,
    )


    dataset.prepare_data()
    dataset.setup()

    model = CycleGAN3DUNet(epoch_decay=epoch_decay, d_lr=d_lr, g_lr=g_lr, img_sz=patch_size,
                           in_channels=in_channels, nb_resblks=3, nb_downsampling=3,
                           unet_weight=1, out_channels=32, n_class=n_class,
                           norm='instance',
                           )


    ckpt_path = get_latest_state(log_root='logs/CycleGAN',
                                 v_num=v_num
                                 )

    save_root_path = os.path.join(os.path.dirname(path), 'pred')

    print(f'loading model from {ckpt_path}')
    checkpoint = torch.load(ckpt_path)
    state_dict = checkpoint["state_dict"]

    model.load_state_dict(state_dict, strict=True)


    tester = Tester(model=model, dataset=dataset, patch_size=patch_size,
                    batch_size=2, flatten_y=False, save_root_path=save_root_path)

    tester.inference_patch(test_set=dataset.test_set, batch_size=1,
                           patch_overlap=patch_size//2,

                           )
