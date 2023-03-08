import os

import imageio.plugins.freeimage
import matplotlib.pyplot as plt
from CycleGAN_wUNet2D import *
# from patchify import patchify, unpatchify
from src.utils import *
# from UNet2dTrain import *

def predict_3D_patches(model, patch_dim, image, step_size, n_classes, device='cpu'):
    """
    TODO
    """ 
    # Get box dim and image dim


    m, n = image.shape[:2]
    p1, p2 = patch_dim
    st1, st2 = step_size
    # Prepare reconstruction volume. Predictions will be summed in this volume.
    recon = np.zeros(shape=(n_classes, m, n), dtype=np.float32)

    for i in np.arange(0, m - p1+1, st1):
        for j in np.arange(0, n - p2+1, st2):
            #
            # patch = np.pad(patch, ((0, d-d_cur[0]), (0, d-d_cur[1]), (0, d-d_cur[2]), (0, 0)),
            #                mode='constant', constant_values=(0))

            single_patch = image[i: i+p1, j: j+p2]
            single_patch = np.expand_dims(single_patch, [0, 1])
            single_patch = torch.from_numpy(single_patch).float().to(device)
            pred = model(single_patch)

            recon[:, i:i+p1, j:j+p2] += pred.squeeze().cpu().detach().numpy()

    for i in np.arange(m - 1, p1-1, -st1):
        for j in np.arange(n-1, p2-1, -st2):
            #
            # patch = np.pad(patch, ((0, d-d_cur[0]), (0, d-d_cur[1]), (0, d-d_cur[2]), (0, 0)),
            #                mode='constant', constant_values=(0))

            single_patch = image[i-p1: i, j-p2:j]
            single_patch = np.expand_dims(single_patch, [0, 1])
            single_patch = torch.from_numpy(single_patch).float().to(device)
            pred = model(single_patch)

            recon[:, i-p1: i, j-p2: j] += pred.squeeze().cpu().detach().numpy()


    print("")

    # Normalize
    # recon /= np.sum(recon, axis=0, keepdims=True)

    return recon


if __name__ == '__main__':

    d_lr: float = 2e-4
    g_lr: float = 2e-4
    epochs = 200
    # epoch_decay = epochs // 2
    img_sz = 256
    in_channels = 1
    out_channels = 64

    patch_dim = [256, 256]
    step_size = [32, 32]
    n_classes = 2



    save_root = '../../data_folder/dataset2D/Test/A/pred'


    if not os.path.exists(save_root):
        os.mkdir(save_root)

    v_num = None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CycleGAN2DUNet(d_lr=d_lr, g_lr=g_lr, img_sz=img_sz,
                           in_channels=in_channels, nb_resblks=2, nb_downsampling=3,
                           out_channels=out_channels,
                           unet_weight=1
                           )


    ckpt_path = get_latest_state(log_root='logs/CycleGAN',
                                 v_num=v_num
                                 )

    print(f'loading model from {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"]

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model = model.eval()



    file_names = [f for f in sorted(os.listdir(save_root)) if not f.startswith('.')]

    for f in file_names:

        image = np.asarray(Image.open(os.path.join(save_root, f)))

        image = (image - image.min())/(image.max() - image.min())

        recon = predict_3D_patches(model.A_Unet, patch_dim, image, step_size, n_classes,
                                   device=device)

        recon = np.argmax(recon, axis=0)

        # save_path = os.path.join(save_root, f)[:-3] + 'png'
        save_path = os.path.join(save_root, f)

        # io.imsave(save_path, recon>0)

        recon = Image.fromarray(recon>0)

        recon.save(save_path)
