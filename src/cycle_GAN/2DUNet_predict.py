import os

import imageio.plugins.freeimage
import matplotlib.pyplot as plt
from UNet2DTrain import *
# from patchify import patchify, unpatchify
from src.utils import *

#
# image = np.random.random((512, 512))
# patches = patchify(image, (256, 256), step=128)
# for i in range(patches.shape[0]):
#     for j in range(patches.shape[1]):
#         single_patch = patches[i, j]
#         single_patch = np.expand_dims(single_patch, [0, 1])
#         single_patch = torch.from_numpy(single_patch)
#


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
    epoch_decay = epochs // 2
    img_sz = 256
    in_channels = 1
    out_channels = 64


    v_num = 9

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet2D(epoch_decay=epoch_decay, d_lr=d_lr, g_lr=g_lr, img_sz=img_sz,
                           in_channels=in_channels, nb_resblks=2, nb_downsampling=3,
                           out_channels=out_channels,
                           unet_weight=3,
                           save_root_path='unet_only_plot'
                           )




    save_root_path = '../saved_images'

    ckpt_path = get_latest_state(log_root='logs/CycleGAN',
                                 v_num=v_num
                                 )

    print(f'loading model from {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"]

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model = model.eval()

    patch_dim = [256, 256]
    step_size = [32, 32]
    n_classes = 2



    test_root = 'CHASE/only2ndLabel/Test/A/images'
    save_root = 'CHASE/only2ndLabel/Test/A/pred_unet'

    # test_root = 'CCO/Test/A/images'
    # save_root = 'CCO/Test/A/pred'

    if not os.path.exists(save_root):
        os.mkdir(save_root)

    file_names = [f for f in sorted(os.listdir(test_root)) if not f.startswith('.')]

    for f in file_names:

        image = np.asarray(Image.open(os.path.join(test_root, f)))

        image = (image - image.min())/(image.max() - image.min())

        recon = predict_3D_patches(model.A_Unet, patch_dim, image, step_size, n_classes,
                                   device=device)

        recon = np.argmax(recon, axis=0)

        # save_path = os.path.join(save_root, f)[:-3] + 'png'
        save_path = os.path.join(save_root, f)

        # io.imsave(save_path, recon>0)

        recon = Image.fromarray(recon>0)

        recon.save(save_path)


# from vtk import *
# import vtk
# from vtk.util.numpy_support import vtk_to_numpy
#
# # load a vtk file as input
# import vtk
# reader = vtk.vtkPolyDataReader()
# reader.SetFileName("/Users/px/ccolab/data/sphere/default-sphere.vtk")
# reader.Update()
# polydata = reader.GetOutput()
#
# polydata.GetCell(1)
#
# import numpy as np
# a = np.loadtxt("/Users/px/ccolab/data/circle/default-circle1.vtk",
#            # skiprows=5
#            )
# import pyvista
# a = np.hstack([a, np.zeros((len(a), 1))])
# pyvista.PolyData(a).save('sb.vtk')
#
# np.mean(a, axis=0)
# np.min(a, axis=0)
# np.max(a, axis=0)
#
# #Grab a scalar from the vtk file
# my_vtk_array = reader.GetOutput().GetPointData().GetArray("my_scalar_name")