import numpy as np
import torch.nn as nn

def softmax(x, axis=0):
    max = np.max(x, axis=axis, keepdims=True)  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(e_x, axis=axis, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x

def pred_to_class(tensor, img_dims=3, threshold=0.5, has_batch_dim=False):
    tensor_dim = img_dims + int(has_batch_dim)
    dims = len(tensor.shape)
    if dims == tensor_dim:
        # Check if already integer targets
        if np.issubdtype(tensor.dtype, np.integer):
            return tensor
        else:
            return tensor >= threshold

    elif tensor.shape[0] == 1:
        if np.issubdtype(tensor.dtype, np.integer):
            # Squeeze last axis
            return np.squeeze(tensor)
        else:
            return tensor >= threshold
    else:
        # Convert predicted probabilities to predicted class
        return tensor.argmax(0).astype(np.uint8)


def reshape_add_axis(X, im_dims=2, n_channels=1):
    X = np.asarray(X)
    if X.shape[0] != n_channels:
        # Reshape
        X = X.reshape((n_channels,) + X.shape)
    if len(X.shape) == im_dims + 1:
        X = X.reshape((1,) + X.shape)
    return X


def mgrid_to_points(mgrid):
    """
    Takes a NxD1xD2xD3 meshgrid or tuple(meshgrid) and outputs a D1*D2*D3xN
    matrix of coordinate points
    """
    points = np.empty(shape=(np.prod(mgrid[0].shape), len(mgrid)),
                      dtype=mgrid[0].dtype)
    for i in range(len(mgrid)):
        points[:, i] = mgrid[i].ravel()
    return points


def standardize_strides(strides):
    if isinstance(strides, list):
        return tuple(strides)
    elif isinstance(strides, tuple):
        return strides
    else:
        return 3 * (int(strides),)


def get_latest_state(log_root='lightning_logs', v_num=None):

    import os

    ckpt_folders = os.listdir(log_root)
    ckpt_folders = [i for i in ckpt_folders if i.startswith('v')]
    ckpt_folders = sorted(ckpt_folders)
    v_num = ckpt_folders[-1] if v_num is None else f'version_{v_num}'

    ckpt_folder = os.path.join(*[log_root, v_num, 'checkpoints'])
    ckpt_name = os.listdir(ckpt_folder)

    ckpt_path = None
    for i in ckpt_name:
        if i.startswith('.'):
            continue
        else:
            ckpt_path = os.path.join(ckpt_folder, i)

    return ckpt_path


class Initializer:

    def __init__(self, init_type: str = 'normal', init_gain: float = 0.02):

        """
        Initializes the weight of the network!

        Parameters:
            init_type: Initializer type - 'kaiming' or 'xavier' or 'normal'
            init_gain: Standard deviation of the normal distribution
        """

        self.init_type = init_type
        self.init_gain = init_gain

    def init_module(self, m):

        cls_name = m.__class__.__name__
        if hasattr(m, 'weight') and (cls_name.find('Conv') != -1 or cls_name.find('Linear') != -1):

            if self.init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif self.init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=self.init_gain)
            elif self.init_type == 'normal':
                nn.init.normal_(m.weight.data, mean=0, std=self.init_gain)
            else:
                raise ValueError('Initialization not found!!')

            if m.bias is not None: nn.init.constant_(m.bias.data, val=0)

        if hasattr(m, 'weight') and cls_name.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, mean=1.0, std=self.init_gain)
            nn.init.constant_(m.bias.data, val=0)

    def __call__(self, net):

        """
        Parameters:
            net: Network
        """

        net.apply(self.init_module)

        return net
