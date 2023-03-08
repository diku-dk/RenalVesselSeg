import os

import matplotlib.pyplot as plt
import torch

from Imports import *
from PIL import Image
warnings.simplefilter("ignore")

# import shutil
# root = '/Volumes/T7/DRIVE/Train/A/images'
# for i in os.listdir(root):
#     if i.startswith('.'): continue
#     name_new = i[:2] + '.tif'
#     shutil.move(os.path.join(root, i), os.path.join(root, name_new))


# a = '/Users/px/GoogleDrive/UNetMonai/src/cycle_GAN/DRIVE/Train/B/orig_images/22.tif'
# plt.imshow(plt.imread(a)[..., 1])
# plt.show()


class Resize(object):

    def __init__(self, image_size: (int, tuple) = 256):

        """
        Parameters:
            image_size: Final size of the image
        """

        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        elif isinstance(image_size, tuple):
            self.image_size = image_size
        else:
            raise ValueError("Unknown DataType of the parameter image_size found!!")

    def __call__(self, sample):

        """
        Parameters:
            sample: Dictionary containing image and label
        """
        for key in list(sample.keys()):

            val = sample[key]
            val = tfm.resize(val, output_shape=self.image_size)
            val = np.clip(val, a_min=0., a_max=1.)
            sample[key] = val

        return sample


class RandomCrop(object):

    def __init__(self, image_size: (int, tuple) = 256):

        """
        Parameters:
            image_size: Final size of the image (should be smaller than current size o/w returns the original image)
        """

        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        elif isinstance(image_size, tuple):
            self.image_size = image_size
        else:
            raise ValueError("Unknown DataType of the parameter image_size found!!")

    def __call__(self, sample):

        """
        Parameters:
            sample: Dictionary containing image and label
        """
        top, lft = 0, 0
        for key in sorted(list(sample.keys())):
            val = sample[key]

            curr_height, curr_width = val.shape[0], val.shape[1]

            ht_diff = max(0, curr_height - self.image_size[0])
            wd_diff = max(0, curr_width - self.image_size[1])

            if 'label' not in key.lower():
                top = np.random.randint(low=0, high=ht_diff)
                lft = np.random.randint(low=0, high=wd_diff)

            val = val[top: top + self.image_size[0], lft: lft + self.image_size[1]]

            sample[key] = val

        return sample


class Random_Flip(object):

    def __call__(self, sample):
        """
        Parameters:
            sample: Dictionary containing image and label
        """

        if np.random.uniform(low=0., high=1.0) > .5:
            for key in list(sample.keys()):
                val = sample[key]
                val = np.fliplr(val)
                sample[key] = val
            return sample

        else:
            return sample



class To_Tensor(object):

    def __call__(self, sample):
        """
        Parameters:
            sample: Dictionary containing image and label
        """
        for key in list(sample.keys()):
            A = sample[key]
            # A = np.transpose(A.astype(np.float, copy=True), (2, 0, 1))
            if 'label' not in key.lower():
                A = np.transpose(A.astype(np.float, copy=True), (2, 0, 1))
                A = torch.tensor(A, dtype=torch.float)

            else:
                A = np.transpose(A.astype(np.uint8, copy=True), (2, 0, 1))
                A = torch.tensor(A, dtype=torch.int64)

            sample[key] = A

        return sample


class Normalize(object):

    def __init__(self, mean=[0.5] * 3, stdv=[0.5] * 3):
        """
        Parameters:
            mean: Normalizing mean
            stdv: Normalizing stdv
        """

        mean = torch.tensor(mean, dtype=torch.float)
        stdv = torch.tensor(stdv, dtype=torch.float)
        self.transforms = T.Normalize(mean=mean, std=stdv)

    def __call__(self, sample):
        """
        Parameters:
            sample: Dictionary containing image and label
        """

        return sample

        for key in list(sample.keys()):

            if 'label' in key.lower():
                continue

            A = sample[key]

            # mean, std = A.mean((1, 2)), A.std((1, 2))

            A = self.transforms(A)
            sample[key] = A

        return sample


class ToGray(object):

    def __init__(self, ):
        """
        Parameters:
            mean: Normalizing mean
            stdv: Normalizing stdv
        """

        self.transforms = T.Grayscale()

    def __call__(self, sample):
        """
        Parameters:
            sample: Dictionary containing image and label
        """

        # return sample

        for key in list(sample.keys()):

            if 'label' in key.lower():
                continue

            A = sample[key]
            A = self.transforms(A)
            sample[key] = A

        return sample


class ToGreen(object):

    def __init__(self, ):
        pass

    def __call__(self, sample):
        """
        Parameters:
            sample: Dictionary containing image and label
        """

        # return sample

        for key in list(sample.keys()):

            if 'label' in key.lower():
                continue

            A = sample[key]
            A = A[1:2]
            sample[key] = A

        return sample


class CustomDataset(Dataset):

    def __init__(self, path: str = None, transforms=None, crop_size=None):
        """
        Parameters:
            transforms: a list of Transformations (Data augmentation)
        """

        super().__init__()
        self.transforms = T.Compose(transforms)

        # file_names_A = sorted(os.listdir(path + 'A/'), key=lambda x: int(x[: -4]))
        file_names_A = sorted(os.listdir(path + 'A/'))

        self.file_names_A = [path + 'A/' + file_name for file_name in file_names_A]

        # file_names_B = sorted(os.listdir(path + 'B/'), key=lambda x: int(x[: -4]))
        file_names_B = sorted(os.listdir(path + 'B/'))

        self.file_names_B = [path + 'B/' + file_name for file_name in file_names_B]

        # self.file_names_A = self.file_names_A[:max_sz]
        # self.file_names_B = self.file_names_B[:max_sz]


        if crop_size is None:
            total = max(len(self.file_names_A), len(self.file_names_B))
        else:
            max_dim = self.get_max_dim()
            total = np.prod(max_dim/crop_size)
            total = int(total * max(len(self.file_names_A), len(self.file_names_B)))

        self.total = total

    def __len__(self):
        return self.total

    def get_max_dim(self):

        max_a = np.max(np.array([Image.open(i).size[:2] for i in self.file_names_A]), axis=0)
        max_b = np.max(np.array([Image.open(i).size[:2] for i in self.file_names_B]), axis=0)

        avg = (max_a + max_b)/2

        return avg

    def __getitem__(self, idx):
        A = io.imread(self.file_names_A[idx % len(self.file_names_A)])
        B = io.imread(self.file_names_B[idx % len(self.file_names_B)])

        if len(A.shape) == 2:
            A = np.expand_dims(A, axis=-1)

        if len(B.shape) == 2:
            B = np.expand_dims(B, axis=-1)

        sample = self.transforms({'A': A, 'B': B})

        return sample


class CustomDatasetWithLabel(Dataset):
    """
        path B is with labels
    """

    def __init__(self, path: str = None, transforms=None, crop_size=None):
        """
        Parameters:
            transforms: a list of Transformations (Data augmentation)
        """
        super().__init__()
        self.transforms = T.Compose(transforms)

        file_names_A = [f for f in sorted(os.listdir(path + 'A/images/')) if not f.startswith('.')]

        self.file_names_A = [path + 'A/images/' + file_name for file_name in file_names_A]
        self.file_names_A_labels = [path + 'A/labels/' + file_name for file_name in file_names_A]

        # file_names_B = sorted(os.listdir(path + 'B/'), key=lambda x: int(x[: -4]))
        file_names_B = [f for f in sorted(os.listdir(path + 'B/images/')) if not f.startswith('.')]

        self.file_names_B = [path + 'B/images/' + file_name for file_name in file_names_B]
        self.file_names_B_labels = [path + 'B/labels/' + file_name for file_name in file_names_B]


        if crop_size is None:
            total = max(len(self.file_names_A), len(self.file_names_B))
            total = len(self.file_names_A) * len(self.file_names_B)

        else:
            max_dim = self.get_max_dim()
            total = np.prod(max_dim/crop_size)
            # total = int(total * max(len(self.file_names_A), len(self.file_names_B)))

            total = int(total * len(self.file_names_A) * len(self.file_names_B))
            
        print(f'total images = {total}')


        self.total = total

    def __len__(self):
        return self.total

    def get_max_dim(self):

        max_a = np.max(np.array([Image.open(i).size[:2] for i in self.file_names_A]), axis=0)
        max_b = np.max(np.array([Image.open(i).size[:2] for i in self.file_names_B]), axis=0)

        avg = (max_a + max_b)/2

        return avg


    def __getitem__(self, idx):

        # A = io.imread(self.file_names_A[idx % len(self.file_names_A)])
        # B = io.imread(self.file_names_B[idx % len(self.file_names_B)])
        # BLabel = io.imread(self.file_names_B_labels[idx % len(self.file_names_B)])
        a_idx = idx % len(self.file_names_A)
        b_idx = idx % len(self.file_names_B)
        a_idx = np.random.randint(len(self.file_names_A))
        b_idx = np.random.randint(len(self.file_names_B))
        A = np.asarray(Image.open(self.file_names_A[a_idx]))
        B = np.asarray(Image.open(self.file_names_B[b_idx]))
        BLabel = np.asarray(Image.open(self.file_names_B_labels[b_idx]))

        # A = (A - A.min())/A.max()
        # B = (B - B.min())/B.max()
        # A = 2 * A - 1
        # B = 2 * B - 1

        A = (A - A.min())/(A.max() - A.min())
        B = (B - B.min())/(B.max() - B.min())


        BLabel = (BLabel>0).astype(np.uint8)

        if len(A.shape) == 2:
            A = np.expand_dims(A, axis=-1)

        if len(B.shape) == 2:
            B = np.expand_dims(B, axis=-1)

        if len(BLabel.shape) == 2:
            BLabel = np.expand_dims(BLabel, axis=-1)

        sample = self.transforms({'A': A, 'B': B, 'BLabel': BLabel})

        return sample



class CustomDatasetWithLabelOnlyB(Dataset):
    """
        path B is with labels
    """

    def __init__(self, path: str = None, transforms=None, crop_size=None):
        """
        Parameters:
            transforms: a list of Transformations (Data augmentation)
        """
        super().__init__()
        self.transforms = T.Compose(transforms)
        file_names_B = [f for f in sorted(os.listdir(path + 'B/images/')) if not f.startswith('.')]
        self.file_names_B = [path + 'B/images/' + file_name for file_name in file_names_B]
        self.file_names_B_labels = [path + 'B/labels/' + file_name for file_name in file_names_B]


        if crop_size is None:
            total = len(self.file_names_B)

        else:
            max_dim = self.get_max_dim()
            total = np.prod(max_dim/crop_size)
            total = int(total * len(self.file_names_B))

            # total = int(total * len(self.file_names_B))//10

        print(f'total images = {total}')

        self.total = total

    def __len__(self):
        return self.total

    def get_max_dim(self):

        max_b = np.max(np.array([Image.open(i).size[:2] for i in self.file_names_B]), axis=0)

        return max_b

    def __getitem__(self, idx):

        # A = io.imread(self.file_names_A[idx % len(self.file_names_A)])
        # B = io.imread(self.file_names_B[idx % len(self.file_names_B)])
        # BLabel = io.imread(self.file_names_B_labels[idx % len(self.file_names_B)])
        b_idx = idx % len(self.file_names_B)
        b_idx = np.random.randint(len(self.file_names_B))
        B = np.asarray(Image.open(self.file_names_B[b_idx]))
        BLabel = np.asarray(Image.open(self.file_names_B_labels[b_idx]))

        # A = (A - A.min())/A.max()
        # B = (B - B.min())/B.max()
        # A = 2 * A - 1
        # B = 2 * B - 1

        B = (B - B.min())/(B.max() - B.min())

        BLabel = (BLabel>0).astype(np.uint8)

        if len(B.shape) == 2:
            B = np.expand_dims(B, axis=-1)

        if len(BLabel.shape) == 2:
            BLabel = np.expand_dims(BLabel, axis=-1)

        sample = self.transforms({'B': B, 'BLabel': BLabel})

        return sample

class DataModule(pl.LightningDataModule):
    """
    Implements the Lightining DataModule!
    """

    def __init__(self, url: str, root_dir: str = "./Dataset/CycleGAN/", img_sz: int = 256, trn_batch_sz: int = 4,
                 tst_batch_sz: int = 64, num_workers: int = 0, in_channels=3):

        """
        Parameters:
            url:          Download URL of the dataset
            root_dir:     Root dir where dataset needs to be downloaded
            img_sz:       Size of the Image
            trn_batch_sz: Training Batch Size
            tst_batch_sz: Test Batch Size
        """

        super().__init__()
        self.num_workers = num_workers
        self.url = url
        self.dataset = url.split("/")[-1]

        self.processed_dir = root_dir + "Processed/"
        self.compressed_dir = root_dir + "Compressed/"
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.compressed_dir, exist_ok=True)

        self.trn_batch_sz = trn_batch_sz
        self.tst_batch_sz = tst_batch_sz

        jitter_sz = int(img_sz * 1.120)


        mean = [0.5] * in_channels
        stdv = [0.5] * in_channels

        self.tst_tfms = [Resize(img_sz), To_Tensor(), Normalize(mean, stdv)]
        self.trn_tfms = [Resize(jitter_sz), RandomCrop(img_sz), Random_Flip(), To_Tensor(), Normalize(mean, stdv)]

    def prepare_data(self):

        if self.dataset in os.listdir(self.compressed_dir):
            print(f"Dataset {self.dataset[:-4]} already exists!")
        else:
            print(f"Downloading dataset {self.dataset[:-4]}!!")
            wget.download(self.url, self.compressed_dir)
            print(f"\nDataset {self.dataset[:-4]} downloaded. Extraction in progress!")

            with zipfile.ZipFile(self.compressed_dir + self.dataset, 'r') as zip_ref:
                zip_ref.extractall(self.processed_dir)
            print(f"Extraction done!")

            # you might need to modify the below code; it's not generic, but works for most of the datasets listed in that url.
            dwnld_dir = self.processed_dir + self.dataset[:-4] + "/"
            for folder in ["testA/", "testB/", "trainA/", "trainB/"]:

                dest_dir = dwnld_dir
                src_dir = dwnld_dir + folder

                dest_dir = dest_dir + "Train/" if folder[:-2] != "test" else dest_dir + "Test/"
                dest_dir = dest_dir + "B/" if folder[-2] != "A" else dest_dir + "A/"
                os.makedirs(dest_dir, exist_ok=True)

                orig_files = [src_dir + file for file in os.listdir(src_dir)]
                modf_files = [dest_dir + "{:06d}.jpg".format(i) for i, file in enumerate(orig_files)]

                for orig_file, modf_file in zip(orig_files, modf_files):
                    shutil.move(orig_file, modf_file)
                os.rmdir(src_dir)

            print(f"Files moved to appropiate folder!")

    def setup(self, stage: str = None):

        """
        stage: fit/test
        """

        dwnld_dir = self.processed_dir + self.dataset[:-4]
        trn_dir = dwnld_dir + "/Train/"
        tst_dir = dwnld_dir + "/Test/"

        if stage == 'fit' or stage is None:
            dataset = CustomDataset(path=trn_dir, transforms=self.trn_tfms)
            train_sz = int(len(dataset) * 0.9)
            valid_sz = len(dataset) - train_sz

            self.train, self.valid = random_split(dataset, [train_sz, valid_sz])
            print(f"Size of the training dataset: {train_sz}, validation dataset: {valid_sz}")

        if stage == 'test' or stage is None:
            self.test = CustomDataset(path=tst_dir, transforms=self.tst_tfms)
            print(f"Size of the test dataset: {len(self.test)}")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.trn_batch_sz, shuffle=True, num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.tst_batch_sz, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.tst_batch_sz, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)



class DataModule2D(pl.LightningDataModule):
    """
    Implements the Lightining DataModule!
    """

    def __init__(self, root_dir: str = "./Dataset/CycleGAN/", img_sz: int = 256, trn_batch_sz: int = 4,
                 tst_batch_sz: int = 64, num_workers: int = 0, in_channels=3, train_portion=0.9):

        super().__init__()
        self.num_workers = num_workers

        self.processed_dir = root_dir
        os.makedirs(self.processed_dir, exist_ok=True)

        self.trn_batch_sz = trn_batch_sz
        self.tst_batch_sz = tst_batch_sz

        # jitter_sz = int(img_sz * 1.120)

        self.crop_size = img_sz

        mean = [0.5] * in_channels
        stdv = [0.2] * in_channels

        self.tst_tfms = [RandomCrop(img_sz), To_Tensor(),
                         # ToGreen(),
                         # Normalize(mean, stdv)
                         ]
        self.trn_tfms = [RandomCrop(img_sz),
                         Random_Flip(), To_Tensor(),
                         # ToGreen(),
                         # Normalize(mean, stdv)
                         ]
        self.train_portion = train_portion

    def prepare_data(self):
        return

    def setup(self, stage: str = None):

        """
        stage: fit/test
        """

        dwnld_dir = self.processed_dir
        trn_dir = dwnld_dir + "/Train/"
        tst_dir = dwnld_dir + "/Train/"

        if stage == 'fit' or stage is None:
            dataset = CustomDatasetWithLabel(path=trn_dir, transforms=self.trn_tfms, crop_size=self.crop_size)
            train_sz = int(len(dataset) * self.train_portion)
            valid_sz = len(dataset) - train_sz

            self.train, self.valid = random_split(dataset, [train_sz, valid_sz])
            print(f"Size of the training dataset: {train_sz}, validation dataset: {valid_sz}")

        if stage == 'test' or stage is None:
            self.test = CustomDatasetWithLabel(path=tst_dir, transforms=self.tst_tfms, crop_size=self.crop_size)
            print(f"Size of the test dataset: {len(self.test)}")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.trn_batch_sz, shuffle=True, num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.tst_batch_sz, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.tst_batch_sz, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)



class DataModule2DSingle(DataModule2D):
    """
    Implements the Lightining DataModule!
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self, stage: str = None):

        """
        stage: fit/test
        """

        dwnld_dir = self.processed_dir
        trn_dir = dwnld_dir + "/Train/"
        tst_dir = dwnld_dir + "/Train/"

        if stage == 'fit' or stage is None:
            dataset = CustomDatasetWithLabelOnlyB(path=trn_dir, transforms=self.trn_tfms, crop_size=self.crop_size)
            train_sz = int(len(dataset) * self.train_portion)
            valid_sz = len(dataset) - train_sz

            self.train, self.valid = random_split(dataset, [train_sz, valid_sz])
            print(f"Size of the training dataset: {train_sz}, validation dataset: {valid_sz}")

        if stage == 'test' or stage is None:
            self.test = CustomDatasetWithLabelOnlyB(path=tst_dir, transforms=self.tst_tfms, crop_size=self.crop_size)
            print(f"Size of the test dataset: {len(self.test)}")



class DataModule2DSingleNoCrop(DataModule2D):
    """
    Implements the Lightining DataModule!
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self, stage: str = None):

        """
        stage: fit/test
        """

        dwnld_dir = self.processed_dir
        trn_dir = dwnld_dir + "/Train/"
        tst_dir = dwnld_dir + "/Train/"

        self.tst_tfms = [To_Tensor(),
                         # ToGreen(),
                         # Normalize(mean, stdv)
                         ]
        self.trn_tfms = [
                         Random_Flip(),
                         To_Tensor(),
                         # ToGreen(),
                         # Normalize(mean, stdv)
                         ]

        if stage == 'fit' or stage is None:
            dataset = CustomDatasetWithLabelOnlyB(path=trn_dir, transforms=self.trn_tfms, crop_size=self.crop_size)
            train_sz = int(len(dataset) * self.train_portion)
            valid_sz = len(dataset) - train_sz

            self.train, self.valid = random_split(dataset, [train_sz, valid_sz])
            print(f"Size of the training dataset: {train_sz}, validation dataset: {valid_sz}")

        if stage == 'test' or stage is None:
            self.test = CustomDatasetWithLabelOnlyB(path=tst_dir, transforms=self.tst_tfms, crop_size=self.crop_size)
            print(f"Size of the test dataset: {len(self.test)}")



def show_image(image):
    plt.imshow(np.transpose(image, (1, 2, 0)),
               cmap='gray'
               )


def get_random_sample(dataset):
    return dataset[np.random.randint(0, len(dataset))]

