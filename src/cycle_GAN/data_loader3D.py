import os

from src.data_loader_lightning import *
from Imports import *

def plot_samples(batch, model=None):
    if 'location' in batch.keys():
        del batch['location']

    img = batch['image'][tio.DATA]

    #
    # label = batch['label'][tio.DATA]
    # label = label.argmax(dim=1, keepdim=True).cpu()
    # batch['label']['data'] = label

    batch_subjects = tio.utils.get_subjects_from_batch(batch)

    if model is not None:
        pred = model(img)
        pred = pred.argmax(dim=1, keepdim=True).cpu()
        tio.utils.add_images_from_batch(batch_subjects, pred, tio.LabelMap)

    else:
        tio.utils.add_images_from_batch(batch_subjects, batch['image'][tio.DATA], tio.ScalarImage)

    for subject in batch_subjects:
        subject.plot()


class PatchDatasetDouble(pl.LightningDataModule):
    def __init__(self, dataset_dir, batch_size, num_workers=1, label_subdir='labels',
                 image_subdir='images', train_val_ratio=1, path_A=None, path_B = None,
                 n_class=-1, data_shape=None, patch_size=64, queue_length=200, samples_per_volume=10,
                 sampling_strategy='Uniform', label_probabilities={0: 1, 1: 1}, include_label=False
                ):
        super().__init__()

        self.train_val_ratio = train_val_ratio
        self.subjects_A = None
        self.subjects_B = None
        self.train_set_A = None
        self.train_set_B = None

        self.batch_size = batch_size
        self.dataset_dir = Path(dataset_dir)

        path_A = os.path.join(dataset_dir + 'A') if path_A is None else path_A
        path_B = os.path.join(dataset_dir + 'B') if path_B is None else path_B

        # self.path_A = Path(path_A)
        # self.path_B = Path(path_B)

        self.include_label = include_label

        self.path_A = path_A
        self.path_B = path_B

        self.subjects = None
        self.test_subjects = None
        self.preprocess = None
        self.transform = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.image_subdir = image_subdir
        self.label_subdir = label_subdir
        self.num_workers = num_workers
        self.n_class = n_class
        self.data_shape = data_shape
        self.patch_size = patch_size
        self.queue_length = queue_length
        self.samples_per_volume = samples_per_volume
        self.sampling_strategy = sampling_strategy
        self.label_probabilities = label_probabilities

        self.max_shape = None
        self.num_samples = None


    def get_max_shape(self, subjects):
        import numpy as np
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)

    def prepare_data(self):

        def get_niis(d):
            return sorted([os.path.join(d, p) for p in os.listdir(d) if not p.startswith('.') and '.nii' in p])

            # return sorted(p for p in d.glob('*.nii*') if not p.name.startswith('.'))

        subjects_names = ['subjects_A', 'subjects_B']

        for subjects_name, dataset_dir in zip(subjects_names, (self.path_A, self.path_B)):

            subjects = []
            image_training_paths = get_niis(os.path.join(dataset_dir, self.image_subdir))
            label_training_paths = get_niis(os.path.join(dataset_dir, self.label_subdir)) \
                if self.include_label else image_training_paths
            label_training_paths = image_training_paths if len(label_training_paths) == 0 else label_training_paths

            for image_path, label_path in zip(image_training_paths, label_training_paths):

                print(f'loading image {image_path}')

                if label_path != image_path:
                    subject = tio.Subject(
                        image=tio.ScalarImage(image_path),
                        label=tio.LabelMap(label_path)
                    )
                else:
                    subject = tio.Subject(
                        image=tio.ScalarImage(image_path),
                    )

                subjects.append(subject)

            setattr(self, subjects_name, subjects)

        # self.test_subjects = []

        # for image_path in image_test_paths:
        #     subject = tio.Subject(image=tio.ScalarImage(image_path))
        #     self.test_subjects.append(subject)

        self.max_shape = self.get_max_shape(self.subjects_A + self.subjects_B) if self.max_shape is None else self.max_shape

    def get_preprocessing_transform(self):
        data_shape = self.get_max_shape(
            self.subjects_A + self.subjects_B) if self.data_shape is None else self.data_shape
        preprocess = tio.Compose([
            tio.RescaleIntensity((-1, 1)),
            # tio.CropOrPad(data_shape),
            # tio.EnsureShapeMultiple(8),  # for the U-Net
            tio.OneHot(num_classes=self.n_class),
        ])
        return preprocess

    def get_augmentation_transform(self):
        augment = tio.Compose([
            tio.RandomAffine(),
            # tio.RandomGamma(p=0.5),
            # tio.RandomNoise(p=0.5),
            # tio.RandomMotion(p=0.1),
            # tio.RandomBiasField(p=0.25),
        ])
        return augment



    def setup(self, stage=None, label_probabilities=None):


        num_subjects = len(self.subjects_A)
        assert num_subjects > 0
        num_subjects = len(self.subjects_B)
        assert num_subjects > 0

        train_set_names = ['train_set_A', 'train_set_B']

        for train_set_name, train_subjects in zip(train_set_names, (self.subjects_A, self.subjects_B)):

            self.preprocess = self.get_preprocessing_transform()

            augment = self.get_augmentation_transform()
            self.transform = tio.Compose([self.preprocess,
                                          # augment
                                          ])

            train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)

            setattr(self, train_set_name, train_set)

        sampler = tio.data.UniformSampler(self.patch_size)

        if self.sampling_strategy == 'Label':
            sampler = tio.data.LabelSampler(patch_size=self.patch_size,
                                            label_probabilities=self.label_probabilities)



        if self.max_shape is not None:
            samples_per_volume = int(np.round(np.prod(self.max_shape/self.patch_size)))
            queue_length = samples_per_volume

            if self.samples_per_volume is not None:
                self.samples_per_volume = min(self.samples_per_volume, samples_per_volume)
                self.queue_length = min(self.samples_per_volume, queue_length)
            else:
                self.samples_per_volume = samples_per_volume
                self.queue_length = queue_length
                
        print(f'samples_per_volume={self.samples_per_volume}')

        if self.queue_length < self.samples_per_volume:
            self.samples_per_volume = self.queue_length


        queueA = tio.Queue(
            getattr(self, 'train_set_A'),
            self.queue_length,
            self.samples_per_volume,
            sampler,
            # num_workers=self.num_workers,
        )

        queue_B = tio.Queue(
            getattr(self, 'train_set_B'),
            self.queue_length,
            self.samples_per_volume,
            sampler,
            # num_workers=self.num_workers,
        )

        train_final = CustomDataset(queueA, queue_B)

        self.train_set = train_final

        self.val_set = self.train_set

        self.test_set = self.train_set


    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, num_workers=self.num_workers)



class CustomDataset(Dataset):

    def __init__(self, q1, q2):
        super().__init__()
        self.cur_queue1 = q1
        self.cur_queue2 = q2

        self.queue1_len = q1.iterations_per_epoch
        self.queue2_len = q2.iterations_per_epoch

    def __len__(self):
        return max(self.queue1_len, self.queue2_len)

    def __getitem__(self, idx):

        return {'A': self.cur_queue1[idx % self.queue1_len],
                'B': self.cur_queue2[idx % self.queue2_len]}
