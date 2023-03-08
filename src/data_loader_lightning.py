import time
from itertools import islice
from pathlib import Path

import gdown
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterDataPipe
from torch.utils.data import random_split
from torch.utils.data.datapipes.iter import Shuffler, UnBatcher

import numpy as np
"""Module for custom transforms."""
from typing import Tuple

import torchio as tio


class CustomQueue(tio.Queue):
    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, _):
        sample_patch = super().__getitem__(_)

        augment = tio.Compose([
            # tio.RandomAffine(scales=1, degrees=(90, 90), translation=0, p=1),
            tio.RandomAffine(scales=(1, 1), degrees=(90, 90, 0, 0, 0, 0), translation=0, p=0.6),
            tio.RandomAffine(scales=(1, 1), degrees=(0, 0, 90, 90, 0, 0), translation=0, p=0.6),
            tio.RandomAffine(scales=(1, 1), degrees=(0, 0, 0, 0, 90, 90), translation=0, p=0.6),
            # tio.CropOrPad(64),

            # tio.RandomNoise(p=0.5),
        ])

        sample_patch = augment(sample_patch)

        return sample_patch

#
# class CustomLabelSampler(tio.data.LabelSampler):
#     def __int__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def __call__(self, *args, **kwargs):
#         res = super().__call__(*args, **kwargs)
#         # return res
#
#         augment = tio.Compose([
#             tio.RandomAffine(scales=1, degrees=60, translation=0, p=0.8),
#             # tio.RandomFlip(axes=2, p=0.3),
#             # tio.RandomNoise(p=0.5),
#         ])
#
#         res = augment(res)
#         return res

class PatchesSampler(IterDataPipe):

    def __init__(self, datapipe, sampler, samples_per_volume, augment=False):
        self.datapipe = datapipe
        self.sampler = sampler
        self.samples_per_volume = samples_per_volume
        self.augment = augment

    def __iter__(self):
        for subject in self.datapipe:
            iterable = self.sampler(subject)
            batch_subjects = list(islice(iterable,
                              self.samples_per_volume))

            if not self.augment:
                yield batch_subjects

            else:
                augment = tio.Compose([
                    # tio.RandomAffine(scales=(1, 1), degrees=90, translation=0, p=1),
                    tio.RandomAffine(scales=(1, 1), degrees=(90, 90, 0, 0, 0, 0), translation=0, p=0.5),
                    tio.RandomAffine(scales=(1, 1), degrees=(0, 0, 90, 90, 0, 0), translation=0, p=0.5),
                    tio.RandomAffine(scales=(1, 1), degrees=(0, 0, 0, 0, 90, 90), translation=0, p=0.5),
                    # tio.CropOrPad(64),
                ])

                # affine = np.array([[ 0.0226    ,  0.        ,  0.        ,  0],
                #                    [ 0.        ,  0.0226    ,  0.        ,  0],
                #                    [ 0.        ,  0.        ,  0.0226    , 0],
                #                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                #
                # for i, im in enumerate(batch_subjects):
                #     im['image'].affine = affine
                #     im['label'].affine = affine
                #     im['weight'].affine = affine

                transformed_batch_subjects = [augment(batch_sub) for batch_sub in batch_subjects]

                yield transformed_batch_subjects

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir, batch_size, train_val_ratio=1,
                 val_dir=None, num_workers=0, label_subdir='labels',
                 image_subdir='images',
                 n_class=-1, data_shape=None, test_dir=None, weight_subdir=None, augment=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_dir = Path(dataset_dir)
        self.train_val_ratio = train_val_ratio
        self.subjects = None
        self.test_subjects = None
        self.preprocess = None
        self.transform = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.image_subdir = image_subdir
        self.label_subdir = label_subdir
        self.val_dir = val_dir
        self.num_workers = num_workers
        self.n_class = n_class
        self.weight_subdir = weight_subdir
        self.test_dir = test_dir
        self.augment = augment

        self.max_shape = None
        self.num_samples = None

    def get_max_shape(self, subjects):
        import numpy as np
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)

    def prepare_data(self):
        def get_niis(d):
            return sorted(p for p in d.glob('*.nii*') if not p.name.startswith('.'))

        image_training_paths = get_niis(self.dataset_dir / self.image_subdir)
        label_training_paths = get_niis(self.dataset_dir / self.label_subdir) if self.label_subdir is not None \
            else image_training_paths

        weight_training_paths = get_niis(self.dataset_dir / self.weight_subdir) if self.weight_subdir is not None \
            else image_training_paths

        image_test_paths = image_training_paths if self.test_dir is None else get_niis(Path(self.test_dir) / self.image_subdir)

        self.subjects = []
        for image_path, label_path, weight_path in zip(image_training_paths, label_training_paths, weight_training_paths):
            # 'image' and 'label' are arbitrary names for the images

            print(f'loading image {image_path}')

            kwargs = {'image': tio.ScalarImage(image_path)}
            if self.label_subdir is not None:
                kwargs['label'] = tio.LabelMap(label_path)

            if self.weight_subdir is not None:
                kwargs['weight'] = tio.LabelMap(weight_path)

            subject = tio.Subject(**kwargs)


            self.subjects.append(subject)

        self.test_subjects = []

        for image_path in image_test_paths:
            subject = tio.Subject(image=tio.ScalarImage(image_path))
            self.test_subjects.append(subject)

        self.max_shape = self.get_max_shape(self.subjects + self.test_subjects) if self.max_shape is None else self.max_shape

    def get_preprocessing_transform(self):

        transforms = [
            # tio.CropOrPad(64, mask_name='label'),
            tio.RescaleIntensity((-1, 1)),
            # tio.Resample(4),
            # tio.CropOrPad(data_shape),
            # tio.CropOrPad(64),
            # tio.EnsureShapeMultiple(8),  # for the U-Net
        ]

        if self.label_subdir is not None:
            transforms.append(tio.OneHot(num_classes=-1,
                                         # label_keys='label'
                                         ))

        preprocess = tio.Compose(transforms)
        return preprocess

    def get_augmentation_transform(self):
        augment = tio.Compose([
            # tio.RandomAffine(scales=1, degrees=45, translation=0, p=0.5),
            tio.RandomFlip(axes=2, p=0.3),
            # tio.RandomElasticDeformation(num_control_points=5, max_displacement=1.5, p=0.2)
            # tio.RandomGamma(p=0.5),
            # tio.RandomNoise(p=0.5),
            # tio.RandomMotion(p=0.1),
            # tio.RandomBiasField(p=0.25),
        ])
        return augment

    def setup(self, stage=None):

        num_subjects = len(self.subjects)
        assert num_subjects > 0

        if self.train_val_ratio < 1 and self.train_val_ratio > 0:

            num_train_subjects = int(round(num_subjects * self.train_val_ratio))
            num_val_subjects = num_subjects - num_train_subjects
            splits = num_train_subjects, num_val_subjects
            train_subjects, val_subjects = random_split(self.subjects, splits)

        else:
            train_subjects, val_subjects = self.subjects, self.subjects


        self.preprocess = self.get_preprocessing_transform()

        if self.augment:
            augment = self.get_augmentation_transform()

            self.transform = tio.Compose([self.preprocess,
                                          augment
                                          ])
        else:
            self.transform = tio.Compose([self.preprocess, ])



        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)

        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.transform)

        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, num_workers=self.num_workers)

    # @property
    # def num_training_steps(self) -> int:
    #     """Total training steps inferred from datamodule and devices."""
    #     if self.trainer.max_steps:
    #         return self.trainer.max_steps
    #
    #     limit_batches = self.trainer.limit_train_batches
    #     batches = len(self.train_dataloader())
    #     batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)
    #
    #     num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
    #     if self.trainer.tpu_cores:
    #         num_devices = max(num_devices, self.trainer.tpu_cores)
    #
    #     effective_accum = self.trainer.accumulate_grad_batches * num_devices
    #     return (batches // effective_accum) * self.trainer.max_epochs

class MedicalDecathlonDataModule(pl.LightningDataModule):
    def __init__(self, task, google_id, batch_size, train_val_ratio):
        super().__init__()
        self.task = task
        self.google_id = google_id
        self.batch_size = batch_size
        self.dataset_dir = Path(task)
        self.train_val_ratio = train_val_ratio
        self.subjects = None
        self.test_subjects = None
        self.preprocess = None
        self.transform = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def get_max_shape(self, subjects):
        import numpy as np
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)

    def download_data(self):
        if not self.dataset_dir.is_dir():
            url = f'https://drive.google.com/uc?id={self.google_id}'
            output = f'{self.task}.tar'
            gdown.download(url, output, quiet=False)

            import tarfile
            tf = tarfile.open(output)
            tf.extractall()

        def get_niis(d):
            return sorted(p for p in d.glob('*.nii*') if not p.name.startswith('.'))

        image_training_paths = get_niis(self.dataset_dir / 'imagesTr')
        label_training_paths = get_niis(self.dataset_dir / 'labelsTr')
        image_test_paths = get_niis(self.dataset_dir / 'imagesTs')
        return image_training_paths, label_training_paths, image_test_paths

    def prepare_data(self):
        image_training_paths, label_training_paths, image_test_paths = self.download_data()

        self.subjects = []
        for image_path, label_path in zip(image_training_paths, label_training_paths):
            # 'image' and 'label' are arbitrary names for the images
            subject = tio.Subject(
                image=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path)
            )
            self.subjects.append(subject)

        self.test_subjects = []
        for image_path in image_test_paths:
            subject = tio.Subject(image=tio.ScalarImage(image_path))
            self.test_subjects.append(subject)


    def get_preprocessing_transform(self):
        preprocess = tio.Compose([
            tio.RescaleIntensity((-1, 1)),
#            tio.CropOrPad(self.get_max_shape(self.subjects + self.test_subjects)),
            tio.CropOrPad(20),

            tio.EnsureShapeMultiple(8),  # for the U-Net
            tio.OneHot(),
        ])
        return preprocess

    def get_augmentation_transform(self):
        augment = tio.Compose([
            tio.RandomAffine(),
            tio.RandomGamma(p=0.5),
            tio.RandomNoise(p=0.5),
            tio.RandomMotion(p=0.1),
            tio.RandomBiasField(p=0.25),
        ])
        return augment

    def setup(self, stage=None):
        num_subjects = len(self.subjects)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects
        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = random_split(self.subjects, splits)

        self.preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        self.transform = tio.Compose([self.preprocess, augment])

        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preprocess)
        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=self.preprocess)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size)

    def inference(self, patch_overlap=1, subjects=None, model=None,
                  patch_size=None, batch_size=4):


        # subjects = self.test_dataloader() if subjects is None else subjects

        subjects = self.test_set if subjects is None else subjects

        for i, cur_subj in enumerate(subjects):

            cur_name = cur_subj['image']['stem']
            model = model.eval()

            with torch.no_grad():

                input_tensor = cur_subj['image'][tio.DATA]
                input_tensor = torch.unsqueeze(input_tensor, dim=0)
                logits = model.net(input_tensor)
                labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)

                labels = torch.squeeze(labels, dim=0)

                affine = cur_subj['image']['affine']
                # affine = torch.squeeze(affine)

                image = tio.LabelMap(tensor=labels, affine=affine)
                image.save(f'{cur_name}.nii.gz')


class PatchDatset(DataModule):
    def __init__(self, patch_size=64, queue_length=200, samples_per_volume=10,
                 sampling_strategy='Uniform', label_probabilities=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.queue_length = queue_length
        self.samples_per_volume = samples_per_volume
        self.sampling_strategy = sampling_strategy
        self.label_probabilities = label_probabilities

    def setup(self, stage=None, label_probabilities=None):

        super().setup(stage=stage)

        if label_probabilities is None:
            label_probabilities = {}
            for i in range(self.n_class):
                label_probabilities[i] = 1


        if self.max_shape is not None and self.samples_per_volume is None:
            self.samples_per_volume = int(np.round(np.prod(self.max_shape/self.patch_size)))
            self.queue_length = self.samples_per_volume

            print(f'samples_per_volume={self.samples_per_volume}')

        for set in ['train_set', 'val_set',
                    # 'test_set'
                    ]:
            if self.sampling_strategy == 'Uniform' or self.label_subdir is None:
                sampler = tio.data.UniformSampler(self.patch_size)

            else:
                sampler = tio.data.LabelSampler(patch_size=self.patch_size,
                                                label_probabilities=self.label_probabilities,
                                                label_name='label'
                                                )


            cur_queue = tio.Queue(
                                getattr(self, set),
                                self.queue_length,
                                self.samples_per_volume,
                                sampler,
                                num_workers=self.num_workers,
                            ) if not self.augment else \
                CustomQueue(getattr(self, set),
                            self.queue_length,
                            self.samples_per_volume,
                            sampler,
                            num_workers=self.num_workers,
                            )

            setattr(self, set, cur_queue)


    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, num_workers=0)






class PatchDatsetNew(DataModule):
    def __init__(self, patch_size=64, queue_length=200, samples_per_volume=10,
                 sampling_strategy='Uniform', label_probabilities=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.queue_length = queue_length
        self.samples_per_volume = samples_per_volume
        self.sampling_strategy = sampling_strategy
        self.label_probabilities = label_probabilities

    def setup(self, stage=None, label_probabilities=None):

        super().setup(stage=stage)

        if label_probabilities is None:
            label_probabilities = {}
            for i in range(self.n_class):
                label_probabilities[i] = 1


        for set in ['train_set', 'val_set',
                    # 'test_set'
                    ]:
            if self.sampling_strategy == 'Uniform' or self.label_subdir is None:
                sampler = tio.data.UniformSampler(patch_size=self.patch_size)

            else:
                # sampler = CustomLabelSampler(patch_size=self.patch_size,
                #                                 label_probabilities=self.label_probabilities,
                #                                 label_name='label')

                sampler = tio.data.LabelSampler(patch_size=self.patch_size,
                                                label_probabilities=self.label_probabilities,
                                                label_name='label')

            datapipe = PatchesSampler(getattr(self, set), sampler, self.samples_per_volume, augment=self.augment)
            datapipe = DataLoader(datapipe, batch_size=None, num_workers=self.num_workers)

            datapipe = UnBatcher(datapipe)
            datapipe = Shuffler(datapipe, buffer_size=self.batch_size * self.samples_per_volume)

            setattr(self, set, datapipe)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, num_workers=0)



class RandomCrop:
    """Random cropping on subject."""

    def __init__(self, roi_size: Tuple):
        """Init.

        Args:
            roi_size: cropping size.
        """
        self.sampler = tio.data.LabelSampler(patch_size=roi_size, label_name="label")

    def __call__(self, subject: tio.Subject) -> tio.Subject:
        """Use patch sampler to crop.

        Args:
            subject: subject having image and label.

        Returns:
            cropped subject
        """
        for patch in self.sampler(subject=subject, num_patches=1):
            return patch


class PatchNoQueue(DataModule):
    def __init__(self, patch_size=64, queue_length=200, samples_per_volume=10,
                 sampling_strategy='Uniform', label_probabilities=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.queue_length = queue_length
        self.samples_per_volume = samples_per_volume
        self.sampling_strategy = sampling_strategy
        self.label_probabilities = label_probabilities

    def setup(self, stage=None, label_probabilities=None):

        super().setup(stage=stage)

        if label_probabilities is None:
            label_probabilities = {}
            for i in range(self.n_class):
                label_probabilities[i] = 1

        for set in ['train_set', 'val_set',
                    # 'test_set'
                    ]:
            if self.sampling_strategy == 'Uniform' or self.label_subdir is None:
                sampler = tio.data.UniformSampler(self.patch_size)

            else:
                sampler = tio.data.LabelSampler(patch_size=self.patch_size,
                                                label_probabilities=self.label_probabilities,
                                                label_name='label')

            roll = yaw = pitch = torch.randn(1, requires_grad=True)
            RX = torch.tensor([
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]
            ], requires_grad=True)
            RY = torch.tensor([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ], requires_grad=True)
            RZ = torch.tensor([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ], requires_grad=True)
            R = torch.mm(RZ, RY).requires_grad_()
            R = torch.mm(R, RX).requires_grad_()
            R = R.mean().requires_grad_()
            R.backward()

            datapipe = sampler(self.subjects[0])

            print('ok')

            setattr(self, set, datapipe)





