from tester_UNet import *



class TesterGAN(Tester):

    def __init__(self, *args, **kwargs
                 ):
        super(TesterGAN, self).__init__(*args, **kwargs)

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

                    logits = model.g_B2A(input_tensor)
                    # labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)

                    # labels = torch.squeeze(labels, dim=0)

                    aggregator.add_batch(logits, locations)

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

            image = tio.ScalarImage(tensor=output_tensor, affine=cur_subj['image'].affine)
            image.save(os.path.join(self.save_root_path, cur_name))



if __name__ == '__main__':

    num_workers = 1
    batch_size = 1
    in_channels = 1

    gpus = 1

    d_lr: float = 2e-4
    g_lr: float = 2e-4
    epochs = 200
    epoch_decay = epochs // 2
    img_sz = patch_size

    path_B = 'dataset3D/Test/B'



    patch_size = 208
    n_class = 2
    label_subdir = 'labels'


    v_num = None


    dataset = PatchDatset(
        dataset_dir=path_B,
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

    save_root_path = os.path.join(os.path.dirname(path_B), 'pred')


    ckpt_path = get_latest_state(log_root='logs/CycleGAN',
                                 v_num=v_num
                                 )

    print(f'loading model from {ckpt_path}')
    checkpoint = torch.load(ckpt_path)
    state_dict = checkpoint["state_dict"]

    model.load_state_dict(state_dict, strict=True)


    tester = TesterGAN(model=model, dataset=dataset, patch_size=patch_size,
                    batch_size=2, flatten_y=False, save_root_path=save_root_path)

    tester.inference_patch(test_set=dataset.test_set, batch_size=1,
                           patch_overlap=patch_size//2,
                           )
                           
                           
                           
                           
                           
