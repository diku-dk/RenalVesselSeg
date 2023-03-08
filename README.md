# Implementation of Paper: Extremely weakly-supervised blood vessel segmentation with physiologically based synthesis and domain adaptation


---

##  Backbone model: CycSeg: 3D CycleGAN with an addtional segmentation branch using U-Net


---
## Quick Start
#### Installation

```
# From GitHub
git clone https://github.com/miccai2023anony/RenalVesselSeg

# install pytorch related packages (you might need to change cuda version according to your environment)
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia 

# install other packages and path setup
pip install -e RenalVesselSeg
```

Note that only installing ```requirements.txt``` is not enough since ```package_dir``` in ```setup.py``` is also necessary
to do correct import in the python files


---


##  Physiologically-based simulation of renal blood vessels

[//]: # (Please follow for detailed explanation of renal blood vessels reconstruction. )

[//]: # (You will need to consult its authors for the code of constructing the tree structure or output files,)

[//]: # (which will probably be made available upon acceptance.)

The code for the reconstructing vascular tree (based on Global Constructive Optimization) 
as well as the synthesized results will be made available upon acceptance of the previous paper 
(attached in appendix currently due to double-blind requirement). 

#### Here this GitHub repo will include all the deep learning part, i.e., code for constructing, training and applying the deep learning models.

---
### Models
*  2D and 3D UNet models are available at ```src/model.py``` and ```src/model3D.py``` 

*  2D and 3D CycleGAN models are available at ```src/cycle_GAN/cycleGAN_PL.py``` and ```src/cycle_GAN/CycleGAN3D.py```

*  The final CycSeg models (CycleGAN with additional segmentation branch) at ```src/cycle_GAN/CycleGAN_w_UNet2D.py``` and ```src/cycle_GAN/CycleGAN_w_UNet.py```

---

### Preparing the data
In order to train the generative model, a set of synthetic image-label pairs and 
unlabeld real images must be stored in ```data_folder``` under the following structure:

```
./data_folder/

|- Train/
|--- A/
|------ images/
|--------- image1.nii.gz
|--------- image5.nii.gz

|--- B/
|------ images/
|--------- image1.nii.gz
|--------- image2.nii.gz
|------ labels/
|--------- image1.nii.gz
|--------- image2.nii.gz


|- Test/
|--- A/
|------ images/
|--------- image1.nii.gz
|--------- image5.nii.gz

```

Note that  domain A is the target domain without any ground truth labels while domain B is the source domain with 
synthesized label maps, and the corresponding synthesized images (by simple random noises). 
Therefore, subfolder ``A`` will only contain ```images``` 
while subfolders in ``B`` must have both ```images``` and ```labels```. 
During inference, model will directly segment over scans from ```Test/A/images```.  

### File formatting
All 3D images must be stored in the ``.nii``/```.nii.gz``` format while 2D images must be stored in ```.tif```.

Currently, it only works for gray-scale images with a binary segmentation task. 
Therefore, the image should either not have an additional dimension or must have an additional dimension of length 1.
label at a given voxel should be an integer representing the class at the given position, 
with foreground class denoted '1' and background class denoted '0'.



---
## Training
The model can be trained as follows for 3D and 2D respectively:

```
cd RenalVesselSeg/src/cycle_GAN
python train3D.py
```
```
cd RenalVesselSeg/src/cycle_GAN
python train2D.py
```

---

We have not integrated ```argparse``` to the entry code file. 
So you will just need to go to the main file (```train3D.py``` or ```train2D.py```) 
to specify the root path (by default they are under ```./data_folder/```) or make any changes to the parameters, 
e.g., number of filters, network depth, patch size, weighting factors, etc. We will integrate this functionality in the future.

---

![](figs/training_pipeline.jpg)


### Saved models

Models will be saved under ```src/cycle_GAN/logs/CycleGAN``` with a unique version number each time.

## Inference
During inference, all the CycleGAN components are discarded, 
while the real scan (domain A) is directly passed to segmenter to output segmentation maps. 
However, note that you need to load the complete CycSeg model first while only calling the UNet branch later.

For 2D, run
```
python 2Dpatch_predict.py 
```

For 3D, run
```
python tester_UNet.py 
```

This will create segmentation output of the testing data of domain A at ```Test/A/images``` 
and will be saved at ```Test/A/pred```

Please note that you will need to specify ```v_num``` (version number) generated during training 
to load the corresponding model, otherwise it will load the latest one under ```src/cycle_GAN/logs/CycleGAN```.

