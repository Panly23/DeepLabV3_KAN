# DeepLabV3_KAN
This project builds on the DeepLabv3 backbone by replacing conventional convolutional layers with the KAN architecture to enhance interpretability and, consequently, improve semantic understanding in autonomous-driving scenarios.

## Project Overview

This repository implements and documents the method proposed in the paper "Kolmogorov-Arnold Networks enhanced hybrid neural architecture for multi-scale semantic segmentation" ([Lingyun Pan, Yuan Jiang, Yancong Deng, Procedia Computer Science 266 (2025) 1171–1183, DOI: 10.1016/j.procs.2025.08.145](https://www.sciencedirect.com/science/article/pii/S1877050925024585). The proposed model, DeepLab_KAN, augments the DeepLabV3+ framework by replacing selected convolutional operations with Kolmogorov–Arnold Network (KAN) layers and replacing Batch Normalization with Layer Normalization to improve multi-scale segmentation performance and model interpretability.

Key reported results on Cityscapes: validation mIoU = 0.7013 (a 2.13% improvement over baseline DeepLabV3+), faster convergence, and improved boundary delineation.

## Recommended repository layout
```plaintext
├── Base_deeplabv3+.ipynb
├── KAN_deeplabv3+ (2).ipynb
├── README.md 
├── requirements.txt # Recommended dependencies
├── datasets/ 
│ └── cityscapes/
├── checkpoints/ # model checkpoints 
├── results/ # visualizations and metric logs
```

## Environment & Dependencies 

Use conda or virtualenv. Example with conda:
```plaintext
conda create -n kan_seg python=3.9 -y
conda activate kan_seg
pip install -r requirements.txt
```
Dependencies in `requirements.txt`:
```plaintext
torch>=1.12.0
torchvision
numpy
opencv-python
matplotlib
jupyterlab
albumentations
tqdm
scipy
Pillow
tensorboard
```
Ensure CUDA drivers are compatible with the installed PyTorch wheel if you use GPU training.


## Dataset: Cityscapes

The experiments reported in the paper use the Cityscapes dataset (fine annotations):

- 5,000 finely annotated images (train/val/test splits)

- 19 semantic classes (Cityscapes standard)

Suggested folder layout:
```plaintext
datasets/cityscapes/
  ├─ leftImg8bit/
  │   ├─ train/
  │   ├─ val/
  │   └─ test/
  └─ gtFine/
      ├─ train/
      ├─ val/
      └─ test/
```
Preprocessing & augmentation used in the paper:

- Random scaling, random cropping, horizontal flipping, color jittering (implemented with Albumentations).

- Input normalization using ImageNet mean and std when using a pre-trained ResNet backbone.

## Training & Reproduction Details (from the paper)

Model

- Backbone: ResNet-101 (pretrained)

- ASPP dilation rates: {6, 12, 18}

- Replace 3×3 convolutions in ASPP and the final conv layer with KAN-conv modules

- Replace BatchNorm with LayerNorm

Loss & Optimization

- Loss: Modified Focal Loss (γ = 3) to handle class imbalance

- Optimizer: SGD with Nesterov momentum (momentum = 0.9)

- Initial LR: 1e-4, polynomial decay schedule

- Early stopping: stop if validation metric does not improve for 10 epochs

Training remarks

- The paper reports experiments on a workstation with RTX 4080 SUPER. Adjust batch size and input resolution according to your GPU memory.

## Visualization
Visualization results obtained by using the DeepLabV3_KAN network to make predictions on some validation sets
