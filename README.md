# DeepLabV3_KAN
This project builds on the DeepLabv3 backbone by replacing conventional convolutional layers with the KAN architecture to enhance interpretability and, consequently, improve semantic understanding in autonomous-driving scenarios.

Project Overview

This repository implements and documents the method proposed in the paper "Kolmogorov-Arnold Networks enhanced hybrid neural architecture for multi-scale semantic segmentation" (Lingyun Pan, Yuan Jiang, Yancong Deng, Procedia Computer Science 266 (2025) 1171–1183, DOI: 10.1016/j.procs.2025.08.145). The proposed model, DeepLab_KAN, augments the DeepLabV3+ framework by replacing selected convolutional operations with Kolmogorov–Arnold Network (KAN) layers and replacing Batch Normalization with Layer Normalization to improve multi-scale segmentation performance and model interpretability.

Key reported results on Cityscapes: validation mIoU = 0.7013 (a 2.13% improvement over baseline DeepLabV3+), faster convergence, and improved boundary delineation.
