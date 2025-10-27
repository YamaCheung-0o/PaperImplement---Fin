# BFENet (Bilateral Feature Enhancement Network)

A PyTorch reimplementation of the bilateral dual-stream interaction CNN for patient-level multi-label ophthalmic disease classification on OIA-ODIR.

- Dual-stream shared-weight ResNet backbone (left/right eyes)
- Multiscale module with dilated convolutions (rates 1, 2, 4)
- Feature Enhancement module via global-local attention interaction
- Classifier: 8192 -> 4096 -> 512 -> 8 with BCE loss
- Poly learning rate decay
- Metrics: Cohen's Kappa, F1, AUC, Final-score (avg of the three)

## Project structure

```
bfenet/
  models/
  datasets/
  utils/
configs/
train.py
eval.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset layout (OIA-ODIR)

```
OIA-ODIR/
  train/
    left_images/
    right_images/
    labels.csv
  off_site_test/
  on_site_test/
```

## Train

```bash
python train.py --config configs/config.yaml --data_root /path/to/OIA-ODIR --backbone resnet50 --epochs 50 --batch_size 16
```

## Evaluate

```bash
python eval.py --config configs/config.yaml --data_root /path/to/OIA-ODIR --checkpoint runs/best.ckpt --split off_site_test
```
## Citation
If you use this code in your research, please cite the original paper：

```bash
Xingyuan Ou, Li Gao, Xiongwen Quan, Han Zhang, Jinglong Yang, Wei Li,
BFENet: A two-stream interaction CNN method for multi-label ophthalmic diseases classification with bilateral fundus images,
Computer Methods and Programs in Biomedicine,
Volume 219,
2022,
106739,
ISSN 0169-2607,
https://doi.org/10.1016/j.cmpb.2022.106739.
(https://www.sciencedirect.com/science/article/pii/S0169260722001250)
Abstract: Background and objective
Early fundus screening and timely treatment of ophthalmology diseases can effectively prevent blindness. Previous studies just focus on fundus images of single eye without utilizing the useful relevant information of the left and right eyes. While clinical ophthalmologists usually use binocular fundus images to help ocular disease diagnosis. Besides, previous works usually target only one ocular diseases at a time. Considering the importance of patient-level bilateral eye diagnosis and multi-label ophthalmic diseases classification, we propose a bilateral feature enhancement network (BFENet) to address the above two problems.
Methods
We propose a two-stream interactive CNN architecture for multi-label ophthalmic diseases classification with bilateral fundus images. Firstly, we design a feature enhancement module, which makes use of the interaction between bilateral fundus images to strengthen the extracted feature information. Specifically, attention mechanism is used to learn the interdependence between local and global information in the designed interactive architecture for two-stream, which leads to the reweighting of these features, and recover more details. In order to capture more disease characteristics, we further design a novel multiscale module, which enriches the feature maps by superimposing feature information of different resolutions images extracted through dilated convolution.
Results
In the off-site set, the Kappa, F1, AUC and Final score are 0.535, 0.892, 0.912 and 0.780, respectively. In the on-site set, the Kappa, F1, AUC and Final score are 0.513, 0.886, 0.903 and 0.767 respectively. Comparing with existing methods, BFENet achieves the best classification performance.
Conclusions
Comprehensive experiments are conducted to demonstrate the effectiveness of this proposed model. Besides, our method can be extended to similar tasks where the correlation between different images is important.
Keywords: Ocular disease classification; Feature enhancement; Patient-level diagnosis; Multi-label; Convolutional neural network
```
