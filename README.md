# MARS: Paying more attention to visual attributes for text-based person search (ACM TOMM)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mars-paying-more-attention-to-visual/nlp-based-person-retrival-on-cuhk-pedes)](https://paperswithcode.com/sota/nlp-based-person-retrival-on-cuhk-pedes?p=mars-paying-more-attention-to-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mars-paying-more-attention-to-visual/text-based-person-retrieval-on-rstpreid-1)](https://paperswithcode.com/sota/text-based-person-retrieval-on-rstpreid-1?p=mars-paying-more-attention-to-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mars-paying-more-attention-to-visual/text-based-person-retrieval-on-icfg-pedes)](https://paperswithcode.com/sota/text-based-person-retrieval-on-icfg-pedes?p=mars-paying-more-attention-to-visual)

This is the official PyTorch implementation of the paper [MARS: Paying more attention to visual attributes for text-based person search](https://dl.acm.org/doi/10.1145/3721482). 
This repository supports training and evaluation on three text-based person search benchmarks: CUHK-PEDES, ICFG-PEDES and RSTPReid.

![](images/architecture.png)

## Usage
### Enviroment preparation
```bash
conda create -n MARS python==3.9
conda activate MARS

pip3 install torch torchvision torchaudio
pip install transformers==4.8.1
pip install timm==0.4.9
pip install spacy
python -m spacy download en
pip install einops
conda install anaconda::ruamel_yaml
pip install matplotlib
```

### Prepare Datasets

For datasets preparation and download, please refer to [RaSA](https://github.com/Flame-Chasers/RaSa/).

### Pretrained Checkpoint
- Please download the [pretrained ALBEF Checkpoint](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth).

### Training
Inside the shell folder there are the script for each training.

To train our model just choose a dataset and do:
```shell
# 1. Training on CUHK-PEDES
bash shell/cuhk_train.sh

# 2. Training on ICFG-PEDES
bash shell/icfg_train.sh

# 3. Training on RSTPReid
bash shell/rstp_train.sh
```

Before training, please update dataset location inside each ```.yaml``` file.


### Testing

Inside the shell folder there are the script to test each model.

```shell
# 1. Testing on CUHK-PEDES
bash shell/cuhk-eval.sh

# 2. Testing on ICFG-PEDES
bash shell/icfg-eval.sh

# 3. Testing on RSTPReid
bash shell/rstp-eval.sh
```

Before testing, please update the checkpoint location inside each ```.sh``` file.

## MARS Performance on Three Text-based Person Search Benchmarks
### CUHK-PEDES dataset

|     Method      |  Rank-1   |  Rank-5   |  Rank-10  |    mAP    |
| :-------------: | :-------: | :-------: | :-------: | :-------: |
|     CMPM/C      |   49.37   |   71.69   |   79.27   |     -     |
|      ViTAA      |   55.97   |   75.84   |   83.52   |     -     |
|      DSSL       |   59.98   |   80.41   |   87.56   |     -     |
|       SAF       |   64.13   |   82.62   |   88.40   |   58.61   |
|      LGUR       |   65.25   |   83.12   |   89.00   |     -     |
|       IVT       |   65.59   |   83.11   |   89.21   |     -     |
|      CFine      |   69.57   |   85.93   |   91.15   |     -     |
|      ALBEF      |   60.28   |   79.52   |   86.34   |   56.67   |
|    **RaSa**     |   76.51   |   90.29   |   94.25   |   69.38   |
| **MARS (ours)** | **77.62** | **90.63** | **94.27** | **71.41** |

[Model for CUHK-PEDES](https://univpr-my.sharepoint.com/:u:/g/personal/alex_ergasti_unipr_it/Eb6kU2z0UXFCqPUkJay4SdEBr958rd0sO8n1SX8MaILCeA?e=6dOApg)

### ICFG-PEDES dataset

|     Method      |  Rank-1   |  Rank-5   |  Rank-10  |    mAP    |
| :-------------: | :-------: | :-------: | :-------: | :-------: |
|     CMPM/C      |   43.51   |   65.44   |   74.26   |     -     |
|      SSAN       |   54.23   |   72.63   |   79.53   |     -     |
|       SAF       |   54.86   |   72.13   |   79.13   |   32.76   |
|       IVT       |   56.04   |   73.60   |   80.22   |     -     |
|      CFine      |   60.83   |   76.55   |   82.42   |     -     |
|      ALBEF      |   34.46   |   52.32   |   60.40   |   19.62   |
|    **RaSa**     |   65.28   |   80.40   |   85.12   |   41.29   |
| **MARS (ours)** | **67.60** | **81.47** | **85.79** | **44.93** |

[Model for ICFG-PEDES](https://univpr-my.sharepoint.com/:u:/g/personal/alex_ergasti_unipr_it/EdSC52AI76VEiAcQzD92HNcBPCxSutaW0HuwkglImhi5gA?e=ERW1XO)

### RSTPReid dataset

|     Method      |  Rank-1   |  Rank-5   |  Rank-10  |    mAP    |
| :-------------: | :-------: | :-------: | :-------: | :-------: |
|      DSSL       |   32.43   |   55.08   |   63.19   |     -     |
|      SSAN       |   43.50   |   67.80   |   77.15   |     -     |
|       SAF       |   44.05   |   67.30   |   76.25   |   36.81   |
|       IVT       |   46.70   |   70.00   |   78.80   |     -     |
|      CFine      |   50.55   |   72.50   |   81.60   |     -     |
|      ALBEF      |   50.10   |   73.70   |   82.10   |   41.73   |
|    **RaSa**     |   66.90   |   86.50   | **91.35** |   52.31   |
| **MARS (ours)** | **67.55** | **86.55** | **91.35** | **52.92** |

[Model for RSTPReid](https://univpr-my.sharepoint.com/:u:/g/personal/alex_ergasti_unipr_it/EddGS5Y5Io5HmPZFkw3HnAsBbstGUFi7xaqTMQhpLDcNmA?e=3kZeSG)

### Ablation study

#### Gradcam
![](images/gradcam.png)

Visual comparison of cross attention maps generated by the baseline model (top) and our model (bottom) using Grad-CAM. The attention maps illustrate the cross-modal encoder focus on different regions corresponding to individual words in the attribute chunks. The proposed attribute loss leads to more consistent and accurate attention distribution across words.


#### Top 10 rank
![](images/topk.png)

Overview of comparison between top 10 predictions of baseline and our model. Predicted images are ranked from left (i.e., position 1) to the right (i.e., position 10). Our model outperforms the baseline in several pairs, i.e., **a,b,c,d**. In pair **c** it is possible to observe how all predictions are with a bike in it, while this is not true in the baseline. Furthermore, even if in pair **e** our model does not predict the second position correctly, it is easy to observe how a higher mAP is achieve by providing 3 correct matches in top 10 positions compared to 2 correct matches in top 10 of the baseline. Lastly, in pair **f** our model is not able to predict any correct image due to the vagueness of the caption, but is still retrieving images closely related to the text.


## Acknowledgments
The implementation of MARS relies on resources from [RaSA](https://github.com/Flame-Chasers/RaSa/), [Huggingface Transformers](https://github.com/huggingface/transformers), and [timm](https://github.com/rwightman/pytorch-image-models/tree/master/timm). We sincerely appreciate the original authors for their open-sourcing.


## Citation
If you find this code useful for your research, please cite our paper.

```tex
@article{10.1145/3721482,
author = {Ergasti, Alex and Fontanini, Tomaso and Ferrari, Claudio and Bertozzi, Massimo and Prati, Andrea},
title = {Mars: Paying More Attention to Visual Attributes for Text-based Person Search},
year = {2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1551-6857},
url = {https://doi.org/10.1145/3721482},
doi = {10.1145/3721482},
abstract = {Text-based person search (TBPS) is a problem that gained significant interest within the research community. The task is that of retrieving one or more images of a specific individual based on a textual description. The multi-modal nature of the task requires learning representations that bridge text and image data within a shared latent space. Existing TBPS systems face two major challenges. One is defined as inter-identity noise that is due to the inherent vagueness and imprecision of text descriptions and it indicates how descriptions of visual attributes can be generally associated to different people; the other is the intra-identity variations, which are all those nuisances e.g., pose, illumination, that can alter the visual appearance of the same textual attributes for a given subject. To address these issues, this paper presents a novel TBPS architecture named MARS (Mae-Attribute-Relation-Sensitive), which enhances current state-of-the-art models by introducing two key components: a Visual Reconstruction Loss and an Attribute Loss. The former employs a Masked AutoEncoder trained to reconstruct randomly masked image patches with the aid of the textual description. In doing so the model is encouraged to learn more expressive representations and textual-visual relations in the latent space. The Attribute Loss, instead, balances the contribution of different types of attributes, defined as adjective-noun chunks of text. This loss ensures that every attribute is taken into consideration in the person retrieval process. Extensive experiments on three commonly used datasets, namely CUHK-PEDES, ICFG-PEDES, and RSTPReid, report performance improvements, with significant gains in the mean Average Precision (mAP) metric w.r.t. the current state of the art. Code will be available at .},
note = {Just Accepted},
journal = {ACM Trans. Multimedia Comput. Commun. Appl.},
month = mar,
keywords = {Multi-modal learning, person retrieval, re-identification}
}
```
