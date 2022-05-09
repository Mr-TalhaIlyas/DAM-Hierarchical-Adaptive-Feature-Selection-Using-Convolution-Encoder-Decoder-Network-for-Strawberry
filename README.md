[![CC BY 4.0][cc-by-shield]][cc-by]
<img alt="Keras" src="https://img.shields.io/badge/Keras%20-%23D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/> <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" /> [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMr-TalhaIlyas%2FDAM-Hierarchical-Adaptive-Feature-Selection-Using-Convolution-Encoder-Decoder-Network-for-Strawberry&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

![alt text](https://github.com/Mr-TalhaIlyas/DAM-Hierarchical-Adaptive-Feature-Selection-Using-Convolution-Encoder-Decoder-Network-for-Strawberr/blob/master/screens/img0.png)
# SEEK: A Framework of Superpixel Learning with CNN Features for Unsupervised Segmentation

In this repo I am uploading the code and necessary scripts for generating results described in our publication in  **_Frontiers in Plant Science_** journal.

You can access the [full paper here](https://www.frontiersin.org/articles/10.3389/fpls.2021.591333/full)
Dataset download from [Google Drive Link](https://drive.google.com/file/d/1qrNc-tgIHsUOXaIcMRL9M2cZRPGT-BRN/view?usp=sharing)

## Abstract

Autonomous harvesters can be used for the timely cultivation of high-value crops such as strawberries, where the robots have the capability to identify ripe and unripe crops. However, the real-time segmentation of strawberries in an unbridled farming environment is a challenging task due to fruit occlusion by multiple trusses, stems, and leaves. In this work, we propose a possible solution by constructing a dynamic feature selection mechanism for convolutional neural networks (CNN). The proposed building block namely a dense attention module (DAM) controls the flow of information between the convolutional encoder and decoder. DAM enables hierarchical adaptive feature fusion by exploiting both inter-channel and intra-channel relationships and can be easily integrated into any existing CNN to obtain category-specific feature maps. We validate our attention module through extensive ablation experiments. In addition, a dataset is collected from different strawberry farms and divided into four classes corresponding to different maturity levels of fruits and one is devoted to background. Quantitative analysis of the proposed method showed a 4.1% and 2.32% increase in mean intersection over union, over existing state-of-the-art semantic segmentation models and other attention modules respectively, while simultaneously retaining a processing speed of 53 frames per second.

### SS1K Dataset
Download from [Google Drive Link](https://drive.google.com/file/d/1qrNc-tgIHsUOXaIcMRL9M2cZRPGT-BRN/view?usp=sharing)

The dataset consist of following four classes,

![alt text](https://github.com/Mr-TalhaIlyas/DAM-Hierarchical-Adaptive-Feature-Selection-Using-Convolution-Encoder-Decoder-Network-for-Strawberr/blob/master/screens/img8.png)

The instance distribution of each class in the SS1K dataset is shown below

![alt text](https://github.com/Mr-TalhaIlyas/DAM-Hierarchical-Adaptive-Feature-Selection-Using-Convolution-Encoder-Decoder-Network-for-Strawberr/blob/master/screens/img9.png)

For details [visit here.](https://www.frontiersin.org/articles/10.3389/fpls.2021.591333/full)


### Network Architecture
__________________
![alt text](https://github.com/Mr-TalhaIlyas/DAM-Hierarchical-Adaptive-Feature-Selection-Using-Convolution-Encoder-Decoder-Network-for-Strawberr/blob/master/screens/img2.png)

![alt text](https://github.com/Mr-TalhaIlyas/DAM-Hierarchical-Adaptive-Feature-Selection-Using-Convolution-Encoder-Decoder-Network-for-Strawberr/blob/master/screens/img3.png)

### Segmentations Grad-CAM
__________________
For the qualitative analysis, we apply the Grad-CAM to show the effects of DAM. Grad-CAM is a gradientbased visualization method, which tries to explain the reasoning
behind the decisions made by the DCNNs. It was mainly
proposed for classification networks. We propose a modified
version of Grad-CAM to evaluate the results of the semantic
segmentation model making it into Segmentation Grad-CAM
(SGC). If {A<sup>k</sup>}<sup>K</sup><sub>k=1</sub> represents the feature map of a selected layer
with K feature maps then Grad-CAM calculates the heatmaps by
taking the gradient of yc(logit for a given class) w.r.t to all N pixels
(indexed by u, v), in all feature maps of {A<sup>k</sup>}<sup>K</sup><sub>k=1</sub>. But in the case
of segmentation models, instead of yc (a single value), for each
class we have y<sub>ij</sub> c (a whole feature map). In this case, the gradients
are computed by taking the mean of all M pixels (indexed by i, j)
in the feature map of class ‘c.’ Finally, the weighing vector ac k is
calculated as;

![alt text](https://github.com/Mr-TalhaIlyas/DAM-Hierarchical-Adaptive-Feature-Selection-Using-Convolution-Encoder-Decoder-Network-for-Strawberr/blob/master/screens/img.png)

![alt text](https://github.com/Mr-TalhaIlyas/DAM-Hierarchical-Adaptive-Feature-Selection-Using-Convolution-Encoder-Decoder-Network-for-Strawberr/blob/master/screens/img4.png)

For details [visit here.](https://www.frontiersin.org/articles/10.3389/fpls.2021.591333/full)


### Visual Reults
__________________
![alt text](https://github.com/Mr-TalhaIlyas/DAM-Hierarchical-Adaptive-Feature-Selection-Using-Convolution-Encoder-Decoder-Network-for-Strawberr/blob/master/screens/img6.png)

### Quantitative Results
__________________
![alt text](https://github.com/Mr-TalhaIlyas/DAM-Hierarchical-Adaptive-Feature-Selection-Using-Convolution-Encoder-Decoder-Network-for-Strawberr/blob/master/screens/img5.png)

### Performance on Different Platforms
__________________
![alt text](https://github.com/Mr-TalhaIlyas/DAM-Hierarchical-Adaptive-Feature-Selection-Using-Convolution-Encoder-Decoder-Network-for-Strawberr/blob/master/screens/img7.png)

#### **_Citation_**
If you like to use our work please consider citing us;
```
@article{ilyas2021dam,
  title={DAM: Hierarchical Adaptive Feature Selection using Convolution Encoder Decoder Network for Strawberry Segmentation},
  author={Ilyas, Talha and Umraiz, Muhammad and Khan, Abbas and Kim, Hyongsuk},
  journal={Frontiers in Plant Science},
  volume={12},
  pages={189},
  year={2021},
  publisher={Frontiers}
```
OR via [google scholar](https://scholar.google.com/scholar?cluster=11817317065915208112&hl=en&oi=scholarr)


This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
