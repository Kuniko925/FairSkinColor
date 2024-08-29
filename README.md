# FairSkinColor

## Abstract
Fairness is a crucial aspect of Trustworthy ML. In prior group fairness-oriented research, the intricacies of image-based data have frequently been overlooked in favour of categorical features such as gender and ethnicity. This paper presents a new technique for verifying fairness for nuanced-sensitive attributes in ML for image classification tasks. To overcome the limitations of earlier work, we handle continuous numerical parameters like skin colour without classifying them, using statistical distance metrics. The paper demonstrates the benefits of this novel method when used for image classification tasks where skin tone is a sensitive characteristic.

## Evaluation Models
- VGG16
- Efficientnet b3
- ResNet50

## Evaluation Dataset

| Dataset  | Tasks Summary | Skin Type (n) | URL |
| ------------- | ------------- | ------------- | ------------- |
| HAM10000  | Binary Classification  | 1 | https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 |
| Celeb A  | Binary Classification  | 2 | https://www.kaggle.com/datasets/jessicali9530/celeba-dataset |
| UTKFace  | Binary Classification  | 4 | https://susanqq.github.io/UTKFace/ |


## Measure Skin Colour

![skin color measure](https://github.com/Kuniko925/FairSkinColor/blob/main/images/Fig%20core.png)

## Training Procedure

![learning procedure](https://github.com/Kuniko925/FairSkinColor/blob/main/images/Fig%20learning%20process.png)
