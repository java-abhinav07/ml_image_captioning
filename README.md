## Image Captioning using Multimodal Semantic Similarity Loss

Image Captioning is a widely studied research problem with a wide variety of applications. We propose to make the traditional image captioning baseline achieve a higher accuracy by a multimodal semantic space loss. Concretely, we map the visual feature space to the same feature space as the word embedding and formulate a similarity objective while jointly optimizing the binary cross entropy. This ensures a tight coupling between the two semantic spaces; namely word embedding and image features.

### Prerequisites
---
1. Git Clone this project<br>
```git clone https://github.com/java-abhinav07/ml_image_captioning.git ```<br>

2. Install Dependencies (inside a virtual environment)<br>
```pip3 install -r requirements.txt```<br>

### Train
---
1. Configure the parameters in ```train.py``` (WIP: argparser needs to be added)<br>
```python3 train.py```<br>

### Test
---
1. Having trained the model you may test it using<br>
``` python3 test.py```<br>


### Dataset
---
Currently the data loader has been configured for Flickr8K dataset only. Feel free to use the COCO data preparation scripts to train with a larger dataset<br>
Flickr8k can be downloaded from:  https://www.kaggle.com/hsankesara/flickr-image-dataset

### Acknowledgments
---
Much of this code has been taken from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
