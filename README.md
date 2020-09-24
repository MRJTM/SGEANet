# SGDANet

### 1.Introduction
Official Implement for the Journal paper《Synthetic Guided Domain Adaptive and Edge Aware 
Network for Crowd Counting》<br> 
The journal is《Image and Vision Computing》. 
In this project, you train a CSRNet or our SGEANet on ShanghaiTech dataset.
<img src="./images/overview1.png" width = "300" height = "200" alt="图片名称" align=center />


### 2.Project Organization
The folders are organized as follows:
* data: dataset processing code, image list file, the soft link of dataset folders.
* experiment: config file for training, testing.
* images: images to show on this readme.md
* main: two main scripts, train.py and val.py.
* src: all other codes 

### 3.Requirement
* pytorch=1.1.0
* torchvision
* progressbar
* visdom

### 4.Tutorial
#### (1) prepare dataset

#### (2) training real and synthetic baseline model

#### (3) training SGEANet

#### (4) evaluating

### 5.Citation
If you use this code for your research, please cite our paper:

<img src="./images/title.png" width = "300" height = "200" alt="图片名称" align=center />

